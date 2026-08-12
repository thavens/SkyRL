"""Import a HuggingFace PEFT LoRA adapter into a running skyrl-tx server.

Workflow:
  1. Connect to the running tinker server via the Tinker SDK.
  2. Create a LoRA model at the adapter's rank (server hardcodes alpha=32).
  3. Save a placeholder sampler checkpoint to materialize the artifact path.
  4. Overwrite that artifact with the adapter weights, padding any LoRA modules
     that the skyrl-tx Llama-3 model expects but the PEFT adapter omits
     (q/k/v/o for attn, gate/up/down for MLP). Missing slots get zeros so
     they're behavioral no-ops.

     Servers publish sampler adapters in one of two layouts, so which artifact
     the placeholder produced decides where the weights go. A jax backend with
     --external-inference-url writes a plain directory under
     --external-inference-lora-base and no tar.gz at all (see
     EngineConfig.publishes_sampler_adapter_in_place); everything else writes
     checkpoints_base/<model_id>/sampler_weights/<id>.tar.gz. Writing to the
     wrong one leaves the zeroed placeholder in place and the import silently
     has no effect, so this tool detects the layout instead of assuming.
  5. Pre-scale lora_B by (adapter_alpha / 32) to compensate for the server
     hardcoding alpha=32; final scaling at sampling time becomes
     (32 / rank) * (adapter_alpha / 32) = adapter_alpha / rank, matching PEFT.

After running this, sample from the adapter via:
    sampling_client = service_client.create_sampling_client(
        base_model="<base_model_path>",
        model_path="tinker://<model_id>/sampler_weights/v1",
    )
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import peft
import safetensors.numpy
import safetensors.torch
import tinker
import torch

from skyrl.tinker.config import EngineConfig
from skyrl.utils.storage import pack_and_upload, write_and_publish_dir

SERVER_HARDCODED_ALPHA = 32.0  # see skyrl/tinker/api.py: create_model

LLAMA31_8B_DIMS = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_layers": 32,
}


def expected_lora_shapes(d: dict, rank: int) -> dict[str, tuple[int, int]]:
    """Return {hf_key_suffix: (shape_A, shape_B)} for every LoRA module skyrl-tx
    expects in a Llama-3 model. Shapes match PEFT layout: A=(rank, in), B=(out, rank).
    """
    q_out = d["num_attention_heads"] * d["head_dim"]
    kv_out = d["num_key_value_heads"] * d["head_dim"]
    o_in = d["num_attention_heads"] * d["head_dim"]
    H = d["hidden_size"]
    inter = d["intermediate_size"]

    per_layer = {
        "self_attn.q_proj": ((rank, H), (q_out, rank)),
        "self_attn.k_proj": ((rank, H), (kv_out, rank)),
        "self_attn.v_proj": ((rank, H), (kv_out, rank)),
        "self_attn.o_proj": ((rank, o_in), (H, rank)),
        "mlp.gate_proj": ((rank, H), (inter, rank)),
        "mlp.up_proj": ((rank, H), (inter, rank)),
        "mlp.down_proj": ((rank, inter), (H, rank)),
    }
    out = {}
    for layer in range(d["num_layers"]):
        for module, (sa, sb) in per_layer.items():
            out[f"base_model.model.model.layers.{layer}.{module}.lora_A.weight"] = sa
            out[f"base_model.model.model.layers.{layer}.{module}.lora_B.weight"] = sb
    return out


def pad_and_rescale(
    src: dict[str, torch.Tensor],
    expected: dict[str, tuple[int, int]],
    adapter_alpha: float,
) -> dict[str, np.ndarray]:
    """Build the final tensor dict at the expected shapes. Missing keys → zeros.
    All lora_B weights are scaled by (adapter_alpha / SERVER_HARDCODED_ALPHA).
    """
    rescale = adapter_alpha / SERVER_HARDCODED_ALPHA
    out: dict[str, np.ndarray] = {}
    missing = 0
    for key, shape in expected.items():
        if key in src:
            t = src[key]
            if t.shape != torch.Size(shape):
                raise ValueError(f"Shape mismatch for {key}: adapter has {tuple(t.shape)}, " f"expected {shape}")
            arr = t.to(torch.float32).cpu().numpy()
        else:
            missing += 1
            arr = np.zeros(shape, dtype=np.float32)
        if "lora_B" in key:
            arr = arr * rescale
        # Store back as bfloat16-compatible float (safetensors numpy uses fp32/16/etc;
        # the skyrl-tx loader casts to param.dtype on load, so float32 is fine).
        out[key] = arr.astype(np.float32)
    print(f"  Padded {missing} missing LoRA tensors with zeros")
    return out


def write_adapter(
    tensors: dict[str, np.ndarray],
    adapter_alpha: float,
    rank: int,
    base_model_name: str,
    dest: Path,
    as_directory: bool,
) -> None:
    """Write the adapter to `dest`, as a plain directory or a tar.gz.

    Mirrors `skyrl.tx.utils.models.save_lora_checkpoint`: same choice of context manager,
    and the same `peft.LoraConfig` writer for adapter_config.json, so neither the archive
    layout nor the config contents can drift from what the server produces and vLLM reads.
    """
    ctx = write_and_publish_dir(dest) if as_directory else pack_and_upload(dest)
    with ctx as tmp_path:
        safetensors.numpy.save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
        peft.LoraConfig(base_model_name_or_path=base_model_name, r=rank, lora_alpha=adapter_alpha).save_pretrained(
            str(tmp_path)
        )
    print(f"  Wrote {'adapter directory' if as_directory else 'tarball'} -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8000")
    # Defaults from the environment so the key need not appear in argv, which is
    # world-readable via `ps` on a shared box. --api-key still works for local servers
    # that ignore it (the loopback deployments in runbook.md take any value).
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TINKER_API_KEY", "tml-dummy"),
        help="Defaults to $TINKER_API_KEY. Prefer the env var over the flag on shared hosts.",
    )
    parser.add_argument(
        "--base-model", required=True, help="The --base-model the server was launched with (path or HF id)."
    )
    parser.add_argument(
        "--adapter-dir",
        required=True,
        help="Path to a HF PEFT adapter directory (must contain adapter_model.safetensors and adapter_config.json).",
    )
    parser.add_argument("--checkpoint-id", default="v1")
    parser.add_argument(
        "--checkpoints-base", default="/tmp/skyrl_checkpoints", help="Must match the server's --checkpoints-base."
    )
    parser.add_argument(
        "--lora-base",
        default="/tmp/lora_models",
        help=(
            "Must match the server's --external-inference-lora-base. Only used when the "
            "server publishes sampler adapters in place (jax backend + external inference)."
        ),
    )
    parser.add_argument("--out", default=None, help="Optional file to write the resulting tinker:// URI.")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank = int(cfg["r"])
    adapter_alpha = float(cfg["lora_alpha"])
    base_name = cfg.get("base_model_name_or_path", args.base_model)
    print(f"Loaded adapter config: rank={rank}, alpha={adapter_alpha}, base={base_name}")
    print(f"Target modules: {cfg.get('target_modules')}")
    print(
        f"Will rescale lora_B by {adapter_alpha / SERVER_HARDCODED_ALPHA:.6f} to "
        f"compensate for server hardcoded alpha={SERVER_HARDCODED_ALPHA}"
    )

    print("\nLoading source adapter weights...")
    src = safetensors.torch.load_file(str(adapter_dir / "adapter_model.safetensors"))
    print(f"  Got {len(src)} tensors")

    expected = expected_lora_shapes(LLAMA31_8B_DIMS, rank)
    print(f"\nExpected tensors for Llama-3.1-8B at rank {rank}: {len(expected)}")

    padded = pad_and_rescale(src, expected, adapter_alpha)

    print("\nConnecting to tinker server...")
    os.environ.setdefault("TINKER_API_KEY", args.api_key)
    client = tinker.ServiceClient(base_url=args.server, api_key=args.api_key)
    training_client = client.create_lora_training_client(
        base_model=args.base_model,
        rank=rank,
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
    )
    model_id = training_client.model_id
    print(f"  Created model_id={model_id}")

    print(f"\nMaterializing placeholder sampler checkpoint name={args.checkpoint_id!r}...")
    # Use save_weights_for_sampler so we control the checkpoint name; the resulting
    # checkpoint_id == name == args.checkpoint_id (see skyrl/tinker/api.py:1002).
    training_client.save_weights_for_sampler(name=args.checkpoint_id).result()
    print("  Placeholder checkpoint registered in DB.")

    # Whichever artifact the placeholder actually produced tells us the server's layout.
    # Both paths come from EngineConfig so this tool cannot desync from the server's own
    # naming (config.py owns the single definition of each layout).
    layout = EngineConfig(
        base_model=args.base_model,
        checkpoints_base=args.checkpoints_base,
        external_inference_lora_base=args.lora_base,
    )
    tar_path = Path(str(layout.sampler_archive_path(model_id, args.checkpoint_id)))
    dir_path = layout.sampler_adapter_dir(model_id, args.checkpoint_id)

    if dir_path.exists():
        print(f"\nServer publishes adapters in place; overwriting {dir_path}...")
        write_adapter(padded, adapter_alpha, rank, base_name, dir_path, as_directory=True)
        artifact_path = dir_path
    elif tar_path.exists():
        print(f"\nOverwriting {tar_path} with padded adapter weights...")
        write_adapter(padded, adapter_alpha, rank, base_name, tar_path, as_directory=False)
        artifact_path = tar_path
    else:
        # Writing anyway would produce a file nothing reads, leaving the zeroed
        # placeholder live -- the import would look successful and do nothing.
        raise SystemExit(
            "Could not find the placeholder sampler checkpoint in either layout:\n"
            f"  directory: {dir_path}\n"
            f"  archive:   {tar_path}\n"
            "Check --lora-base / --checkpoints-base against the server's "
            "--external-inference-lora-base / --checkpoints-base."
        )

    tinker_uri = f"tinker://{model_id}/sampler_weights/{args.checkpoint_id}"
    print("\n✅ Done. Use this from clients:")
    print(f"    base_model={args.base_model!r}")
    print(f"    model_path={tinker_uri!r}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "checkpoint_id": args.checkpoint_id,
                    "tinker_uri": tinker_uri,
                    "base_model": args.base_model,
                    "rank": rank,
                    "alpha": adapter_alpha,
                    "artifact_path": str(artifact_path),
                },
                indent=2,
            )
        )
        print(f"  Wrote info to {args.out}")


if __name__ == "__main__":
    main()
