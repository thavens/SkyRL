"""Export Tinker LoRA checkpoints off the Modal state Volume into HF/vLLM format.

Produces, per run, both artifacts:

    <out>/<run>/lora/      PEFT adapter   (adapter_config.json + adapter_model.safetensors)
    <out>/<run>/merged/    merged weights (base + adapter, sharded safetensors + tokenizer)

The adapter is already PEFT-format on the Volume: with the JAX backend and external
inference, save_weights_for_sampler writes a plain directory to
external_inference_lora_base/<model_id>_<checkpoint_id> rather than a tarball under
checkpoints_base (skyrl/tinker/engine.py:683-687). So `lora/` is a copy plus two
metadata repairs, not a conversion.

Two things the server leaves for us to fix in adapter_config.json:
  * `base_model_name_or_path` points at the *container* path (/models/Qwen/...),
    which does not exist anywhere else.
  * `target_modules` is null. vLLM infers targets from the tensor keys so it does
    not care, but `peft.PeftModel.from_pretrained` needs the list.

The merge is done by hand rather than via peft.merge_and_unload() so that it needs
neither a GPU nor transformers support for this architecture -- it is a streaming
pass over the base shards, adding scaling * (B @ A) to each targeted weight.

Usage:

    uv run --isolated --with torch --with safetensors \
        tools/export_tinker_checkpoints.py \
        --run-dir /path/to/logs/injecagent_gpt4omini/20260731_180030 \
        --out /path/to/rl-hammer-hardening/models
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

VOLUME = "skyrl-tinker-state"
MODAL = "/home/ajoe/.local/bin/modal"
DEFAULT_BASE = "/scr1/public_models/huggingface/Qwen/Qwen3.5-9B-Base"
BASE_REPO_NAME = "Qwen/Qwen3.5-9B-Base"

# Files that make the merged dir loadable: everything except the base weights,
# which we rewrite. `.cache` and the README are noise.
SIDECAR_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
]


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def pick_checkpoint(run_dir: Path, prefer: str) -> tuple[str, str]:
    """Return (model_id, checkpoint_name) from a run's checkpoints.jsonl.

    Prefers the post-loop "final" save (hardening/run.py:39); falls back to the
    newest periodic save, which is what exists if the run died early.
    """
    rows = [json.loads(line) for line in (run_dir / "checkpoints.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no checkpoints recorded in {run_dir}")

    sampler_rows = [r for r in rows if "sampler_path" in r]
    if not sampler_rows:
        raise SystemExit(f"no sampler checkpoints in {run_dir} -- nothing servable to export")

    chosen = next((r for r in sampler_rows if r["name"].endswith(prefer)), None)
    if chosen is None:
        chosen = sampler_rows[-1]
        log(f"WARNING: no '{prefer}' checkpoint; falling back to {chosen['name']!r} " f"(run may not have completed)")

    # sampler_path is tinker://<model_id>/<checkpoint_id>
    _, _, rest = chosen["sampler_path"].partition("tinker://")
    model_id, _, checkpoint_id = rest.partition("/")
    return model_id, checkpoint_id


def fetch_adapter(model_id: str, checkpoint_id: str, dest: Path) -> None:
    """Download the adapter directory to exactly `dest`.

    `modal volume get` appends the remote basename when the destination is an
    existing directory, and writes a *file* named after the destination when it is
    not -- neither of which lands the contents at `dest`. So stage into a scratch
    directory and move the one child up.
    """
    remote = f"lora_models/{model_id}_{checkpoint_id}"
    staging = dest.parent / f".{dest.name}.staging"
    for path in (dest, staging):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    staging.mkdir(parents=True)

    log(f"downloading {VOLUME}:{remote}")
    subprocess.run([MODAL, "volume", "get", VOLUME, remote, str(staging)], check=True)

    children = list(staging.iterdir())
    if len(children) != 1 or not children[0].is_dir():
        raise SystemExit(f"unexpected download layout under {staging}: {children}")
    children[0].rename(dest)
    staging.rmdir()

    if not (dest / "adapter_model.safetensors").exists():
        raise SystemExit(f"{dest} has no adapter_model.safetensors")


def repair_adapter_config(lora_dir: Path, keys: list[str]) -> dict:
    cfg_path = lora_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())

    # Leaf module names, e.g. "q_proj" from
    # base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight
    targets = sorted({k.rsplit(".lora_", 1)[0].rsplit(".", 1)[-1] for k in keys})
    cfg["target_modules"] = targets
    cfg["base_model_name_or_path"] = BASE_REPO_NAME
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    log(f"adapter_config.json: target_modules={targets}")
    return cfg


def merge(lora_dir: Path, base_dir: Path, out_dir: Path, cfg: dict) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    rank = cfg["r"]
    alpha = cfg["lora_alpha"]
    if cfg.get("use_rslora"):
        scaling = alpha / (rank**0.5)
    else:
        scaling = alpha / rank
    log(f"merge scaling = {alpha}/{rank} = {scaling}")

    # module prefix (base-model key, minus .weight) -> {"A": tensor, "B": tensor}
    pairs: dict[str, dict[str, "torch.Tensor"]] = {}
    with safe_open(lora_dir / "adapter_model.safetensors", "pt") as f:
        for key in f.keys():
            mod, ab = key.rsplit(".lora_", 1)
            base_key = mod[len("base_model.model.") :] + ".weight"
            pairs.setdefault(base_key, {})[ab.split(".")[0]] = f.get_tensor(key)

    incomplete = [m for m, v in pairs.items() if set(v) != {"A", "B"}]
    if incomplete:
        raise SystemExit(f"adapter has unpaired lora_A/lora_B for: {incomplete[:5]}")
    log(f"{len(pairs)} target modules to merge")

    index = json.loads((base_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    missing = [m for m in pairs if m not in weight_map]
    if missing:
        raise SystemExit(f"adapter targets weights absent from base: {missing[:5]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(set(weight_map.values()))
    merged_count = 0

    for shard in shards:
        src = base_dir / shard
        log(f"merging into {shard}")
        tensors = {}
        with safe_open(src, "pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if key in pairs:
                    a = pairs[key]["A"].to(torch.float32)  # [r, in]
                    b = pairs[key]["B"].to(torch.float32)  # [out, r]
                    delta = (b @ a) * scaling  # [out, in]
                    if delta.shape != t.shape:
                        raise SystemExit(
                            f"shape mismatch for {key}: base {tuple(t.shape)} " f"vs delta {tuple(delta.shape)}"
                        )
                    t = (t.to(torch.float32) + delta).to(t.dtype)
                    merged_count += 1
                tensors[key] = t
        save_file(tensors, out_dir / shard, metadata={"format": "pt"})
        del tensors

    if merged_count != len(pairs):
        raise SystemExit(f"merged {merged_count} weights but adapter had {len(pairs)}")
    log(f"merged {merged_count} weights")

    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    for name in SIDECAR_FILES:
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    log(f"merged model written to {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="a logs/<run>/<timestamp> dir")
    ap.add_argument("--out", type=Path, required=True, help="destination root")
    ap.add_argument("--base", type=Path, default=Path(DEFAULT_BASE))
    ap.add_argument("--checkpoint", default="final", help="checkpoint name suffix to prefer")
    ap.add_argument("--skip-merge", action="store_true", help="export the adapter only")
    args = ap.parse_args()

    run_name = args.run_dir.parent.name
    dest = args.out / run_name
    lora_dir = dest / "lora"

    model_id, checkpoint_id = pick_checkpoint(args.run_dir, args.checkpoint)
    log(f"{run_name}: model={model_id} checkpoint={checkpoint_id}")

    fetch_adapter(model_id, checkpoint_id, lora_dir)

    from safetensors import safe_open

    with safe_open(lora_dir / "adapter_model.safetensors", "pt") as f:
        keys = list(f.keys())
    cfg = repair_adapter_config(lora_dir, keys)

    (dest / "SOURCE.json").write_text(
        json.dumps(
            {
                "run": run_name,
                "run_dir": str(args.run_dir),
                "model_id": model_id,
                "checkpoint_id": checkpoint_id,
                "base_model": BASE_REPO_NAME,
                "volume": VOLUME,
            },
            indent=2,
        )
        + "\n"
    )

    if not args.skip_merge:
        merge(lora_dir, args.base, dest / "merged", cfg)

    log(f"done: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
