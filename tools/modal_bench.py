"""Sweep trainer backend configurations on one B200, in one container.

Why a separate app from modal_tinker_server: this needs no vLLM, no proxy and no
public URL, and the sweep is the expensive part of the analysis, so it should not
pay for the sampler image layer or a server boot.

Why one container for the whole sweep: each configuration re-loads 18 GB of base
weights and re-compiles the XLA graph, but a *container* cold start on top of
that is pure waste. The sweep therefore runs configurations as sequential
subprocesses inside a single function call. One process each, because
max_lora_adapters / train_micro_batch_size / gradient_checkpointing are baked
into the JIT graph at construction.

The trainer is capped at the same XLA_PYTHON_CLIENT_MEM_FRACTION production gives
it (0.45), even though nothing else is on the card here. Benchmarking against the
full 192 GB would happily validate a configuration that OOMs the moment the
sampler is put back beside it.

    modal run tools/modal_bench.py::sweep
    modal run tools/modal_bench.py::sweep --only slots5_mb4      # single config
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "skyrl-tinker-bench"
GPU_SPEC = os.environ.get("SKYRL_MODAL_GPU", "B200")
REPO = "/root/SkyRL"
MODEL_DIR = "/models/Qwen/Qwen3.5-9B-Base"
JAX_CACHE = "/cache/jax_compilation_cache"

# Matches the production trainer's share of the shared card (modal_tinker_server.py).
TRAINER_MEM_FRACTION = "0.45"

# Production geometry, from the injecagent_gpt5nano rollouts: 256 sequences per
# step, mean 2260 tokens. The sweep uses a uniform 2048 so that every
# configuration compiles exactly one sequence-length bucket -- at ~170 s per
# bucket (runbook §9.1) a ragged batch would spend the entire budget in XLA.
# Ragged replay is left to the real-run A/B, which is the actual proof.
SEQ_LEN = 2048
BATCH_SEQS = 32
ADAPTERS = 3

BASE_CFG = {
    "max_lora_rank": 64,
    "tensor_parallel_size": 1,
    "fully_sharded_data_parallel_size": 1,
    "train_bucket_seq_len_per_micro_batch": True,
    "loss_chunk_size": 128,
    "gradient_checkpointing": True,
    "train_micro_batch_size": 1,
    "max_lora_adapters": 8,
}


def cfg(**over) -> dict:
    return {**BASE_CFG, **over}


# Ordered by information-per-dollar: the baseline and the two single-variable
# comparisons come first, so a truncated sweep still answers the main question.
SWEEP: list[tuple[str, dict]] = [
    ("prod8", cfg()),                                                   # what ran in production
    ("slots5", cfg(max_lora_adapters=5)),                               # isolate slot count
    ("slots5_mb4", cfg(max_lora_adapters=5, train_micro_batch_size=4)),
    ("slots5_mb2", cfg(max_lora_adapters=5, train_micro_batch_size=2)),
    ("slots5_mb8", cfg(max_lora_adapters=5, train_micro_batch_size=8)),
    ("slots5_mb4_nockpt", cfg(max_lora_adapters=5, train_micro_batch_size=4, gradient_checkpointing=False)),
    ("slots5_mb4_chunk256", cfg(max_lora_adapters=5, train_micro_batch_size=4, loss_chunk_size=256)),
    ("slots4_mb4", cfg(max_lora_adapters=4, train_micro_batch_size=4)),
    # Worst-case-bucket probe set: run these at --seq-len 4096 to find which
    # micro-batch size still fits when a micro-batch is filled with the longest
    # sequences the production distribution actually contains.
    ("slots5_mb1_chunk256", cfg(max_lora_adapters=5, train_micro_batch_size=1, loss_chunk_size=256)),
    ("slots5_mb2_chunk256", cfg(max_lora_adapters=5, train_micro_batch_size=2, loss_chunk_size=256)),
]

models_vol = modal.Volume.from_name("skyrl-tinker-models", create_if_missing=True)
cache_vol = modal.Volume.from_name("skyrl-tinker-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("skyrl-tinker-bench-results", create_if_missing=True)

# Same base as the server image, minus the vLLM venv: this never samples.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential", "ninja-build")
    .pip_install("uv==0.9.26")
    .env({"UV_LINK_MODE": "copy", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(
        Path(__file__).resolve().parent.parent,
        remote_path=REPO,
        ignore=[
            "**/.venv", "**/.venv-*", "**/.git", "**/logs",
            "**/*.pyc", "**/__pycache__", "**/*.egg-info",
            "**/.pytest_cache", "**/node_modules",
        ],
        copy=True,
    )
    .run_commands(f"cd {REPO} && uv sync --extra gpu --extra tinker --extra jax")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU_SPEC,
    volumes={"/models": models_vol, "/cache": cache_vol, "/results": results_vol},
    timeout=14400,
)
def sweep(only: str = "", smoke_only: bool = False, skip_smoke: bool = False,
          seq_len: int = SEQ_LEN, batch_seqs: int = BATCH_SEQS, tag: str = "sweep"):
    """Run each configuration in its own process and collect the RESULT lines.

    seq_len is a parameter because the headline sweep runs at 2048 (one bucket,
    one compile) but the *memory* question is decided at 4096: bucketing pads each
    micro-batch to its own longest sequence, so a micro-batch of 4x4096 allocates
    twice the activations of 4x2048, and the production length distribution
    reaches 4095. A configuration that is fastest at 2048 can OOM in production.
    """
    Path(JAX_CACHE).mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "VIRTUAL_ENV": f"{REPO}/.venv",
        "PATH": f"{REPO}/.venv/bin:" + os.environ["PATH"],
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": TRAINER_MEM_FRACTION,
        "JAX_COMPILATION_CACHE_DIR": JAX_CACHE,
    }

    def run(label, backend_cfg, seq_len, batch_seqs, adapters, steps, warmup, timeout):
        argv = [
            f"{REPO}/.venv/bin/python", f"{REPO}/tools/bench_trainer_configs.py",
            "--base-model", MODEL_DIR,
            "--backend-config", json.dumps(backend_cfg),
            "--label", label,
            "--seq-len", str(seq_len), "--batch-seqs", str(batch_seqs),
            "--adapters", str(adapters), "--steps", str(steps), "--warmup", str(warmup),
        ]
        print(f"\n{'=' * 70}\n[{label}] {json.dumps(backend_cfg)}\n{'=' * 70}", flush=True)
        try:
            proc = subprocess.run(argv, env=env, cwd=REPO, timeout=timeout,
                                  capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            print(f"[{label}] TIMEOUT after {timeout}s", flush=True)
            return {"label": label, "error": "timeout", "backend_config": backend_cfg}

        # Echo the child's output so a failure is diagnosable from `modal app logs`.
        sys.stdout.write(proc.stdout[-8000:])
        if proc.returncode != 0:
            sys.stdout.write(proc.stderr[-8000:])
            # OOM is an expected outcome for the aggressive configurations, not a bug.
            tail = (proc.stderr or "")[-400:].replace("\n", " ")
            print(f"[{label}] FAILED rc={proc.returncode}", flush=True)
            return {"label": label, "error": f"rc={proc.returncode}: {tail}", "backend_config": backend_cfg}

        for line in proc.stdout.splitlines():
            if line.startswith("RESULT:"):
                return json.loads(line[len("RESULT:"):])
        return {"label": label, "error": "no RESULT line", "backend_config": backend_cfg}

    # Cheap end-to-end validation of the harness before committing to the sweep:
    # tiny shapes, one adapter, so a broken script costs ~3 minutes not ~50.
    results = []
    if not skip_smoke:
        smoke = run("smoke", cfg(max_lora_adapters=2), seq_len=256, batch_seqs=2,
                    adapters=1, steps=1, warmup=1, timeout=1800)
        print("SMOKE: " + json.dumps(smoke), flush=True)
        if "error" in smoke:
            raise RuntimeError(f"smoke failed, aborting sweep: {smoke['error']}")
        if smoke_only:
            return {"smoke": smoke}
        results.append(smoke)

    todo = [(n, c) for n, c in SWEEP if not only or n in only.split(",")]
    for label, backend_cfg in todo:
        results.append(run(label, backend_cfg, seq_len, batch_seqs, ADAPTERS,
                           steps=3, warmup=1, timeout=2700))
        Path(f"/results/{tag}.json").write_text(json.dumps(results, indent=2))
        results_vol.commit()  # persist incrementally; the sweep may be cut short

    print("\nSWEEP RESULTS:\n" + json.dumps(results, indent=2), flush=True)
    return results
