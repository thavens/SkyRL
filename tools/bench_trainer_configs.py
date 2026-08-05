"""Time one trainer backend configuration on a production-shaped batch.

Runs *inside* the Modal container (or any box with the jax extra). One process
per configuration, because `max_lora_adapters`, `train_micro_batch_size` and
`gradient_checkpointing` are all baked into the JIT graph at engine construction
and cannot be varied in place.

Reports, as a single JSON line on stdout prefixed with RESULT:, the warm
per-sequence forward_backward cost and the warm optim_step cost. Both matter and
they scale differently: forward_backward is dominated by the model pass, while
optim_step carries the O(max_lora_adapters) stacked-tree work in
AccumulatedGradients.get_mean and optax.global_norm (jax.py:233, jax.py:640).

Cold numbers are meaningless here -- the first batch per sequence-length bucket
pays ~170 s of XLA compilation (runbook §9.1), so every measurement below is
taken after an explicit warmup at the same shapes.

Usage:

    python tools/bench_trainer_configs.py \
        --base-model /models/Qwen/Qwen3.5-9B-Base \
        --backend-config '{"max_lora_adapters":5,...}' \
        --seq-len 2048 --batch-seqs 64 --adapters 3
"""

import argparse
import json
import statistics
import time

import jax

from skyrl.benchmarks.benchmark_engine import build_engine, make_fwd_bwd_input
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.engine import TinkerEngine


def gpu_mem_gib() -> dict:
    """Live and peak bytes from the JAX allocator, in GiB."""
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:
        return {}
    g = 1024**3
    return {
        "live_gib": round(stats.get("bytes_in_use", 0) / g, 2),
        "peak_gib": round(stats.get("peak_bytes_in_use", 0) / g, 2),
        "limit_gib": round(stats.get("bytes_limit", 0) / g, 2),
    }


def build(base_model: str, backend_config: dict, adapters: int) -> tuple[TinkerEngine, list[str]]:
    config = EngineConfig(
        base_model=base_model,
        backend="jax",
        backend_config=backend_config,
        # Set so the backend releases the reserved slot 0 exactly as production
        # does (engine.py:258-259). Never dialled: forward_backward and optim_step
        # do not touch the sampling path.
        external_inference_url="http://127.0.0.1:9",
        database_url="sqlite:///:memory:",
    )
    engine = build_engine(config, adapters)
    return engine, list(engine.backend.models.keys())


def timed(fn, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--backend-config", required=True, type=json.loads)
    ap.add_argument("--label", default="")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-seqs", type=int, default=64, help="sequences per forward_backward request")
    ap.add_argument("--adapters", type=int, default=3, help="concurrent models to create")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    result = {"label": args.label, "backend_config": args.backend_config,
              "seq_len": args.seq_len, "batch_seqs": args.batch_seqs, "adapters": args.adapters}

    t0 = time.perf_counter()
    engine, model_ids = build(args.base_model, args.backend_config, args.adapters)
    result["build_s"] = round(time.perf_counter() - t0, 1)

    # Deterministic pseudo-tokens; content is irrelevant to cost, only shape is.
    tokens = [[(i * 7919 + j) % 1000 + 1 for j in range(args.seq_len)] for i in range(args.batch_seqs)]
    fb_input = make_fwd_bwd_input(tokens)
    reqs = {"0": (model_ids[0], fb_input)}

    t0 = time.perf_counter()
    for _ in range(args.warmup):
        engine.process_forward_backward(reqs)
    result["warmup_s"] = round(time.perf_counter() - t0, 1)

    fb = timed(lambda: engine.process_forward_backward(reqs), args.steps)
    result["fb_s"] = round(statistics.median(fb), 3)
    result["fb_s_per_seq"] = round(statistics.median(fb) / args.batch_seqs, 4)
    result["fb_all_s"] = [round(x, 3) for x in fb]
    result["tokens_per_s"] = round(args.batch_seqs * args.seq_len / statistics.median(fb))

    # optim_step needs accumulated gradients present, so it follows a fresh
    # forward_backward each time -- the reset at the end of the step would
    # otherwise make every call after the first a no-op (jax.py:984).
    def one_optim():
        engine.process_forward_backward(reqs)
        engine.backend.optim_step(model_ids[0], types.OptimStepInput(
            adam_params=types.AdamParams(
                learning_rate=2e-5, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)))

    try:
        one_optim()  # warm the optimizer graph
        os_times = timed(one_optim, max(2, args.steps - 1))
        # subtract the forward_backward each sample includes
        result["optim_s"] = round(max(0.0, statistics.median(os_times) - statistics.median(fb)), 3)
    except Exception as exc:
        result["optim_error"] = f"{type(exc).__name__}: {exc}"

    result.update(gpu_mem_gib())
    print("RESULT:" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
