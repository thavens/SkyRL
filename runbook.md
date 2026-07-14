# Runbook: Qwen3.5 JAX Tinker + vLLM Local Servers

Launches two colocated services on the local 4× RTX-5000 Ada (32 GB, sm_89) box:

```text
GPU 0,1: JAX Tinker training server, port 8001
GPU 3:   vLLM sampling server,       port 8000
```

Clients point the Tinker SDK at `http://localhost:8001`; sampling is forwarded
to vLLM internally. §10 covers the different H100 box.

## 0. Conventions

Every bash block recomputes these locally so each block is copy-paste-standalone:

```bash
REPO=/scr1/michael/SkyRL                 # your SkyRL checkout
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
PG_BIN=$REPO/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin
```

- **`STATE_DIR`** holds all durable server state (pgdata, checkpoints, XLA
  dumps). It must resolve to the same path across launch, verify, clean, and
  restarts — otherwise the trainer points at a different pgdata/database. If
  the cluster was created on `/tmp` and `/storage_slow` later appears, re-run
  the §2 one-time setup on the new location. (`/storage_slow` itself is
  root-owned; use the per-user subdir.)
- **Sampler LoRA adapters live on tmpfs**: `--external-inference-lora-base`
  points at `/dev/shm/$USER/..._lora_models`, not `$STATE_DIR`. The JAX backend
  writes each sampler adapter there as a plain directory that vLLM loads in
  place (no tar/untar round-trip through the slow disk). Adapters are transient
  — needed only until vLLM loads them — but vLLM never unloads them, so the dir
  grows in RAM until a vLLM bounce: bounce vLLM periodically (§11) and `rm -rf`
  the `/dev/shm/...` dir on restart.
- **Model paths**: `/scr1/public_models/huggingface/Qwen/Qwen3.5-{2B,4B,9B}-Base`.
- If your checkout's `.venv` lacks the tinker extras (e.g. `ModuleNotFoundError:
  sqlalchemy`), swap `--active --no-sync` for `--isolated` in the trainer launch.

## 1. Stop Existing Servers

Do NOT stop PostgreSQL here — it is shared state, kept running across launches.
The dots in the patterns must be escaped: an unescaped `skyrl.tinker` also
matches the postgres cmdline (`skyrl_tinker_pg_data`) and kills the database.

```bash
pkill -f '[s]kyrl\.tinker' || true
pkill -f '[v]llm serve /scr1/public_models/huggingface/Qwen/Qwen3\.5-' || true

for i in $(seq 1 30); do
  pgrep -f '[s]kyrl\.tinker|[v]llm serve /scr1/public_models/huggingface/Qwen/Qwen3\.5-' >/dev/null || break
  sleep 1
done
```

## 2. PostgreSQL

Tinker uses PostgreSQL instead of SQLite: many parallel `asample` calls collide
with SQLite's writer queue during `save_weights`, leaving zombie `PENDING`
futures that block the SDK until its 7200 s timeout (`database is locked`).

We use the embedded Postgres from the `pgserver` PyPI package — **binaries
only**. Never run a Python script that imports `pgserver` against this pgdata
while Postgres is up: `pgserver` registers an `atexit` handler that SIGTERMs
the postmaster on script exit, even with `cleanup_mode=None`.

```text
PGDATA:  $STATE_DIR/skyrl_tinker_pg_data
SOCKET:  $STATE_DIR/skyrl_tinker_pg_data/.s.PGSQL.5432
DBNAME:  skyrl_tinker
PG_URL:  postgresql://postgres@/skyrl_tinker?host=$STATE_DIR/skyrl_tinker_pg_data
```

One-time setup (only if pgdata does not exist):

```bash
REPO=/scr1/michael/SkyRL
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
PG_BIN=$REPO/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

cd $REPO && uv pip install pgserver     # one-time

mkdir -p $STATE_DIR/skyrl_tinker_pg_data
$PG_BIN/initdb -D $STATE_DIR/skyrl_tinker_pg_data -U postgres --auth=trust
```

Start + create DB (idempotent — safe on every launch). Tinker auto-runs
`SQLModel.metadata.create_all` on startup, so no Alembic step is needed:

```bash
REPO=/scr1/michael/SkyRL
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
PG_BIN=$REPO/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

$PG_BIN/pg_ctl -D $STATE_DIR/skyrl_tinker_pg_data status >/dev/null 2>&1 || \
nohup setsid $PG_BIN/pg_ctl -D $STATE_DIR/skyrl_tinker_pg_data \
  -l $STATE_DIR/skyrl_tinker_pg_data/log \
  -o "-h '' -k $STATE_DIR/skyrl_tinker_pg_data" start \
  </dev/null >$STATE_DIR/pg_start.log 2>&1

$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -tAc "SELECT 1 FROM pg_database WHERE datname='skyrl_tinker'" | grep -q 1 \
  || $PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
       -c "CREATE DATABASE skyrl_tinker;"

ls $STATE_DIR/skyrl_tinker_pg_data/.s.PGSQL.5432 && echo pg_ok
```

## 3. Launch Sampling Server (vLLM)

Set the model-specific variables, then run the shared block. Values per model
(reasons in §7):

| Model | `VLLM_ARGS` |
|-------|-------------|
| 2B | `--max-loras 2 --max-model-len 4096` |
| 4B | `--max-loras 2 --max-model-len 8192` |
| 9B | `--max-loras 4 --max-model-len 4096 --limit-mm-per-prompt '{"image":0,"video":0}'` |

```bash
REPO=/scr1/michael/SkyRL
TAG=4b                                   # 2b | 4b | 9b — names logs/state dirs
MODEL=/scr1/public_models/huggingface/Qwen/Qwen3.5-4B-Base
VLLM_ARGS='--max-loras 2 --max-model-len 8192'

cd $REPO && mkdir -p logs

CUDA_VISIBLE_DEVICES=3 \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
setsid uv run --no-sync --extra fsdp vllm serve $MODEL \
  --tensor-parallel-size 1 \
  --port 8000 \
  --enable-lora \
  --max-lora-rank 64 \
  --dtype bfloat16 \
  $VLLM_ARGS \
  > logs/qwen35_${TAG}_vllm.log 2>&1 < /dev/null &

until curl -fsS -m 5 http://localhost:8000/v1/models >/dev/null; do sleep 2; done
echo vllm_ok
```

## 4. Launch Training Server (Tinker, JAX backend)

Per-model settings (reasons in §7):

| Model | `MEM_FRAC` | `BACKEND_CONFIG` |
|-------|-----------|------------------|
| 2B | 0.95 | `{"max_lora_adapters":2,"max_lora_rank":64,"tensor_parallel_size":1,"fully_sharded_data_parallel_size":2,"train_micro_batch_size":1,"sample_max_num_sequences":16,"gradient_checkpointing":true}` |
| 4B | 0.90 | `{"max_lora_adapters":2,"max_lora_rank":64,"tensor_parallel_size":2,"fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,"sample_max_num_sequences":8,"gradient_checkpointing":true,"loss_chunk_size":128,"train_pad_seq_len_to":4096}` |
| 9B | 0.95 | `{"max_lora_adapters":4,"max_lora_rank":64,"tensor_parallel_size":2,"fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,"sample_max_num_sequences":4,"gradient_checkpointing":true,"loss_chunk_size":128}` |

```bash
REPO=/scr1/michael/SkyRL
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
TAG=4b
MODEL=/scr1/public_models/huggingface/Qwen/Qwen3.5-4B-Base
MEM_FRAC=0.90
BACKEND_CONFIG='{"max_lora_adapters":2,"max_lora_rank":64,"tensor_parallel_size":2,"fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,"sample_max_num_sequences":8,"gradient_checkpointing":true,"loss_chunk_size":128,"train_pad_seq_len_to":4096}'

cd $REPO && mkdir -p logs

CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRAC \
JAX_COMPILATION_CACHE_DIR=$STATE_DIR/skyrl_jax_compilation_cache \
NCCL_NET=Socket \
setsid uv run --active --no-sync --extra gpu --extra tinker --extra jax \
  -m skyrl.tinker.api \
  --base-model $MODEL \
  --backend jax \
  --port 8001 \
  --external-inference-url http://localhost:8000 \
  --external-inference-lora-base /dev/shm/$USER/qwen35_${TAG}_jax_settings_lora_models \
  --checkpoints-base $STATE_DIR/skyrl_qwen35_${TAG}_jax_settings_checkpoints \
  --database-url "postgresql://postgres@/skyrl_tinker?host=$STATE_DIR/skyrl_tinker_pg_data" \
  --backend-config "$BACKEND_CONFIG" \
  > logs/qwen35_${TAG}_jax_settings_tinker.log 2>&1 < /dev/null &

until curl -fsS -m 5 http://localhost:8001/api/v1/get_server_capabilities >/dev/null; do sleep 2; done
echo tinker_ok
```

`JAX_COMPILATION_CACHE_DIR` enables JAX's persistent compilation cache: each
seq-len bucket costs ~160–170 s of XLA JIT (§7), and trainer bounces are routine
(§11 — adapter-slot cleanup, allocator reset), so without the cache every
restart re-pays the full compile ladder. With it, the first launch populates
the cache and later restarts reload compiled executables in seconds. The cache
key includes the jax version, GPU type, and XLA flags — upgrading jax or adding
the §8 debug `XLA_FLAGS` just misses the cache and recompiles (safe). The dir
grows slowly; it is safe to `rm -rf` anytime. Do NOT put it on `/dev/shm`
(it must survive reboots to be useful).

`NCCL_NET=Socket` works around NCCL 2.28.9's net-transport auto-probe
(`ncclNetPluginInit`) segfaulting on any 2-GPU collective on this box — it is
not the SkyRL/Triton/LoRA stack (survives a clean reinstall and every config
change; single-GPU compiles fine; reproduces in a bare torch allreduce).
`NCCL_IB_DISABLE=1` is an equivalent fix; `NCCL_NET_PLUGIN=none` does NOT help.
Apply the same to vLLM if it is ever run with TP=2.

## 5. Verify

```bash
REPO=/scr1/michael/SkyRL
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
PG_BIN=$REPO/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

pgrep -af '[s]kyrl\.tinker|[v]llm serve /scr1/public_models/huggingface/Qwen/Qwen3\.5-'
$PG_BIN/pg_ctl -D $STATE_DIR/skyrl_tinker_pg_data status | head -1
curl -fsS -m 5 http://localhost:8000/v1/models >/dev/null && echo vllm_ok
curl -fsS -m 5 http://localhost:8001/api/v1/get_server_capabilities >/dev/null && echo tinker_ok
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

tail -f logs/qwen35_${TAG}_vllm.log logs/qwen35_${TAG}_jax_settings_tinker.log
```

## 6. Clean State (optional)

Stop the tinker server first (§1) so nothing holds DB connections. All model
variants share the one `skyrl_tinker` database — dropping it clears every
variant's sessions/futures/checkpoints; the on-disk artifacts are per-`TAG`:

```bash
REPO=/scr1/michael/SkyRL
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
PG_BIN=$REPO/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin
TAG=4b

rm -rf /dev/shm/$USER/qwen35_${TAG}_jax_settings_lora_models
rm -rf $STATE_DIR/skyrl_qwen35_${TAG}_jax_settings_checkpoints

# Optional: drop + recreate the shared DB (affects ALL variants)
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "DROP DATABASE IF EXISTS skyrl_tinker;"
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "CREATE DATABASE skyrl_tinker;"
```

To unstick zombie `PENDING` futures without dropping everything (the common
case after an SDK timeout) — flip them to `FAILED` so the client unblocks:

```bash
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d skyrl_tinker -c "
UPDATE futures
SET status = 'FAILED',
    result_data = '{\"error\":\"Marked FAILED by operator\",\"status\":\"failed\"}',
    completed_at = now()
WHERE status = 'PENDING';
SELECT status, COUNT(*) FROM futures GROUP BY status;"
```

## 7. Why Each Setting (per-model memory notes)

### 2B
- FSDP=2/TP=1 fits; the defaults in the §3/§4 tables are sufficient.

### 4B
- **Trainer TP=2 is required** (GPU 0,1 are PCIe `PIX`, not NVLink). Under
  FSDP=2/TP=1 the tied 248K `lm_head` logits are replicated per card instead of
  vocab-sharded, doubling the `forward_backward` transient → OOM. Do not use FSDP.
- **`MEM_FRAC=0.90`, not 0.95**: CUDA-graph / XLA command-buffer instantiations
  are driver allocations *outside* the JAX BFC pool, living in the non-reserved
  headroom. At 0.95 that headroom (~1.6 GiB) is too thin — over a long run,
  accumulated command buffers across seq-len buckets exhaust it, a graph
  re-instantiation OOMs one TP rank, desyncs the TP=2 collective, and aborts
  the process (exit 134, `Retry CUDA graph instantiation after OOM` →
  `rendezvous` termination; 2026-06-10 crash analysis).
- **Do not enable the XLA latency-hiding scheduler**
  (`--xla_gpu_enable_latency_hiding_scheduler`). Tried to overlap the PCIe TP
  all-reduces, but it extends buffer live-ranges and inflated startup high-water
  to ~16 GiB/card before any step ran (vs ~8.5 GiB floor without). Fit beats
  overlap on these 32 GB cards.
- **Train seq-len limit: the 4096 bucket.** Empirically (2026-06-09): buckets
  1024–4096 all compile and run (~160–170 s JIT each); **6144 OOMs** (14.55 GiB
  alloc fails on both cards) even freshly restarted — a pure fit problem. Cap
  client sequences at ≤4096 raw tokens (next bucket above 4096 is 6144; see
  `round_up_seq_len`). Test driver: `seq_len_limit_test.py` (repo root).
- **Concurrent runs need the shared `AdamState` optimizer (2026-07-13 fix in
  `skyrl/backends/jax.py`).** The stock backend allocated a separate
  `nnx.Optimizer` per created model, each holding fp32 Adam moments over the
  *entire* `max_lora_adapters`-stacked LoRA tree (~5–6 GiB/GPU per model at 5
  slots × rank 64). fwd_bwd at the 4096 bucket needs a fixed ~9.94 GiB
  workspace, so one model trained fine for hours but the moment a second
  client created its model, the next 4096-shape step OOM'd (each per-model
  optimizer also paid its own ~40 s optim JIT). The OOM wedges the engine:
  JAX high-water never drops and every train op returns 400 RESOURCE_EXHAUSTED
  until a bounce. The fix replaces per-model optimizers with one shared
  slot-masked `AdamState` (mu/nu allocated once, `optim_step` updates only the
  stepping model's adapter slot — same pattern as `AccumulatedGradients`), so
  optimizer memory is constant in the number of models: 4 concurrent rank-64
  runs at 4096 verified passing. Note: full-state checkpoints saved before the
  fix use the old `optimizer_state` layout and won't reload; sampler weight
  checkpoints are unaffected.
- **`XLA_PYTHON_CLIENT_PREALLOCATE=true`** (changed from `false`): reserves
  MEM_FRAC as one contiguous slab at startup, so a big allocation can always
  be carved from coalesced free space (with `false`, the pool grows as
  multiple regions that cannot coalesce across region boundaries). Safe
  because GPUs 0,1 are dedicated to the trainer. Not sufficient on its own —
  the per-model optimizer bug above OOM'd either way — but keeps the ~10 GiB
  fwd_bwd workspace robust to fragmentation.
- **Pin the train bucket: `"train_pad_seq_len_to":4096`** (knob in
  `JaxBackendConfig`). Pads every fwd/fwd_bwd batch to 4096 so there is exactly
  one train executable and one allocation pattern: slower on short batches, but
  avoids per-bucket executables eating headroom and keeps step memory
  shape-invariant across any number of concurrent runs (fwd_bwd is serialized
  by the engine, so concurrency does not multiply the transient).
- `max_lora_adapters=1` can reject client creation; use 2. vLLM on GPU 3 has
  headroom, so keep `--max-model-len 8192` and no `--enforce-eager`; reduce
  only if the sampling server itself OOMs.
- Do not enable offloading unless the above still produces a real OOM.

### 9B
- **`Qwen3.5-9B-Base` is multimodal** (has `vision_config`). Without
  `--limit-mm-per-prompt '{"image":0,"video":0}'`, vLLM allocates a vision
  encoder buffer that pushes the process to ~31 GB during profiling and OOMs
  the card even at `max-loras=2`. Always pass that flag.
- With multimodal disabled, `max-loras=4` at rank 64 fits alongside the 19.1 GB
  weights on a 32 GB card (~29.3 GB total). Keep `--max-model-len 4096`; no
  `--enforce-eager` or reduced `--gpu-memory-utilization` needed.

## 8. Diagnosing an OOM (which op?)

Use this instead of the normal §4 launch to find *which op* requested the
failing allocation. Env vars must be set at process start: stop the trainer
(§1), relaunch with the additions below, then re-run the triggering workload.
Keep the **same `BACKEND_CONFIG` as the run that OOM'd** so it reproduces —
only the env vars and dump dir are diagnostic additions.

Two capture paths:
- **BFC allocator dump** → the tinker log (`TF_CPP_MIN_LOG_LEVEL=0` +
  `TF_CPP_VMODULE=bfc_allocator=2` give the full per-bin/per-chunk summary).
- **XLA buffer-assignment dump** → `$STATE_DIR/skyrl_xla_dump`
  (`--xla_dump_hlo_as_long_text` includes op name + source file/line).

```bash
# Fresh dump dir each run so only this run's modules appear
rm -rf $STATE_DIR/skyrl_xla_dump && mkdir -p $STATE_DIR/skyrl_xla_dump

# Add to the §4 trainer launch env:
TF_CPP_MIN_LOG_LEVEL=0 \
TF_CPP_VMODULE=bfc_allocator=2 \
JAX_TRACEBACK_FILTERING=off \
XLA_FLAGS="--xla_dump_to=$STATE_DIR/skyrl_xla_dump --xla_dump_hlo_as_long_text --xla_dump_hlo_as_text" \
...
```

After it OOMs, find the culprit — the `forward_backward` module is the big one;
its largest buffer's **shape** alone settles logits vs attention:

```bash
# 0. Allocator headline (confirms the failing size)
grep -nE "ran out of memory|requested by op|Sum Total|Bin \(" \
  logs/qwen35_${TAG}_jax_settings_tinker.log | tail -40

# 1. Biggest module = the forward_backward graph
ls -S $STATE_DIR/skyrl_xla_dump/*-memory-usage-report.txt | head

# 2. Largest buffers first, WITH shapes:
#    f32[6144,248320]-ish   => logits over the 248K vocab
#    f32[2,8,3072,3072]-ish => attention scores
head -15 "$(ls -S $STATE_DIR/skyrl_xla_dump/*-memory-usage-report.txt | head -1)"

# 3. Map size -> op + source line via buffer-assignment. Entries look like:
#      allocation N: size SSSS, ...:
#       value: <id op_name @k> (size=SSSS,offset=...): <shape>
BA="$(ls -S $STATE_DIR/skyrl_xla_dump/*-buffer-assignment.txt | head -1)"
grep -nE "size (2[0-9]{10}|289155)" "$BA" | head   # adjust to the failing size
```

## 9. Attention Backend (hardware note)

Qwen3.5's `head_dim=256` exceeds cuDNN flash attention's **128** head-dim cap
on Ampere/Ada (this sm_89 box), so `dot_product_attention` falls back off
cuDNN: causal prefill/training uses Pallas/Triton, non-causal decode uses XLA.

The cap is set at launch via `SKYRL_CUDNN_MAX_HEAD_DIM` (read by
`skyrl/tx/layers/attention.py`, default **128**). Pick by GPU — no auto-detection:

- **Ampere/Ada (≤ sm_89)**: leave unset. Setting 256 is rejected by JAX
  (`NotImplementedError: head dim must be <= 128` — confirmed empirically).
- **Hopper (sm_90)+**: export `SKYRL_CUDNN_MAX_HEAD_DIM=256` on the trainer
  launch so head_dim=256 runs the native cuDNN kernel (faster than XLA,
  skips the Pallas path).

## 10. 9B on the H100 Box (multi-concurrent matrix)

Config that converged running the InjecAgent matrix (up to 4 concurrent client
runs, Qwen3.5-9B-Base attacker) on the **3× H100 80 GB modal box**. Differences
from the local box:

- **GPU split:** GPU 0,1 = JAX trainer (TP=2); **GPU 2** = vLLM (TP=1).
- **sm_90:** export `SKYRL_CUDNN_MAX_HEAD_DIM=256` on the trainer (§9).
- **Env quirks baked into the launch scripts:** `TMPDIR=/root/tmp`,
  `UV_PROJECT_ENVIRONMENT=.venv-jax` (trainer), `VLLM_USE_DEEP_GEMM=0` (vLLM
  0.20.2 crashes in DeepGEMM warmup otherwise), `HF_HUB_ENABLE_HF_TRANSFER=1`.
- Launch via `/root/launch_tinker.sh` and `/root/launch_vllm.sh`
  (self-contained, detached; each `pkill`s its own predecessor, then re-execs).
- Add `JAX_COMPILATION_CACHE_DIR` (§4) to `launch_tinker.sh` there too —
  restarts are even more frequent on that box (adapter-slot cleanup after
  crashed matrix runs).

Trainer backend-config:

```json
{"max_lora_adapters":5,"max_lora_rank":64,"tensor_parallel_size":2,
 "fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,
 "sample_max_num_sequences":16,"gradient_checkpointing":true,"loss_chunk_size":64}
```

vLLM: `--max-loras 9 --max-num-seqs 512 --max-num-batched-tokens 16384
--max-model-len 4096 --limit-mm-per-prompt '{"image":0,"video":0}'`
(`--max-loras` only needs `>=` the trainer's `max_lora_adapters`; 9 is
harmless headroom over 5).

Why each value (learned the hard way):

- **`train_micro_batch_size=1`** — `=4` × padded InjecAgent sequences made the
  backward pass need 93.82 GiB on one card → OOM. mb=1 peaks ~23 GiB. Do NOT raise.
- **`max_lora_adapters=5`** (⇒ at most **4 live adapters**; slot 0 is the base
  model, a JAX-backend quirk). `=9` was a double mistake: (a) the JAX backend
  stores LoRA grads stacked over ALL slots (`accumulated_grads =
  zeros_like(lora_params)`), so grad + accum-buffer memory scale with slot
  count — `=9` pinned ~66 GiB/GPU and OOM'd 4-concurrent; (b) it slowed
  `forward_backward` to ~83–91 s (vs ~17–25 s at 5).
- **`loss_chunk_size=64`** — halves the logits chunk vs 128, fitting 4
  concurrent fwd_bwd under the ~76 GiB/GPU usable cap. If it OOMs again, drop to 32.
- **`sample_max_num_sequences=16`** — THE fix for the worst failure. At 64,
  four concurrent runs (each `n_attacks=8`) admitted 100+ concurrent sample
  sequences into vLLM, which wedged at 0 tokens/s. The trainer's
  `/load_lora_adapter` then hit a 300 s ReadTimeout, surfaced to clients as an
  opaque `400 {'detail': ''}` on a `SampleResponse` — killing all 4 runs at
  once. 16 caps the per-cycle burst. If it recurs, drop to 8 and/or throttle
  sampling client-side.

## 11. Operational Gotchas

- **Crashed runs do NOT free trainer LoRA-adapter slots.** A dead client leaves
  its adapter registered; each crash burns a slot until a **trainer restart**.
  If the registry fills (`400 {'detail':'Maximum number of LoRA adapters (N)
  reached'}`), bounce the trainer. JAX OOMs return as 400s without killing the
  trainer process, so the registry is never auto-cleaned.
- **vLLM never unloads sampler adapters.** A fresh ephemeral LoRA is loaded per
  checkpoint and `/unload` is never called, so vLLM accumulates them (saw 25).
  Bounce vLLM periodically; this also clears any sampling stall.
- **Restart playbook:** bouncing the trainer resets the JAX allocator
  high-water (it does NOT drop on its own after an OOM) and clears zombie
  adapter slots. vLLM can stay up unless wedged or bloated with adapters.
  Cap client concurrency at **4** on the H100 box.
