# Qwen3.5 JAX Tinker + vLLM Server Launch

This file is only for launching the two local services:

```text
GPU 0,1: JAX Tinker training server, port 8001
GPU 3:   vLLM sampling server, port 8000
```

Default model:

```text
/scr1/public_models/huggingface/Qwen/Qwen3.5-2B-Base
```

State directory. All local server state (pgdata, checkpoints, LoRA adapters,
XLA dumps, logs) lives under `$STATE_DIR`, which resolves to
`/storage_slow/michael` when that per-user dir is writable (large slow disk)
and falls back to `/tmp` otherwise. (The `/storage_slow` root itself is
root-owned and not user-writable, so we use the per-user subdir.)
Every bash block below recomputes it locally so each block stays
copy-paste-standalone:

```bash
STATE_DIR=$([ -w /storage_slow/michael ] && echo /storage_slow/michael || echo /tmp)
```

**Sampler LoRA adapters live on tmpfs.** `--external-inference-lora-base` is set to
`/dev/shm/$USER/..._lora_models` (RAM-backed tmpfs), NOT `$STATE_DIR`. The JAX backend
now writes each sampler adapter directly there as a plain directory that vLLM loads in
place — no tar pack, no read-back, no un-tar (previously every
`save_weights_and_get_sampling_client` wrote a tar to the slow disk and then re-extracted
it). These adapters are transient (needed only until vLLM loads them), so tmpfs is ideal.
Two caveats: (1) `/dev/shm` must be shared by the colocated trainer/API/vLLM — it is, on
this single-node box; (2) vLLM never unloads adapters, so the dir grows in RAM until a
vLLM bounce — keep bouncing vLLM periodically (see §10) and `rm -rf` the `/dev/shm/...`
dir on restart. Durable training checkpoints still go to `--checkpoints-base` under
`$STATE_DIR`.

## 1. Stop Existing Servers

```bash
pkill -f '[s]kyrl.tinker' || true
pkill -f '[v]llm serve /scr1/public_models/huggingface/Qwen/Qwen3.5-' || true

for i in $(seq 1 30); do
  if ! pgrep -f '[s]kyrl.tinker|[v]llm serve /scr1/public_models/huggingface/Qwen/Qwen3.5-' >/dev/null; then
    break
  fi
  sleep 1
done
```

Do NOT stop PostgreSQL here. It is shared state; keep it running across launches.

## 2. Ensure PostgreSQL is Running

Tinker uses PostgreSQL instead of SQLite to avoid `database is locked` hangs when
the client fires many parallel `asample` calls (SQLite's writer queue collides
during `save_weights` and leaves zombie `PENDING` futures that block the SDK
until its 7200s timeout).

We use an embedded Postgres bundled with the `pgserver` PyPI package. We only
use its binaries — **never run a Python script that imports `pgserver` against
this pgdata while Postgres is up**. `pgserver` registers an `atexit` handler
that SIGTERMs the postmaster on script exit even when `cleanup_mode=None`.

Paths (with `STATE_DIR=$([ -w /storage_slow/michael ] && echo /storage_slow/michael || echo /tmp)`):

```text
STATE_DIR: /storage_slow/michael if writable, else /tmp
PGDATA:   $STATE_DIR/skyrl_tinker_pg_data
SOCKET:   $STATE_DIR/skyrl_tinker_pg_data/.s.PGSQL.5432
DBNAME:   skyrl_tinker
PG_BIN:   /scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin
PG_URL:   postgresql://postgres@/skyrl_tinker?host=$STATE_DIR/skyrl_tinker_pg_data
```

**Note:** `$STATE_DIR` must resolve to the same path across launch, verify, and
clean blocks (and across server restarts) — otherwise the trainer points at a
different pgdata/database. As long as `/storage_slow` is mounted consistently
this is stable; if you create the cluster on `/tmp` and `/storage_slow` later
appears, the §2a one-time setup must be re-run on the new location.

### 2a. One-time setup (only if pgdata does not exist)

```bash
STATE_DIR=$([ -w /storage_slow/michael ] && echo /storage_slow/michael || echo /tmp)

# Install pgserver into the venv (one-time)
cd /scr1/michael/SkyRL
uv pip install pgserver

PG_BIN=/scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

# initdb a fresh cluster
mkdir -p $STATE_DIR/skyrl_tinker_pg_data
$PG_BIN/initdb -D $STATE_DIR/skyrl_tinker_pg_data -U postgres --auth=trust

# Start it (see 2b)
nohup setsid $PG_BIN/pg_ctl -D $STATE_DIR/skyrl_tinker_pg_data \
  -l $STATE_DIR/skyrl_tinker_pg_data/log \
  -o "-h '' -k $STATE_DIR/skyrl_tinker_pg_data" start \
  </dev/null >$STATE_DIR/pg_start.log 2>&1

# Create the tinker database (idempotent)
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -tAc "SELECT 1 FROM pg_database WHERE datname='skyrl_tinker'" | grep -q 1 \
  || $PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
       -c "CREATE DATABASE skyrl_tinker;"
```

Tinker auto-runs `SQLModel.metadata.create_all` on startup, so no Alembic
migration step is needed.

### 2b. Start (idempotent — safe to run on every launch)

```bash
STATE_DIR=$([ -w /storage_slow/michael ] && echo /storage_slow/michael || echo /tmp)
PG_BIN=/scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

$PG_BIN/pg_ctl -D $STATE_DIR/skyrl_tinker_pg_data status >/dev/null 2>&1 || \
nohup setsid $PG_BIN/pg_ctl -D $STATE_DIR/skyrl_tinker_pg_data \
  -l $STATE_DIR/skyrl_tinker_pg_data/log \
  -o "-h '' -k $STATE_DIR/skyrl_tinker_pg_data" start \
  </dev/null >$STATE_DIR/pg_start.log 2>&1

$PG_BIN/pg_ctl -D $STATE_DIR/skyrl_tinker_pg_data status
ls $STATE_DIR/skyrl_tinker_pg_data/.s.PGSQL.5432 && echo pg_ok
```

## 3. Launch 2B Sampling Server

```bash
cd /scr1/michael/SkyRL
mkdir -p logs

CUDA_VISIBLE_DEVICES=3 \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
setsid uv run --no-sync --extra fsdp vllm serve \
  /scr1/public_models/huggingface/Qwen/Qwen3.5-2B-Base \
  --tensor-parallel-size 1 \
  --port 8000 \
  --enable-lora \
  --max-loras 2 \
  --max-lora-rank 64 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  > logs/qwen35_2b_vllm.log 2>&1 < /dev/null &
```

Wait for vLLM:

```bash
until curl -fsS -m 5 http://localhost:8000/v1/models >/dev/null; do
  sleep 2
done
echo vllm_ok
```

## 4. Launch 2B Training Server

```bash
cd /scr1/michael/SkyRL
mkdir -p logs

STATE_DIR=$([ -w /storage_slow/michael ] && echo /storage_slow/michael || echo /tmp)

CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
setsid uv run --active --no-sync --extra gpu --extra tinker --extra jax \
  -m skyrl.tinker.api \
  --base-model /scr1/public_models/huggingface/Qwen/Qwen3.5-2B-Base \
  --backend jax \
  --port 8001 \
  --external-inference-url http://localhost:8000 \
  --external-inference-lora-base /dev/shm/$USER/qwen35_2b_jax_settings_lora_models \
  --checkpoints-base $STATE_DIR/skyrl_qwen35_2b_jax_settings_checkpoints \
  --database-url "postgresql://postgres@/skyrl_tinker?host=$STATE_DIR/skyrl_tinker_pg_data" \
  --backend-config '{"max_lora_adapters":2,"max_lora_rank":64,"tensor_parallel_size":1,"fully_sharded_data_parallel_size":2,"train_micro_batch_size":1,"sample_max_num_sequences":16,"gradient_checkpointing":true}' \
  > logs/qwen35_2b_jax_settings_tinker.log 2>&1 < /dev/null &
```

Wait for Tinker:

```bash
until curl -fsS -m 5 http://localhost:8001/api/v1/get_server_capabilities >/dev/null; do
  sleep 2
done
echo tinker_ok
```

## 4b. Diagnostic launch: identify the OOM op

Use this instead of §4 when you need to find *which op* requested the failing
allocation (e.g. the 26.93 GiB OOM at the seq_len=3072 train bucket). The extra
env vars must be set at process start, so stop the trainer (§1) and relaunch
with this block, then re-run the workload that triggers the OOM.

Keep the **same `--backend-config` as the run that OOM'd** so it reproduces
(the failing run used `loss_chunk_size:32`); only the env vars and the dump dir
below are the diagnostic additions. If your client uses a different config, mirror
that instead.

It captures the culprit two ways:

- **BFC allocator dump** → printed into `logs/qwen35_2b_jax_settings_tinker.log`
  (`TF_CPP_MIN_LOG_LEVEL=0` + `TF_CPP_VMODULE=bfc_allocator=2` enable the full
  per-bin / per-chunk summary, not just the truncated headline).
- **XLA buffer-assignment dump** → written to `$STATE_DIR/skyrl_xla_dump`
  (`--xla_dump_hlo_as_long_text` includes op metadata: op name + source file/line),
  so the big buffer can be mapped back to the model code.

```bash
cd /scr1/michael/SkyRL
mkdir -p logs

STATE_DIR=$([ -w /storage_slow/michael ] && echo /storage_slow/michael || echo /tmp)

# Fresh dump dir each run so we only see this run's modules
rm -rf $STATE_DIR/skyrl_xla_dump && mkdir -p $STATE_DIR/skyrl_xla_dump

CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
TF_CPP_MIN_LOG_LEVEL=0 \
TF_CPP_VMODULE=bfc_allocator=2 \
JAX_TRACEBACK_FILTERING=off \
XLA_FLAGS="--xla_dump_to=$STATE_DIR/skyrl_xla_dump --xla_dump_hlo_as_long_text --xla_dump_hlo_as_text" \
setsid uv run --active --no-sync --extra gpu --extra tinker --extra jax \
  -m skyrl.tinker.api \
  --base-model /scr1/public_models/huggingface/Qwen/Qwen3.5-2B-Base \
  --backend jax \
  --port 8001 \
  --external-inference-url http://localhost:8000 \
  --external-inference-lora-base /dev/shm/$USER/qwen35_2b_jax_settings_lora_models \
  --checkpoints-base $STATE_DIR/skyrl_qwen35_2b_jax_settings_checkpoints \
  --database-url "postgresql://postgres@/skyrl_tinker?host=$STATE_DIR/skyrl_tinker_pg_data" \
  --backend-config '{"max_lora_adapters":2,"max_lora_rank":64,"tensor_parallel_size":1,"fully_sharded_data_parallel_size":2,"train_micro_batch_size":1,"sample_max_num_sequences":16,"gradient_checkpointing":true,"loss_chunk_size":32}' \
  > logs/qwen35_2b_jax_settings_tinker.log 2>&1 < /dev/null &
```

After it OOMs, pull out the culprit. The forward_backward module is the big
one — find it, then read its largest buffer's **shape** (this alone settles
logits vs attention) and map it to an op.

```bash
STATE_DIR=$([ -w /storage_slow/michael ] && echo /storage_slow/michael || echo /tmp)

# 0. Allocator headline from the run (confirms the failing size)
grep -nE "ran out of memory|requested by op|Sum Total|Bin \(" \
  logs/qwen35_2b_jax_settings_tinker.log | tail -40

# 1. The biggest module is the forward_backward graph
ls -S $STATE_DIR/skyrl_xla_dump/*-memory-usage-report.txt | head

# 2. Largest buffers first, WITH shapes. The ~27 GiB row's shape tells you what
#    it is: f32[6144,248320]-ish => logits over the 248K vocab;
#           f32[2,8,3072,3072]-ish => attention scores.
head -15 "$(ls -S $STATE_DIR/skyrl_xla_dump/*-memory-usage-report.txt | head -1)"

# 3. Map that shape/value back to an op + source line via buffer-assignment.
#    Entries look like:
#      allocation N: size SSSS, ...:
#       value: <id op_name @k> (size=SSSS,offset=...): <shape>
#    Find the allocation whose size is ~28.9e9 (28915510016) and read its op_name.
BA="$(ls -S $STATE_DIR/skyrl_xla_dump/*-buffer-assignment.txt | head -1)"
grep -nE "size (2[0-9]{10}|289155)" "$BA" | head     # ~27 GiB allocations
```

## 5. Verify

```bash
STATE_DIR=$([ -w /storage_slow/michael ] && echo /storage_slow/michael || echo /tmp)
PG_BIN=/scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

pgrep -af '[s]kyrl.tinker|[v]llm serve /scr1/public_models/huggingface/Qwen/Qwen3.5-'
$PG_BIN/pg_ctl -D $STATE_DIR/skyrl_tinker_pg_data status | head -1
curl -fsS -m 5 http://localhost:8000/v1/models >/dev/null && echo vllm_ok
curl -fsS -m 5 http://localhost:8001/api/v1/get_server_capabilities >/dev/null && echo tinker_ok
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Logs:

```bash
tail -f logs/qwen35_2b_vllm.log
tail -f logs/qwen35_2b_jax_settings_tinker.log
```

## 6. Optional Clean 2B State

Only run this when you want to discard local 2B server state and checkpoints.
Stop the tinker server first (section 1) so nothing is holding open connections
to the database:

```bash
STATE_DIR=$([ -w /storage_slow/michael ] && echo /storage_slow/michael || echo /tmp)
PG_BIN=/scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

# On-disk artifacts
rm -rf /dev/shm/$USER/qwen35_2b_jax_settings_lora_models
rm -rf $STATE_DIR/skyrl_qwen35_2b_jax_settings_checkpoints

# Drop + recreate the tinker database (clears all sessions, futures, checkpoints)
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "DROP DATABASE IF EXISTS skyrl_tinker;"
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "CREATE DATABASE skyrl_tinker;"
```

To unstick zombie `PENDING` futures without dropping everything (the most common
case after an SDK timeout) — flip them to `FAILED` so the client unblocks:

```bash
STATE_DIR=$([ -w /storage_slow/michael ] && echo /storage_slow/michael || echo /tmp)
PG_BIN=/scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d skyrl_tinker -c "
UPDATE futures
SET status = 'FAILED',
    result_data = '{\"error\":\"Marked FAILED by operator\",\"status\":\"failed\"}',
    completed_at = now()
WHERE status = 'PENDING';
SELECT status, COUNT(*) FROM futures GROUP BY status;"
```

## 7. 4B Variant

Use the same GPU split and ports, but replace the launch commands with these
model-specific settings.

Memory notes from the 4B local runs:

- vLLM had enough headroom on GPU 3, so do not use the most aggressive vLLM memory reductions by default. Use `--max-model-len 8192`, `--max-loras 2`, and no `--enforce-eager`; reduce these only if the sampling server itself OOMs.
- Keep Tinker at the smallest workable adapter pool. `max_lora_adapters=1` can reject client creation in this setup, so use `2` rather than a larger value.
- Keep `train_micro_batch_size=1`, `sample_max_num_sequences=8`, `gradient_checkpointing=true`, and `loss_chunk_size=128`.
- Keep `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`
  so JAX does not grab the entire card up front. **Use 0.90, not 0.95**: CUDA
  graph / XLA command-buffer instantiations are allocated by the driver *outside*
  the JAX BFC pool, so they live in the non-reserved headroom. At 0.95 that
  headroom (~1.6 GiB) is too thin — over a long run with many distinct seq-len
  buckets, accumulated command buffers exhaust it and a graph re-instantiation
  OOMs one TP rank, which desyncs the TP=2 collective and aborts the whole
  process (exit 134, `cuda_command_buffer.cc: Retry CUDA graph instantiation
  after OOM` → `rendezvous` termination). 0.90 leaves room for this. See the
  2026-06-10 crash analysis.
- Do not enable offloading by default. Use it only after the above settings still produce a real OOM.
- **Train seq-len limit: the 4096 bucket.** Empirically (2026-06-09, this config):
  buckets 1024/1536/2048/3072/4096 all compile and run (~160–170 s JIT each);
  **6144 OOMs** (14.55 GiB alloc fails on both cards) even on a freshly
  restarted trainer — a pure fit problem. Cap client sequences at ≤4096 raw
  tokens (the next bucket above 4096 is 6144; see `round_up_seq_len`).
  Test driver: `seq_len_limit_test.py` (repo root).

Sampling server:

```bash
cd /scr1/michael/SkyRL
mkdir -p logs

CUDA_VISIBLE_DEVICES=3 \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
setsid uv run --no-sync --extra fsdp vllm serve \
  /scr1/public_models/huggingface/Qwen/Qwen3.5-4B-Base \
  --tensor-parallel-size 1 \
  --port 8000 \
  --enable-lora \
  --max-loras 2 \
  --max-lora-rank 64 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  > logs/qwen35_4b_vllm.log 2>&1 < /dev/null &
```

Training server — **TP=2 on GPU 0,1** (GPU 0,1 are PCIe `PIX`, not NVLink).

- **TP=2 is required.** Under FSDP=2/TP=1 the tied 248K `lm_head` logits are
  replicated per card instead of vocab-sharded, which doubles the
  `forward_backward` transient and OOMs the 4B. Do not use FSDP here.
- **Do not enable the XLA latency-hiding scheduler**
  (`--xla_gpu_enable_latency_hiding_scheduler`). It was tried to overlap the PCIe
  TP all-reduces, but it extends buffer live-ranges and inflated the **startup**
  memory high-water to ~16 GiB/card *before any step ran* — leaving no room for
  `forward_backward`. Without it the floor sits at ~8.5 GiB/card. Accept the lower
  GPU util; correctness/fit beats overlap on these 32 GB cards.

```bash
cd /scr1/michael/SkyRL
mkdir -p logs

STATE_DIR=$([ -w /storage_slow/michael ] && echo /storage_slow/michael || echo /tmp)

CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
setsid uv run --active --no-sync --extra gpu --extra tinker --extra jax \
  -m skyrl.tinker.api \
  --base-model /scr1/public_models/huggingface/Qwen/Qwen3.5-4B-Base \
  --backend jax \
  --port 8001 \
  --external-inference-url http://localhost:8000 \
  --external-inference-lora-base /dev/shm/$USER/qwen35_4b_jax_settings_lora_models \
  --checkpoints-base $STATE_DIR/skyrl_qwen35_4b_jax_settings_checkpoints \
  --database-url "postgresql://postgres@/skyrl_tinker?host=$STATE_DIR/skyrl_tinker_pg_data" \
  --backend-config '{"max_lora_adapters":2,"max_lora_rank":64,"tensor_parallel_size":2,"fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,"sample_max_num_sequences":8,"gradient_checkpointing":true,"loss_chunk_size":128}' \
  > logs/qwen35_4b_jax_settings_tinker.log 2>&1 < /dev/null &
```

Optional clean 4B state. Both 2B and 4B share the same `skyrl_tinker` database,
so dropping it clears 2B state too — only do this if you actually want a clean
slate for the database side. The on-disk artifacts below are 4B-specific:

```bash
STATE_DIR=$([ -w /storage_slow/michael ] && echo /storage_slow/michael || echo /tmp)
PG_BIN=/scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

rm -rf /dev/shm/$USER/qwen35_4b_jax_settings_lora_models
rm -rf $STATE_DIR/skyrl_qwen35_4b_jax_settings_checkpoints

# Optional: drop the shared tinker DB (affects 2B too)
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "DROP DATABASE IF EXISTS skyrl_tinker;"
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "CREATE DATABASE skyrl_tinker;"
```

## 8. 9B Variant

Same GPU split and ports as the 2B/4B variants. Trainer TP=2 on GPU 0,1; vLLM
TP=1 on GPU 3.

Memory notes from the 9B local runs:

- **`Qwen3.5-9B-Base` is multimodal** (has `vision_config`). vLLM without
  `--limit-mm-per-prompt '{"image":0,"video":0}'` allocates a vision encoder
  buffer that pushes the process to ~31 GB during profiling and OOMs the card
  — even with `max-loras=2`. Always pass that flag.
- With multimodal disabled, `max-loras=4` at rank 64 fits alongside the 19.1 GB
  model weights on a 32 GB card (~29.3 GB total). Without it, even
  `max-loras=2` OOMs.
- Keep `max-model-len=4096` and the default `max-num-seqs`. No
  `--enforce-eager` or reduced `--gpu-memory-utilization` needed once
  multimodal is disabled.
- Trainer side: `tensor_parallel_size=2`, `fully_sharded_data_parallel_size=1`,
  `train_micro_batch_size=1`, `sample_max_num_sequences=4`,
  `gradient_checkpointing=true`, `loss_chunk_size=128`.
- Keep `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`.

Sampling server:

```bash
cd /scr1/michael/SkyRL
mkdir -p logs

CUDA_VISIBLE_DEVICES=3 \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
setsid uv run --no-sync --extra fsdp vllm serve \
  /scr1/public_models/huggingface/Qwen/Qwen3.5-9B-Base \
  --tensor-parallel-size 1 \
  --port 8000 \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  > logs/qwen35_9b_vllm.log 2>&1 < /dev/null &
```

Training server:

```bash
cd /scr1/michael/SkyRL
mkdir -p logs

STATE_DIR=$([ -w /storage_slow/michael ] && echo /storage_slow/michael || echo /tmp)

CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
setsid uv run --active --no-sync --extra gpu --extra tinker --extra jax \
  -m skyrl.tinker.api \
  --base-model /scr1/public_models/huggingface/Qwen/Qwen3.5-9B-Base \
  --backend jax \
  --port 8001 \
  --external-inference-url http://localhost:8000 \
  --external-inference-lora-base /dev/shm/$USER/qwen35_9b_jax_settings_lora_models \
  --checkpoints-base $STATE_DIR/skyrl_qwen35_9b_jax_settings_checkpoints \
  --database-url "postgresql://postgres@/skyrl_tinker?host=$STATE_DIR/skyrl_tinker_pg_data" \
  --backend-config '{"max_lora_adapters":4,"max_lora_rank":64,"tensor_parallel_size":2,"fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,"sample_max_num_sequences":4,"gradient_checkpointing":true,"loss_chunk_size":128}' \
  > logs/qwen35_9b_jax_settings_tinker.log 2>&1 < /dev/null &
```

Optional clean 9B state. The 2B, 4B, and 9B variants all share the same
`skyrl_tinker` database, so dropping it clears the others too — only do this
if you actually want a clean slate for the database side. The on-disk
artifacts below are 9B-specific:

```bash
STATE_DIR=$([ -w /storage_slow/michael ] && echo /storage_slow/michael || echo /tmp)
PG_BIN=/scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

rm -rf /dev/shm/$USER/qwen35_9b_jax_settings_lora_models
rm -rf $STATE_DIR/skyrl_qwen35_9b_jax_settings_checkpoints

# Optional: drop the shared tinker DB (affects 2B and 4B too)
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "DROP DATABASE IF EXISTS skyrl_tinker;"
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "CREATE DATABASE skyrl_tinker;"
```

## 9. Attention backend (hardware note)

Qwen3.5's `head_dim=256` exceeds cuDNN flash attention's **128** head-dim cap on
Ampere/Ada (e.g. the RTX 5000 Ada / sm_89 box here), so `dot_product_attention`
falls back off cuDNN: the causal prefill/training path uses Pallas/Triton, the
non-causal decode path uses XLA.

The cap is set at launch via `SKYRL_CUDNN_MAX_HEAD_DIM` (read by
`skyrl/tx/layers/attention.py`, default **128**). The operator picks the value by
GPU — there is no auto-detection:

- **Ampere/Ada (sm_89 and earlier):** leave unset (128); uses the Pallas/XLA
  fallback. Setting 256 here is rejected by JAX with
  `NotImplementedError: head dim must be <= 128` — confirmed empirically.
- **Hopper (sm_90)+:** export `SKYRL_CUDNN_MAX_HEAD_DIM=256` on the §8
  training-server launch so head_dim=256 runs the native cuDNN kernel (faster
  than XLA, skips the Pallas path).

## 10. 9B on the H100 box (multi-concurrent matrix)

This section documents the server config that converged while running the
InjecAgent matrix (up to 4 concurrent client runs, Qwen3.5-9B-Base attacker
trained via Tinker) on the **3× H100 80GB modal box** — a different environment
from §8 (which targets the local sm_89 / 32 GB RTX-5000 box). Differences:

- **GPU split:** GPU 0,1 = JAX trainer (TP=2); **GPU 2** = vLLM (TP=1). Not GPU 3.
- **sm_90 (Hopper):** `SKYRL_CUDNN_MAX_HEAD_DIM=256` so head_dim=256 runs native
  cuDNN (per §9). The trainer launch exports it.
- **Env quirks baked into the launch scripts:** `TMPDIR=/root/tmp`,
  `UV_PROJECT_ENVIRONMENT=.venv-jax` (trainer), `VLLM_USE_DEEP_GEMM=0` (vLLM
  0.20.2 crashes in DeepGEMM warmup otherwise), `HF_HUB_ENABLE_HF_TRANSFER=1`.
- Launch via `/root/launch_tinker.sh` and `/root/launch_vllm.sh` (self-contained,
  detached; each `pkill`s its own predecessor then re-execs).

### Converged backend-config (trainer)

```json
{"max_lora_adapters":5,"max_lora_rank":64,"tensor_parallel_size":2,
 "fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,
 "sample_max_num_sequences":16,"gradient_checkpointing":true,"loss_chunk_size":64}
```

vLLM: `--max-loras 9 --max-num-seqs 512 --max-num-batched-tokens 16384
--max-model-len 4096 --limit-mm-per-prompt '{"image":0,"video":0}'`.
(`--max-loras` only needs to be `>=` the trainer's `max_lora_adapters`; 9 is
harmless headroom over 5.)

### Why each value (learned the hard way this run)

- **`train_micro_batch_size=1`** — `=4` × padded InjecAgent sequence lengths made
  the backward pass need 93.82 GiB on a single card → OOM. mb=1 peaks ~23 GiB
  per pass. Do NOT raise it.
- **`max_lora_adapters=5`** (⇒ at most **4 live adapters**; slot 0 is the base
  model, a JAX-backend quirk). `=9` was a double mistake: (a) the JAX backend
  stores LoRA grads as one tensor stacked over ALL slots
  (`accumulated_grads = zeros_like(lora_params)`), so per-fwd_bwd grad memory +
  the accum buffer scale with slot count — `=9` pinned ~66 GiB/GPU and OOM'd
  4-concurrent; and (b) it slowed `forward_backward` to ~83–91 s (vs ~17–25 s at
  5). `5` is the proven value: enough for 4 concurrent runs, small floor.
- **`loss_chunk_size=64`** (down from 128) — halves the logits chunk, shrinking
  the per-pass transient enough that 4 concurrent fwd_bwd fit under the
  ~76 GiB/GPU usable cap. If 4-concurrent ever OOMs again, drop to 32.
- **`sample_max_num_sequences=16`** (down from 64) — THE fix for the worst
  failure. At 64, four concurrent runs (each `n_attacks=8`) admitted ~100+
  concurrent sample sequences into vLLM, which wedged at **0 tokens/s**
  (`"Running: 102 reqs, Avg generation throughput: 0.0 tokens/s"` in the vLLM
  log). The trainer's `/load_lora_adapter` (external_inference.py) then hit a
  300 s ReadTimeout and the SDK surfaced it to the client as an opaque
  `400 {'detail': ''}` on a `SampleResponse` — which killed all 4 runs at once.
  16 caps the per-cycle sampling burst so vLLM never wedges. If it recurs, drop
  to 8 and/or throttle attacker sampling client-side.

### Operational gotchas (cost us a full matrix)

- **Crashed runs do NOT free trainer LoRA-adapter slots.** A dead client leaves
  its adapter registered; with only 4 usable slots, every crash permanently
  burns one until a **trainer restart**. So the goal is ZERO crashes — and if
  the registry fills (`400 {'detail':'Maximum number of LoRA adapters (5)
  reached'}`), bounce the trainer to reset it. The JAX OOMs return as 400s
  without killing the trainer process, so the registry isn't auto-cleaned.
- **vLLM never unloads sampler adapters.** external_inference.py loads a fresh
  ephemeral LoRA per checkpoint and never calls `/unload`, so vLLM accumulates
  them (saw 25). Bounce vLLM periodically (it also clears any sampling stall).
- **Restart playbook:** bouncing the trainer resets the JAX allocator
  high-water (it does NOT drop on its own after an OOM) and clears zombie
  adapter slots; vLLM can stay up unless it has wedged or accumulated too many
  adapters, in which case bounce it too. Cap client concurrency at **4**.
