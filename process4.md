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

Paths:

```text
PGDATA:   /tmp/skyrl_tinker_pg_data
SOCKET:   /tmp/skyrl_tinker_pg_data/.s.PGSQL.5432
DBNAME:   skyrl_tinker
PG_BIN:   /scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin
PG_URL:   postgresql://postgres@/skyrl_tinker?host=/tmp/skyrl_tinker_pg_data
```

### 2a. One-time setup (only if pgdata does not exist)

```bash
# Install pgserver into the venv (one-time)
cd /scr1/michael/SkyRL
uv pip install pgserver

PG_BIN=/scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

# initdb a fresh cluster
mkdir -p /tmp/skyrl_tinker_pg_data
$PG_BIN/initdb -D /tmp/skyrl_tinker_pg_data -U postgres --auth=trust

# Start it (see 2b)
nohup setsid $PG_BIN/pg_ctl -D /tmp/skyrl_tinker_pg_data \
  -l /tmp/skyrl_tinker_pg_data/log \
  -o "-h '' -k /tmp/skyrl_tinker_pg_data" start \
  </dev/null >/tmp/pg_start.log 2>&1

# Create the tinker database (idempotent)
$PG_BIN/psql -h /tmp/skyrl_tinker_pg_data -U postgres -d postgres \
  -tAc "SELECT 1 FROM pg_database WHERE datname='skyrl_tinker'" | grep -q 1 \
  || $PG_BIN/psql -h /tmp/skyrl_tinker_pg_data -U postgres -d postgres \
       -c "CREATE DATABASE skyrl_tinker;"
```

Tinker auto-runs `SQLModel.metadata.create_all` on startup, so no Alembic
migration step is needed.

### 2b. Start (idempotent — safe to run on every launch)

```bash
PG_BIN=/scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

$PG_BIN/pg_ctl -D /tmp/skyrl_tinker_pg_data status >/dev/null 2>&1 || \
nohup setsid $PG_BIN/pg_ctl -D /tmp/skyrl_tinker_pg_data \
  -l /tmp/skyrl_tinker_pg_data/log \
  -o "-h '' -k /tmp/skyrl_tinker_pg_data" start \
  </dev/null >/tmp/pg_start.log 2>&1

$PG_BIN/pg_ctl -D /tmp/skyrl_tinker_pg_data status
ls /tmp/skyrl_tinker_pg_data/.s.PGSQL.5432 && echo pg_ok
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
  --max-loras 1 \
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

CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
setsid uv run --active --no-sync --extra gpu --extra tinker --extra jax \
  -m skyrl.tinker.api \
  --base-model /scr1/public_models/huggingface/Qwen/Qwen3.5-2B-Base \
  --backend jax \
  --port 8001 \
  --external-inference-url http://localhost:8000 \
  --external-inference-lora-base /tmp/qwen35_2b_jax_settings_lora_models \
  --checkpoints-base /tmp/skyrl_qwen35_2b_jax_settings_checkpoints \
  --database-url 'postgresql://postgres@/skyrl_tinker?host=/tmp/skyrl_tinker_pg_data' \
  --backend-config '{"max_lora_adapters":1,"max_lora_rank":64,"tensor_parallel_size":1,"fully_sharded_data_parallel_size":2,"train_micro_batch_size":1,"sample_max_num_sequences":16,"gradient_checkpointing":true}' \
  > logs/qwen35_2b_jax_settings_tinker.log 2>&1 < /dev/null &
```

Wait for Tinker:

```bash
until curl -fsS -m 5 http://localhost:8001/api/v1/get_server_capabilities >/dev/null; do
  sleep 2
done
echo tinker_ok
```

## 5. Verify

```bash
PG_BIN=/scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

pgrep -af '[s]kyrl.tinker|[v]llm serve /scr1/public_models/huggingface/Qwen/Qwen3.5-'
$PG_BIN/pg_ctl -D /tmp/skyrl_tinker_pg_data status | head -1
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
PG_BIN=/scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

# On-disk artifacts
rm -rf /tmp/qwen35_2b_jax_settings_lora_models
rm -rf /tmp/skyrl_qwen35_2b_jax_settings_checkpoints

# Drop + recreate the tinker database (clears all sessions, futures, checkpoints)
$PG_BIN/psql -h /tmp/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "DROP DATABASE IF EXISTS skyrl_tinker;"
$PG_BIN/psql -h /tmp/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "CREATE DATABASE skyrl_tinker;"
```

To unstick zombie `PENDING` futures without dropping everything (the most common
case after an SDK timeout) — flip them to `FAILED` so the client unblocks:

```bash
PG_BIN=/scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin
$PG_BIN/psql -h /tmp/skyrl_tinker_pg_data -U postgres -d skyrl_tinker -c "
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

- vLLM had enough headroom on GPU 3, so do not use the most aggressive vLLM memory reductions by default. Start with `--max-model-len 4096`, `--max-loras 2`, and no `--enforce-eager`; reduce these only if the sampling server itself OOMs.
- Keep Tinker at the smallest workable adapter pool. `max_lora_adapters=1` can reject client creation in this setup, so use `2` rather than a larger value.
- Keep `train_micro_batch_size=1`, `sample_max_num_sequences=8`, `gradient_checkpointing=true`, and `loss_chunk_size=128`.
- Keep `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` so JAX does not grab the entire card up front.
- Do not enable offloading by default. Use it only after the above settings still produce a real OOM.

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
  --max-model-len 4096 \
  --dtype bfloat16 \
  > logs/qwen35_4b_vllm.log 2>&1 < /dev/null &
```

Training server:

```bash
cd /scr1/michael/SkyRL
mkdir -p logs

CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
setsid uv run --active --no-sync --extra gpu --extra tinker --extra jax \
  -m skyrl.tinker.api \
  --base-model /scr1/public_models/huggingface/Qwen/Qwen3.5-4B-Base \
  --backend jax \
  --port 8001 \
  --external-inference-url http://localhost:8000 \
  --external-inference-lora-base /tmp/qwen35_4b_jax_settings_lora_models \
  --checkpoints-base /tmp/skyrl_qwen35_4b_jax_settings_checkpoints \
  --database-url 'postgresql://postgres@/skyrl_tinker?host=/tmp/skyrl_tinker_pg_data' \
  --backend-config '{"max_lora_adapters":2,"max_lora_rank":64,"tensor_parallel_size":2,"fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,"sample_max_num_sequences":8,"gradient_checkpointing":true,"loss_chunk_size":128}' \
  > logs/qwen35_4b_jax_settings_tinker.log 2>&1 < /dev/null &
```

Optional clean 4B state. Both 2B and 4B share the same `skyrl_tinker` database,
so dropping it clears 2B state too — only do this if you actually want a clean
slate for the database side. The on-disk artifacts below are 4B-specific:

```bash
PG_BIN=/scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

rm -rf /tmp/qwen35_4b_jax_settings_lora_models
rm -rf /tmp/skyrl_qwen35_4b_jax_settings_checkpoints

# Optional: drop the shared tinker DB (affects 2B too)
$PG_BIN/psql -h /tmp/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "DROP DATABASE IF EXISTS skyrl_tinker;"
$PG_BIN/psql -h /tmp/skyrl_tinker_pg_data -U postgres -d postgres \
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

CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
setsid uv run --active --no-sync --extra gpu --extra tinker --extra jax \
  -m skyrl.tinker.api \
  --base-model /scr1/public_models/huggingface/Qwen/Qwen3.5-9B-Base \
  --backend jax \
  --port 8001 \
  --external-inference-url http://localhost:8000 \
  --external-inference-lora-base /tmp/qwen35_9b_jax_settings_lora_models \
  --checkpoints-base /tmp/skyrl_qwen35_9b_jax_settings_checkpoints \
  --database-url 'postgresql://postgres@/skyrl_tinker?host=/tmp/skyrl_tinker_pg_data' \
  --backend-config '{"max_lora_adapters":4,"max_lora_rank":64,"tensor_parallel_size":2,"fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,"sample_max_num_sequences":4,"gradient_checkpointing":true,"loss_chunk_size":128}' \
  > logs/qwen35_9b_jax_settings_tinker.log 2>&1 < /dev/null &
```

Optional clean 9B state. The 2B, 4B, and 9B variants all share the same
`skyrl_tinker` database, so dropping it clears the others too — only do this
if you actually want a clean slate for the database side. The on-disk
artifacts below are 9B-specific:

```bash
PG_BIN=/scr1/michael/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

rm -rf /tmp/qwen35_9b_jax_settings_lora_models
rm -rf /tmp/skyrl_qwen35_9b_jax_settings_checkpoints

# Optional: drop the shared tinker DB (affects 2B and 4B too)
$PG_BIN/psql -h /tmp/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "DROP DATABASE IF EXISTS skyrl_tinker;"
$PG_BIN/psql -h /tmp/skyrl_tinker_pg_data -U postgres -d postgres \
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
