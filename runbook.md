# Tinker Server Runbook

Qwen3.5 LoRA training (JAX) + vLLM sampling on 4× RTX-5000 Ada (32 GB each, sm_89, PCIe — no NVLink).

| Service | GPUs | Port | Process |
|---------|------|------|---------|
| Trainer (JAX) | 0, 1 | 8001 | `skyrl.tinker.api` |
| Sampler (vLLM, TP=2) | 2, 3 | 8000 | `vllm serve` |

Clients connect to `http://127.0.0.1:8001`. Sampling is forwarded internally; clients never touch port 8000.

**No authentication.** Both servers bind `127.0.0.1`. For remote access: `ssh -N -L 8001:127.0.0.1:8001 <host>`.

---

## Quick start (9B)

Run steps 1–4 in order. Steps 2 and 3 can start concurrently (they use different GPU pairs).

### 1. Start PostgreSQL

Tinker requires Postgres (SQLite deadlocks under concurrent `asample` + `save_weights`).

```bash
REPO=/storage_slow/ajoe/code/SkyRL
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
PG_BIN=$REPO/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin
```

**First time only** (skip if `$STATE_DIR/skyrl_tinker_pg_data` exists):

```bash
cd $REPO && uv pip install pgserver
mkdir -p $STATE_DIR/skyrl_tinker_pg_data
$PG_BIN/initdb -D $STATE_DIR/skyrl_tinker_pg_data -U postgres --auth=trust
```

**Every launch** (idempotent):

```bash
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

Postgres stays up across trainer restarts. Never stop it in step 5.

> **Warning:** Do not `import pgserver` in Python while this Postgres is running — it registers an `atexit` handler that kills the postmaster on interpreter exit.

### 2. Start the sampler (vLLM)

vLLM 0.25.1 installed in an isolated env (independent of the repo's pinned 0.20.2). Startup takes 2–4 minutes.

```bash
REPO=/storage_slow/ajoe/code/SkyRL
MODEL=/scr1/public_models/huggingface/Qwen/Qwen3.5-9B-Base
cd $REPO && mkdir -p logs

CUDA_VISIBLE_DEVICES=2,3 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True NCCL_NET=Socket \
setsid uv run --isolated \
  --with "vllm==0.25.1" --index https://wheels.vllm.ai/0.25.1/cu130 \
  vllm serve $MODEL \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 2 \
  --enable-lora --max-lora-rank 64 --max-loras 2 \
  --max-model-len 4096 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-num-seqs 256 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  > logs/qwen35_9b_vllm.log 2>&1 < /dev/null &

until curl -fsS -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done
echo vllm_ok
```

Verify text-only mode in the log (saves ~27% throughput):

```bash
grep "running in text-only mode" logs/qwen35_9b_vllm.log
```

### 3. Start the trainer (JAX)

Startup takes ~5 s for the API, then ~2 min to load weights (warm compilation cache).

```bash
REPO=/storage_slow/ajoe/code/SkyRL
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
TAG=9b
MODEL=/scr1/public_models/huggingface/Qwen/Qwen3.5-9B-Base
MEM_FRAC=0.90
BACKEND_CONFIG='{"max_lora_adapters":1,"max_lora_rank":64,"tensor_parallel_size":2,"fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,"gradient_checkpointing":true,"loss_chunk_size":128,"train_bucket_seq_len_per_micro_batch":true}'

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
  --host 127.0.0.1 \
  --port 8001 \
  --external-inference-url http://127.0.0.1:8000 \
  --external-inference-lora-base /dev/shm/$USER/qwen35_${TAG}_lora_models \
  --checkpoints-base $STATE_DIR/skyrl_qwen35_${TAG}_checkpoints \
  --database-url "postgresql://postgres@/skyrl_tinker?host=$STATE_DIR/skyrl_tinker_pg_data" \
  --backend-config "$BACKEND_CONFIG" \
  > logs/qwen35_${TAG}_tinker.log 2>&1 < /dev/null &

until curl -fsS -m 5 http://127.0.0.1:8001/api/v1/get_server_capabilities >/dev/null; do sleep 5; done
echo tinker_ok
```

### 4. Verify

```bash
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
PG_BIN=/storage_slow/ajoe/code/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

# All three services alive
$PG_BIN/pg_ctl -D $STATE_DIR/skyrl_tinker_pg_data status | head -1
curl -fsS -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo vllm_ok
curl -fsS -m 5 http://127.0.0.1:8001/api/v1/get_server_capabilities >/dev/null && echo tinker_ok

# Must show 127.0.0.1, never 0.0.0.0
ss -ltnp | grep -E ':(8000|8001)\b'

# GPU memory
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# KV cache size (throughput constraint)
grep -oE "GPU KV cache size: [0-9,]+ tokens" logs/qwen35_9b_vllm.log | tail -1
```

---

## Stop, restart, and reset

### Stop processes

**Always kill by PID in a standalone command.** Two real failure modes:
1. An unescaped pattern like `skyrl.tinker` matches the Postgres command line (`skyrl_tinker_pg_data`) and kills the database.
2. Putting `pgrep` and the relaunch in the same command matches the shell's own arguments and self-kills (exit 144).

```bash
# Step 1: collect PIDs (must not mention launch arguments)
PIDS=$(pgrep -f 'skyrl\.tinker\.(api|engine)' | tr '\n' ' ')

# Step 2: kill (separate invocation)
kill $PIDS
for i in $(seq 1 30); do pgrep -f 'skyrl\.tinker\.(api|engine)' >/dev/null || break; sleep 2; done

# Step 3: relaunch (third invocation)
```

Same pattern for vLLM: `pgrep -f 'vllm serve /scr1/public_models'`.

**Do not run this while `pytest tests/tinker/` is running** — those tests spawn their own `skyrl.tinker.api` processes.

Never stop Postgres.

### When to restart

| Restart | When |
|---------|------|
| **Trainer** | After any JAX OOM (allocator high-water never drops; returns HTTP 400 without dying). Also clears leaked adapter slots. |
| **vLLM** | When sampling stalls, or periodically to drop accumulated LoRA adapters. Also `rm -rf /dev/shm/$USER/qwen35_${TAG}_lora_models`. |

### Reset state

Stop the trainer first.

```bash
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
PG_BIN=/storage_slow/ajoe/code/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin
TAG=9b

# Clear on-disk artifacts
rm -rf /dev/shm/$USER/qwen35_${TAG}_lora_models
rm -rf $STATE_DIR/skyrl_qwen35_${TAG}_checkpoints

# Drop and recreate the database (shared across all model variants)
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "DROP DATABASE IF EXISTS skyrl_tinker;"
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "CREATE DATABASE skyrl_tinker;"
```

To unblock a client stuck on a `PENDING` future without dropping the database:

```bash
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d skyrl_tinker -c "
UPDATE futures
SET status = 'FAILED',
    result_data = '{\"error\":\"Marked FAILED by operator\",\"status\":\"failed\"}',
    completed_at = now()
WHERE status = 'PENDING';
SELECT status, COUNT(*) FROM futures GROUP BY status;"
```

---

## Configuration reference

### Backend configs by model

| Model | MEM_FRAC | BACKEND_CONFIG |
|-------|----------|----------------|
| 2B | 0.95 | `{"max_lora_adapters":1,"max_lora_rank":64,"tensor_parallel_size":1,"fully_sharded_data_parallel_size":2,"train_micro_batch_size":1,"gradient_checkpointing":true}` |
| 4B | 0.90 | `{"max_lora_adapters":1,"max_lora_rank":64,"tensor_parallel_size":2,"fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,"gradient_checkpointing":true,"loss_chunk_size":128,"train_bucket_seq_len_per_micro_batch":true}` |
| 9B | 0.90 | `{"max_lora_adapters":1,"max_lora_rank":64,"tensor_parallel_size":2,"fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,"gradient_checkpointing":true,"loss_chunk_size":128,"train_bucket_seq_len_per_micro_batch":true}` |

Model paths: `/scr1/public_models/huggingface/Qwen/Qwen3.5-{2B,4B,9B}-Base`.

### Variables

- **`STATE_DIR`** — durable state (pgdata, checkpoints, JAX cache). Must be the same path every launch. Use `/storage_slow/$USER`, not `/tmp`.
- **Sampler LoRA adapters** go on tmpfs (`/dev/shm/$USER/...`), not `STATE_DIR`. vLLM never unloads them; `rm -rf` when bouncing vLLM.
- **Virtualenv** — needs `tinker`, `jax`, and `gpu` extras. If missing, you'll get `ModuleNotFoundError: sqlalchemy`.

### Key environment variables (trainer)

All four are required:

| Variable | Why |
|----------|-----|
| `XLA_PYTHON_CLIENT_PREALLOCATE=true` | Reserves memory as one contiguous slab. Without it, fragmented regions cause OOM on the ~10 GiB `forward_backward` workspace. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 0.90 for 4B/9B, 0.95 for 2B. **Not 0.95 for 4B/9B** — XLA command buffers live outside the pool and need ~3 GiB headroom, or the process aborts with exit 134. |
| `JAX_COMPILATION_CACHE_DIR` | Each sequence-length bucket costs ~170 s to compile. Without the cache, every restart re-pays the full cost. Safe to `rm -rf`. |
| `NCCL_NET=Socket` | Works around NCCL 2.28.9 segfault on any 2-GPU collective on this box. Also needed for vLLM at TP=2. |

### Critical constraints

- **TP=2 is required for 4B/9B; do not use FSDP.** The 248K-vocab `lm_head` logits replicate under FSDP instead of sharding, doubling the transient and OOMing.
- **Max trainable sequence length: 4096 tokens.** The 6144 bucket OOMs. Cap client sequences accordingly.
- **`gradient_checkpointing: true` is required for 9B** — full activations at 4096 tokens do not fit without it.
- **`--limit-mm-per-prompt '{"image":0,"video":0}'` is required for the 9B sampler** — the vision encoder buffer OOMs the card during profiling without it.
- **Do not enable the XLA latency-hiding scheduler** — it inflates startup memory from ~8.5 to ~16 GiB/card.
- **Do not use `--enable-prefix-caching`** on this hybrid architecture — it costs 15% of KV cache with 0% hit rate for this workload.
- **Do not use MTP speculative decoding** on the 4B/9B sampler — it's a 46% regression at production concurrency (compute-bound regime, unlike the 27B which is bandwidth-bound).

### LoRA adapter slots

`max_lora_adapters` sets the number of trainable adapters. With `--external-inference-url` (always used in this runbook), all N slots go to user models — there is no base-model slot to subtract.

Each slot costs 1.68 GiB/GPU (stacked LoRA weights + fp32 grad/mu/nu at rank 64, 9B TP=2). Set it to the number of concurrent training clients you need.

Without an external sampler, slot 0 is reserved for the base model, so N slots serve only N-1 models.

### Attention backend

Qwen3.5's `head_dim=256` exceeds cuDNN's cap of 128 on Ada (sm_89). Training falls back to Pallas/Triton. Leave `SKYRL_CUDNN_MAX_HEAD_DIM` unset on this box.

On Hopper or newer (sm_90+), set `SKYRL_CUDNN_MAX_HEAD_DIM=256` for the native cuDNN kernel.

---

## Troubleshooting

### `Maximum number of LoRA adapters (N) reached`

First: this may be the cap doing its job. `max_lora_adapters: 1` allows exactly one concurrent client.

If a previous client died, its slot is reclaimed by session expiry (~6 min). To clear immediately, restart the trainer.

**Before restarting**, check for stale queued requests that would replay on startup:

```bash
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d skyrl_tinker \
  -c "SELECT request_type, status, count(*) FROM futures WHERE status='PENDING' GROUP BY 1,2;"
```

If any `CREATE_MODEL` is `PENDING`, mark it failed (use the `UPDATE futures` command from the reset section) before relaunching.

### vLLM won't start

Usually a previous server still holding the GPU. Check and kill it:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

If the GPU is genuinely free, lower `--gpu-memory-utilization` in steps of 0.04.

Don't grep the log for readiness — benign warnings fire before the port binds. Poll `/v1/models`.

### Sampling stalls or vLLM memory creeps

vLLM never unloads sampler adapters (25 observed in one run). Restart vLLM and `rm -rf /dev/shm/$USER/qwen35_${TAG}_lora_models`.

### `sample_max_num_sequences` has no effect

With `--external-inference-url`, sample requests bypass the engine's batching entirely. Throttle via `forwarding_inference_max_connections`, vLLM's `--max-num-seqs`, or client-side.

### `Number of devices 1 must be >= the product of mesh_shape`

JAX found only CPU devices. Check the log for:

```text
WARNING: ... a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

The GPU backend comes from `--extra gpu`, not `--extra jax`. Fix:

```bash
uv run --active --no-sync python -c "import jax; print(jax.devices())"
# Must show [CudaDevice(id=0), CudaDevice(id=1)]

uv pip install --python .venv/bin/python \
  "jax[cuda12]==$(uv run --active --no-sync python -c 'import jax; print(jax.__version__)')"
```

### Diagnosing a trainer OOM

Stop the trainer, add these env vars to the launch, and reproduce:

```bash
rm -rf $STATE_DIR/skyrl_xla_dump && mkdir -p $STATE_DIR/skyrl_xla_dump

# Add to the trainer launch environment:
TF_CPP_MIN_LOG_LEVEL=0 \
TF_CPP_VMODULE=bfc_allocator=2 \
JAX_TRACEBACK_FILTERING=off \
XLA_FLAGS="--xla_dump_to=$STATE_DIR/skyrl_xla_dump --xla_dump_hlo_as_long_text --xla_dump_hlo_as_text" \
```

After OOM:

```bash
grep -nE "ran out of memory|requested by op|Sum Total" logs/qwen35_${TAG}_tinker.log | tail -40
ls -S $STATE_DIR/skyrl_xla_dump/*-memory-usage-report.txt | head
head -15 "$(ls -S $STATE_DIR/skyrl_xla_dump/*-memory-usage-report.txt | head -1)"
```

### Do not bump repo dependencies for newer vLLM

The sampler runs vLLM 0.25.1 out-of-tree via `uv run --isolated`. `uv lock --upgrade` would move torch 2.11→2.13, breaking three hand-built wheels (`flash-attn`, `causal-conv1d`, `mamba-ssm`) pinned to the torch 2.11 ABI. It would also not actually move vLLM (it's an exact `==0.20.2` pin).

---

## 2B sampler (alternate)

The 2B uses the repo's pinned vLLM 0.20.2 on a single card:

```bash
REPO=/scr1/michael/SkyRL
TAG=2b
MODEL=/scr1/public_models/huggingface/Qwen/Qwen3.5-2B-Base
cd $REPO && mkdir -p logs

CUDA_VISIBLE_DEVICES=3 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
setsid uv run --no-sync --extra fsdp vllm serve $MODEL \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 \
  --enable-lora --max-lora-rank 64 --dtype bfloat16 \
  --max-loras 2 --max-model-len 4096 \
  > logs/qwen35_${TAG}_vllm.log 2>&1 < /dev/null &

until curl -fsS -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done
echo vllm_ok
```

---

## Other environments

### H100 box (3× H100 80 GB)

GPUs 0,1 = trainer (TP=2); GPU 2 = vLLM (TP=1). Up to 4 concurrent clients.

Key differences from local:
- `SKYRL_CUDNN_MAX_HEAD_DIM=256` (sm_90)
- `TMPDIR=/root/tmp`, `UV_PROJECT_ENVIRONMENT=.venv-jax`, `VLLM_USE_DEEP_GEMM=0`
- Trainer: `max_lora_adapters: 4`, `loss_chunk_size: 64` (halved to fit 4 concurrent `forward_backward`)
- vLLM: `--max-loras 9 --max-num-seqs 512 --max-num-batched-tokens 16384 --max-model-len 4096 --limit-mm-per-prompt '{"image":0,"video":0}'`

### Modal B200 (single GPU, 192 GB)

`tools/modal_tinker_server.py` deploys trainer + sampler + auth proxy in one container. 8 trainable LoRA slots.

```bash
modal run    tools/modal_tinker_server.py::download_model   # once
modal run    tools/modal_tinker_server.py::smoke            # before first deploy
modal deploy tools/modal_tinker_server.py
```

Client config:

```bash
export TINKER_API_KEY=tml-...
export TINKER_BASE_URL=https://<workspace>--skyrl-tinker-server.modal.run
```

Memory split (both processes share one GPU):

| | Fraction | Purpose |
|---|---|---|
| Trainer | 0.45 | weights + 8 LoRAs + activations |
| Sampler | 0.40 | weights + LoRA + KV cache |
| Free | 0.15 | XLA command buffers (exhaustion → exit 134) |

Operations:

```bash
modal app list                               # is it up?
curl -s $TINKER_BASE_URL/_status             # boot stage
modal app logs skyrl-tinker
modal app stop --yes skyrl-tinker            # --yes required (no tty → abort)
```

**Warning:** deployed with `min_containers=1`, so it bills continuously until explicitly stopped.

Checkpoints: `tools/export_tinker_checkpoints.py` — read its docstring first. The `.tar.gz` entries are directories (Orbax format), not tarballs.

**Known failure:** vLLM sampler can wedge silently, causing all clients to die after the SDK's 7200 s no-progress timeout. Monitor step progress, not process liveness.

### 27B defender (Qwen3.6-27B-NVFP4)

Model: `/scr1/public_models/huggingface/nvidia/Qwen3.6-27B-NVFP4` (21 GB, single GPU).

```bash
CUDA_VISIBLE_DEVICES=3 setsid uv run --isolated \
  --with "vllm==0.25.1" --index https://wheels.vllm.ai/0.25.1/cu130 \
  vllm serve /scr1/public_models/huggingface/nvidia/Qwen3.6-27B-NVFP4 \
  --served-model-name Qwen3.6-27B-NVFP4 \
  --host 127.0.0.1 --port 8002 \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --async-scheduling \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","model":"/scr1/public_models/huggingface/nvidia/Qwen3.6-27B-NVFP4","num_speculative_tokens":1}' \
  > logs/qwen36_27b_nvfp4_vllm.log 2>&1 < /dev/null &

until curl -fsS -m 5 http://127.0.0.1:8002/v1/models >/dev/null; do sleep 5; done
echo vllm_ok
```

Key notes:
- Use `0.92`, not `0.96` — the GDN `causal_conv1d` prefill transient OOMs under burst load at 0.96, killing the engine entirely.
- MTP gives +63% throughput (bandwidth-bound at 95% of the card's 576 GB/s without it).
- No `--enable-prefix-caching` — costs 15% KV with 0% hit rate on this workload.
- Requires vLLM ≥ 0.23.0 (0.20.2 cannot load NVFP4 + quantized `lm_head`).
- Cold start ~6 min; warm ~2 min.
- Thinking is on by default. Disable per-request: `"chat_template_kwargs": {"enable_thinking": false}`.
- Reasoning field is `message.reasoning` (not `reasoning_content`).

For two-replica load balancing (1.87x throughput):

```bash
# Launch replica B on GPU 2, port 8004 (same command, different CUDA_VISIBLE_DEVICES and port)

cd /storage_slow/ajoe/code/SkyRL
setsid uv run --isolated --with aiohttp python tools/defender_lb.py \
  --port 8002 --backend http://127.0.0.1:8003 --backend http://127.0.0.1:8004 \
  > logs/defender_lb.log 2>&1 < /dev/null &

# Verify
curl -fsS http://127.0.0.1:8002/lb_stats
```

Balancer `--max-connections` (default 512) must exceed the client's `max_concurrency`.

---

## Performance notes

Measurements taken 2026-07-24, 4B model unless stated.

### Sequence-length bucketing (1.42x trainer speedup)

`train_bucket_seq_len_per_micro_batch: true` pads each micro-batch independently instead of to the batch maximum. Production sequences average ~2,450 tokens but the batch maximum is usually 4096, so ~40% of compute was padding.

Costs: ~7 XLA executables instead of 1 (~170 s compilation each, cached by `JAX_COMPILATION_CACHE_DIR`), and ~1.4 GiB more command-buffer memory. If the trainer crashes with `Retry CUDA graph instantiation after OOM`, lower `MEM_FRAC` to 0.87.

### Text-only sampler mode (+27% throughput)

`--limit-mm-per-prompt '{"image":0,"video":0}'` disables the vision encoder, freeing ~107% more KV cache and +27% peak throughput versus multimodal-enabled.

### Co-residency cost (historical)

Lowering trainer `MEM_FRAC` from 0.90 to 0.65 (to share GPUs with the sampler) costs 4.1% on `forward_backward`. Co-residency with the sampler itself adds no measurable overhead beyond the smaller pool.

### Rejected sampler settings

- **MTP on 4B/9B:** −46% at 128 seqs (compute-bound regime; MTP only helps bandwidth-bound models like the 27B).
- **Prefix caching:** 0% hit rate with `num_samples=16` (prefill shared across samples). Costs 15% KV.
- **`sample_max_num_sequences`:** no effect with external inference (requests bypass engine batching).
