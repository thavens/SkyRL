# Tinker Server Runbook

Launching **attacker Tinker servers** (Qwen3.5 LoRA training, JAX backend + vLLM sampling):

- **Local**: 4× RTX 5000 Ada (32 GB, sm_89, PCIe — no NVLink). 9B primary; 2B/4B variants.
- **Remote**: H100 box (3× H100 80 GB) and Modal (1× B200 192 GB).

Every flag and config below is load-bearing: verified against the code and (for the local 9B
stack) re-measured 2026-08-11. Optimizations with no measured benefit were removed.

| Service | GPUs | Port | Process |
|---------|------|------|---------|
| Trainer (JAX) | 0, 1 | 8001 | `skyrl.tinker.api` |
| Sampler (vLLM 0.25.1, TP=2) | 2, 3 | 8000 | `vllm serve` |

Clients connect to `http://127.0.0.1:8001`. Sampling is forwarded internally; clients never
touch port 8000.

**No authentication.** Both servers bind `127.0.0.1`. Remote access: `ssh -N -L 8001:127.0.0.1:8001 <host>`.

> **`--host 127.0.0.1` is load-bearing — never omit it.** `skyrl.tinker.api` defaults to
> `0.0.0.0`, has no auth, and can write files anywhere under `--checkpoints-base`. Every
> launch here passes it explicitly; the `ss -ltnp` check in the verify step catches a launch
> that didn't. The only deployment safe to expose is Modal, which fronts the API with an
> `X-API-Key` proxy and keeps the API itself on loopback.

> **Shared box.** Other users run jobs on these GPUs. Before any launch, confirm the target
> GPUs are yours to take: `nvidia-smi --query-compute-apps=pid --format=csv,noheader`, then
> `ps -o user,cmd -p <pid>`. Never kill a PID you have not identified as your own.

---

## Quick start (9B)

Run steps 1–4 in order. Steps 2 and 3 can start concurrently (different GPU pairs).

### 1. Start PostgreSQL

Tinker needs Postgres: the API (async) and engine (separate process, sync) write the same
database, and SQLite stalls on its 30 s busy-timeout under concurrent `asample` + `save_weights`.

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

Postgres stays up across trainer restarts. Never stop it.

> **Warning:** Do not `import pgserver` in Python while this Postgres is running — it
> registers an `atexit` handler that kills the postmaster on interpreter exit.

### 2. Start the sampler (vLLM)

vLLM 0.25.1 in an isolated env (the repo itself pins 0.23.0; neither is used here).
Startup takes 2–4 minutes.

```bash
REPO=/storage_slow/ajoe/code/SkyRL
TAG=9b
MODEL=/scr1/public_models/huggingface/Qwen/Qwen3.5-9B-Base
cd $REPO && mkdir -p logs

CUDA_VISIBLE_DEVICES=2,3 NCCL_NET=Socket \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
VLLM_LORA_RESOLVER_CACHE_DIR=/dev/shm/$USER/qwen35_${TAG}_lora_models \
setsid uv run --isolated \
  --with "vllm==0.25.1" --index https://wheels.vllm.ai/0.25.1/cu130 \
  vllm serve $MODEL \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 2 \
  --enable-lora --max-lora-rank 64 --max-loras 2 \
  --max-model-len 5120 --dtype bfloat16 --max-logprobs 128 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.90 --max-num-seqs 256 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  > logs/qwen35_${TAG}_vllm.log 2>&1 < /dev/null &

until curl -fsS -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done
echo vllm_ok
```

Why each non-default flag is there (all re-verified 2026-08-11 on vLLM 0.25.1):

| Flag / env var | Reason |
|---|---|
| `VLLM_LORA_RESOLVER_CACHE_DIR` + `--enable-lora` | Adapters load **by name, on demand** via vLLM's filesystem resolver from this directory — the trainer only publishes a directory and passes its name. Without the cache dir, every LoRA sample 404s. |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` | Attaches `/v1/unload_lora_adapter`, which the trainer's quiet-adapter sweep calls. Without it, adapters accumulate until vLLM is restarted (25 observed in one run). |
| `NCCL_NET=Socket` | NCCL 2.28.9 segfaults on any 2-GPU collective on this box (re-verified: removal = SIGSEGV). Needed at TP=2. |
| `--max-num-batched-tokens 1024` | Caps the prompt-logprobs fp32 transient (chunk × 248,320 vocab × 4 B). At 2048 a single 4k-token `prompt_logprobs` request allocates 1.89 GiB and **kills the engine** (reproduced). At 1024 the same load survives; throughput cost vs 2048 is ~1%. Do not raise it. |
| `--gpu-memory-utilization 0.90` | +10.6% KV cache vs 0.85 (845k vs 764k tokens) at identical throughput; survived 24 concurrent 4k-token logprob probes under decode load. (vLLM ≥0.21 counts CUDA-graph memory inside this budget, so 0.90 today ≈ 0.87 in pre-0.21 terms.) |
| `--max-num-seqs 256` | 128 loses ~13% throughput at 256-seq offered load; 256 measured 2,733 tok/s aggregate. |
| `--max-logprobs 128` | vLLM default is 20; client `topk_prompt_logprobs` above the cap = 400 per request. |
| `--max-model-len 5120` | Must exceed the trainer's 4096 cap: probing prompt logprobs on a max-length trajectory needs prompt + 1 token, so equal caps 400 every end-of-trajectory probe. |
| `--limit-mm-per-prompt '{"image":0,"video":0}'` | **Required** — vLLM 0.25.1 fails to boot on this model without it (crash in multimodal profiling init). Also saves the vision encoder's memory/KV. |

Adapters live on tmpfs (`/dev/shm`) on purpose; see "Expected warnings" below.

### 3. Start the trainer (JAX)

~5 s for the API, then ~2 min to load weights (with a warm compilation cache).

```bash
REPO=/storage_slow/ajoe/code/SkyRL
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
TAG=9b
MODEL=/scr1/public_models/huggingface/Qwen/Qwen3.5-9B-Base
MEM_FRAC=0.90
BACKEND_CONFIG='{"max_lora_adapters":1,"max_lora_rank":64,"tensor_parallel_size":2,"fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,"gradient_checkpointing":true,"loss_chunk_size":0,"linear_attention_chunk_size":128,"train_bucket_seq_len_per_micro_batch":true}'

cd $REPO && mkdir -p logs

CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRAC \
JAX_COMPILATION_CACHE_DIR=$STATE_DIR/skyrl_jax_compilation_cache \
XLA_FLAGS="--xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false" \
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

Launch rules baked into the code:

- **Must be launched via `uv run … -m skyrl.tinker.api`** — the API reconstructs its uv flags
  from the parent command line to spawn the engine subprocess, and refuses to start otherwise.
  The GPU backend comes from `--extra gpu` (it carries `jax[cuda12]`), not `--extra jax`.
- **`--port 8001` is not optional**: the API's default port is 8000 — the sampler's port.
- **`--checkpoints-base` and `--external-inference-lora-base` default to `/tmp`** — forgetting
  either silently writes state to `/tmp`.
- **`--backend-config` keys are `extra="forbid"`**: a typo'd key is a hard engine-startup
  failure, which takes the API down with it (see Troubleshooting).
- Dead-client LoRA slots free after `--session-timeout-sec` (default 300 s) + up to
  `--session-cleanup-interval-sec` (default 60 s). Lower the timeout instead of restarting
  the trainer if slot churn is frequent.

**Expected warnings at boot — ignore both:**

1. *"external_inference_lora_base ... appears to be ephemeral storage"* — deliberate. Sampler
   adapters are per-step and disposable; tmpfs keeps publish latency off the training path. A
   restart 410s old sampler checkpoints, which clients re-publish.
2. *"sample_max_num_sequences is not set ... can lead to OOMs"* — inert with
   `--external-inference-url`: forwarded samples bypass the engine's batching entirely.

### 4. Verify

```bash
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
PG_BIN=/storage_slow/ajoe/code/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

$PG_BIN/pg_ctl -D $STATE_DIR/skyrl_tinker_pg_data status | head -1
curl -fsS -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo vllm_ok
curl -fsS -m 5 http://127.0.0.1:8001/api/v1/get_server_capabilities >/dev/null && echo tinker_ok

# Must show 127.0.0.1, never 0.0.0.0
ss -ltnp | grep -E ':(8000|8001)\b'

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Sampler capacity and text-only mode
grep -oE "GPU KV cache size: [0-9,]+ tokens" logs/qwen35_9b_vllm.log | tail -1   # expect ~845k
grep -c "running in text-only mode" logs/qwen35_9b_vllm.log                      # expect >=1
```

---

## Stop, restart, and reset

### Stop processes

**Always kill by PID in a standalone command.** Three real failure modes:

1. An unescaped pattern like `skyrl.tinker` matches the Postgres command line
   (`skyrl_tinker_pg_data`) and kills the database.
2. Putting `pgrep` and the relaunch in the same command matches the shell's own arguments
   and self-kills (exit 144).
3. On a shared box, a pattern can match **another user's process**.

```bash
# Step 1: collect PIDs (must not mention launch arguments), and eyeball them
PIDS=$(pgrep -u $USER -f 'skyrl\.tinker\.(api|engine)' | tr '\n' ' ')
ps -o pid,user,cmd -p $PIDS

# Step 2: kill (separate invocation)
kill $PIDS
for i in $(seq 1 30); do pgrep -u $USER -f 'skyrl\.tinker\.(api|engine)' >/dev/null || break; sleep 2; done

# Step 3: relaunch (third invocation)
```

Same pattern for vLLM: `pgrep -u $USER -f 'vllm serve /scr1/public_models'`. After killing
vLLM, check for a surviving `VLLM::EngineCore` process — on 0.25.1 it can outlive its process
group — and kill it by its PID:

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # then ps -o user,cmd -p <pid>
```

**Do not stop anything while `pytest tests/tinker/` is running** — those tests spawn their own
`skyrl.tinker.api` processes on fixed ports.

Never stop Postgres.

### When to restart

| Restart | When |
|---------|------|
| **Trainer** | After any JAX OOM (allocator high-water never drops; the request itself returns HTTP 400 without killing the server). Also clears leaked adapter slots — though `--session-timeout-sec` handles those without a restart. |
| **vLLM** | When sampling stalls. Also `rm -rf /dev/shm/$USER/qwen35_${TAG}_lora_models` while it's down — published adapter directories are never garbage-collected. |

### Reset state

Stop the trainer first.

```bash
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
PG_BIN=/storage_slow/ajoe/code/SkyRL/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin
TAG=9b

rm -rf /dev/shm/$USER/qwen35_${TAG}_lora_models
rm -rf $STATE_DIR/skyrl_qwen35_${TAG}_checkpoints

# The database is shared across all model variants
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "DROP DATABASE IF EXISTS skyrl_tinker;" \
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

| Model | MEM_FRAC | BACKEND_CONFIG deltas from the 9B quick start |
|-------|----------|-----------------------------------------------|
| 2B | 0.95 | `"tensor_parallel_size":1,"fully_sharded_data_parallel_size":2`; drop `loss_chunk_size`/`linear_attention_chunk_size`/bucketing keys |
| 4B | 0.90 | none (same shape as 9B) |
| 9B | 0.90 | — (the quick-start config) |
| 9B, 2 LoRAs | 0.90 | `"max_lora_adapters":2,"max_lora_rank":32,"loss_chunk_size":256` (measured 1,538 tok/s, peak 26.6–27.1 GiB of the 28.3 GiB pool — fits, verified) |

Model paths: `/scr1/public_models/huggingface/Qwen/Qwen3.5-{2B,4B,9B}-Base`.

Tuning that is measured, not guessed (9B, TP=2, 2026-08-11; tok/s = trained tokens/s per request):

| Knob | Setting | Why |
|---|---|---|
| `loss_chunk_size` | **0 (disabled)** for 1 slot | +9% over the old 128 at both 2048 and 4096 tokens; peak memory +0.6 GiB, irrelevant at 1 slot. Use **256** for the 2-slot config where memory is tight (chunk 64 costs 7%). |
| `linear_attention_chunk_size` | **128** (default 64) | +2% tok/s **and** −1.0 GiB peak. 32 is worse on both axes. |
| `train_bucket_seq_len_per_micro_batch` | true | 1.42× on production mixed-length batches (2026-07-24). Costs ~7 XLA executables (~170 s each to compile, cached) + ~1.4 GiB command buffers. |
| `gradient_checkpointing` | true | Required at 9B: without it the 4096-token backward tries a 33 GiB allocation (reproduced). |
| `train_micro_batch_size` | 1 | Compute is per-sequence; larger values only change padding behavior. |

### Environment variables (trainer) — all five required

| Variable | Why (all re-verified 2026-08-11) |
|----------|-----|
| `XLA_FLAGS=--xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false` | Removal reproduces `INVALID_ARGUMENT: Unsupported AllReduce kernel` on every TP=2 `forward_backward` (sm_89, no NVLink). Compiles fine; fails at execution. |
| `XLA_PYTHON_CLIENT_PREALLOCATE=true` | One contiguous slab. Perf-neutral, but fragmented regions OOM the ~10 GiB `forward_backward` workspace on long runs. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 0.90 for 4B/9B, 0.95 for 2B. Not higher: XLA command buffers live **outside** the pool and need ~3 GiB, or the process aborts with exit 134. (0.87 also fits the 1-slot 9B config if command-buffer OOM ever appears.) |
| `JAX_COMPILATION_CACHE_DIR` | Each sequence-length bucket costs ~170 s to compile; the cache makes restarts cheap. Safe to `rm -rf`. |
| `NCCL_NET=Socket` | NCCL 2.28.9 segfaults on any 2-GPU collective on this box (removal = SIGSEGV, reproduced). Needed by trainer and sampler. |

JAX CUDA plugin: the venv runs `jax`/`jaxlib` 0.10.2 with `jax-cuda12-{plugin,pjrt}` **0.10.2**
(upgraded 2026-08-11 to match; perf-neutral, removes the boot warning about a 0.9.2 plugin).
If a jax upgrade ever reintroduces that warning, reinstall matching versions:
`uv pip install --python .venv/bin/python "jax-cuda12-plugin[with-cuda]==<jaxlib version>" "jax-cuda12-pjrt==<jaxlib version>"`.

### Hard constraints (each one reproduced, not theoretical)

- **TP=2 for 4B/9B; never FSDP** — the 248K-vocab `lm_head` logits replicate under FSDP,
  doubling the transient and OOMing.
- **Max trainable sequence length: 4096.** The 6144 bucket OOMs (16.8 GiB temp-arena
  allocation failure, reproduced 2026-08-11). Cap client sequences accordingly.
- **2 LoRA slots at rank 64 OOM at 4096 tokens** (16.71 GiB temp arena, reproduced) — use
  rank 32 for 2 slots, which fits. Prefer lowering `max_lora_rank` over losing sequence length.
- **Do not enable the XLA latency-hiding scheduler** — inflates startup memory ~8.5 → ~16 GiB/card.
- **Do not use `--enable-prefix-caching`** — 0% hit rate on this workload (prefill is shared
  across `num_samples` already), costs 15% KV.
- **Do not use MTP speculative decoding on the 4B/9B sampler** — −46% at production
  concurrency (compute-bound regime).

### LoRA adapter slots

`max_lora_adapters` = number of trainable adapters. With `--external-inference-url` (always,
in this runbook) all N slots serve user models; without it, slot 0 is reserved for the base
model (and `max_lora_adapters: 1` refuses to start). The startup log prints the usable-slot
count — grep `adapter slots`.

Each rank-64 slot costs 1.68 GiB/GPU (9B, TP=2: stacked LoRA weights + fp32 grad/mu/nu);
cost scales with rank. On 32 GB cards every rank-64 slot beyond the first costs one bucket of
max sequence length — hence the rank-32 2-slot config above.

Dead clients pin their slot until session expiry (`--session-timeout-sec`, default 300 s
+ up to 60 s sweep interval; the sweep runs between engine batches, so a long
`forward_backward` delays it). A client retrying `create_lora_training_client` in a tight
loop against a full server can livelock with expiry; retry with ≥30 s backoff.

### Attention backend

Qwen3.5's `head_dim=256` exceeds cuDNN's 128 cap on Ada (sm_89): training falls back to
Pallas/Triton automatically. Leave `SKYRL_CUDNN_MAX_HEAD_DIM` unset on this box; set it to
`256` on Hopper+ (sm_90) for the native cuDNN kernel.

---

## Troubleshooting

### `Maximum number of LoRA adapters (N) reached`

Usually the cap doing its job: `max_lora_adapters: 1` allows exactly one concurrent client.
A dead client's slot frees at session expiry. For faster turnaround, launch with
`--session-timeout-sec 60`. To clear immediately, restart the trainer — but first check for
stale queued requests that would replay on startup:

```bash
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d skyrl_tinker \
  -c "SELECT request_type, status, count(*) FROM futures WHERE status='PENDING' GROUP BY 1,2;"
```

If any `CREATE_MODEL` is `PENDING`, mark it failed (the `UPDATE futures` command in the reset
section) before relaunching.

### Trainer dies at startup, API exits with it

The engine subprocess and API live or die together (`monitor_engine` SIGTERMs the API when
the engine exits). Distinguish:

- **HTTP 400 on a request, server stays up** — recoverable in-Python OOM. Restart at leisure.
- **Port 8001 gone** — engine crashed (XLA abort/exit 134, or a bad `--backend-config`).
  A typo'd or stale backend-config key is a hard failure: the config is `extra="forbid"`.
  Check the tail of the trainer log.

### vLLM won't start

Usually a previous server (or another user) still holds the GPU:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

If the GPU is genuinely free, lower `--gpu-memory-utilization` in steps of 0.04.
Don't grep the log for readiness — poll `/v1/models`.

### Sampling stalls or vLLM memory creeps

After each successful sample the forwarder sweeps that model's adapters and unloads any that
are quiet (no requests in flight, not the newest). vLLM should hold roughly one adapter per
active model. If adapters pile up, check:

1. `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` was set on the vLLM launch — without it the unload
   endpoint doesn't exist and every unload fails.
2. The server log for `Failed to unload stale LoRA adapter`.

An unload is recoverable: the adapter directory stays on disk, and the filesystem resolver
reloads it by name if that checkpoint is sampled again. The **published adapter directories**
under `/dev/shm/$USER/..._lora_models` are what accumulate on disk (the server never deletes
them) — `rm -rf` the directory when vLLM is down.

### Samples fail with a 400 from `/v1/completions`

The error detail now includes vLLM's reason. Known triggers: `topk_prompt_logprobs` above
`--max-logprobs`, `temperature=0` with `num_samples>1`, prompt+max_tokens over `--max-model-len`.
Note each forwarded sample also has a hard 300 s timeout (10 s connect).

### vLLM engine dies on prompt-logprob requests

`prompt_logprobs` materializes full-vocab fp32 logits for every prompt position in the prefill
chunk: chunk 2048 × 248,320 vocab × 4 B = 1.89 GiB in one allocation — reproduced killing the
engine at `--max-num-batched-tokens 2048` regardless of memory utilization. The launch's
`1024` is the fix (verified surviving 24 concurrent probes); if it ever recurs, drop to 512
(costs ~3% throughput).

### `Number of devices 1 must be >= the product of mesh_shape`

JAX found only CPU devices — the venv lost its CUDA backend. Fix:

```bash
uv run --active --no-sync python -c "import jax; print(jax.devices())"
# Must show [CudaDevice(id=0), CudaDevice(id=1)]

uv pip install --python .venv/bin/python \
  "jax[cuda12]==$(uv run --active --no-sync python -c 'import jax; print(jax.__version__)')"
```

### Diagnosing a trainer OOM

Stop the trainer, add to the launch env, and reproduce:

```bash
rm -rf $STATE_DIR/skyrl_xla_dump && mkdir -p $STATE_DIR/skyrl_xla_dump

TF_CPP_MIN_LOG_LEVEL=0 \
TF_CPP_VMODULE=bfc_allocator=2 \
JAX_TRACEBACK_FILTERING=off \
XLA_FLAGS="--xla_dump_to=$STATE_DIR/skyrl_xla_dump --xla_dump_hlo_as_long_text --xla_dump_hlo_as_text" \
```

After the OOM:

```bash
grep -nE "ran out of memory|requested by op|Sum Total" logs/qwen35_${TAG}_tinker.log | tail -40
head -15 "$(ls -S $STATE_DIR/skyrl_xla_dump/*-memory-usage-report.txt | head -1)"
```

### Do not bump repo dependencies for a newer vLLM

The sampler already runs 0.25.1 out-of-tree via `uv run --isolated`. The repo pins
`vllm==0.23.0` exactly, and `uv lock --upgrade` would move torch off 2.11, breaking four
hand-built wheels pinned to the torch-2.11 ABI (`flash-attn`, `causal-conv1d`, `mamba-ssm`,
`transformer-engine-torch`).

---

## 2B / 4B variants

Same commands as the 9B quick start with `TAG`, `MODEL`, `MEM_FRAC`, and `BACKEND_CONFIG`
swapped per the table above. The 2B sampler additionally simplifies to a single card
(boot-verified on 0.25.1, 1.36M KV tokens):

```bash
# 2B sampler: CUDA_VISIBLE_DEVICES=3, --tensor-parallel-size 1, --max-model-len 4096,
# drop NCCL_NET (single GPU) and --max-num-seqs/--max-num-batched-tokens overrides.
```

The old 2B path (`uv run --no-sync --extra fsdp vllm serve`, using the venv's vLLM) still
works but depends on the venv staying at its currently-installed vLLM — one `uv sync` breaks
it. Prefer the isolated launch.

---

## Remote environments

### H100 box (3× H100 80 GB)

GPUs 0,1 = trainer (TP=2); GPU 2 = vLLM (TP=1). Up to 4 concurrent clients.
Differences from local (not re-benchmarked; last verified on that box):

- `SKYRL_CUDNN_MAX_HEAD_DIM=256` (sm_90 native cuDNN attention)
- `TMPDIR=/root/tmp`, `UV_PROJECT_ENVIRONMENT=.venv-jax`, `VLLM_USE_DEEP_GEMM=0`
- Trainer: `max_lora_adapters: 4`, `loss_chunk_size: 64` (sized for 4 concurrent
  `forward_backward`; the local chunking measurements don't transfer directly — re-measure
  before changing it there)
- vLLM: `--max-loras 9 --max-num-seqs 512 --max-num-batched-tokens 16384 --max-model-len 4096
  --limit-mm-per-prompt '{"image":0,"video":0}'` (80 GB cards tolerate the larger prefill
  chunk; the sm_89 logprob-transient math above does not apply at these margins)

### Modal B200 (one 192 GB GPU, single container)

`tools/modal_tinker_server.py` deploys trainer + sampler + `X-API-Key` auth proxy in one
container; 8 trainable LoRA slots at rank 64.

```bash
modal run    tools/modal_tinker_server.py::download_model         # once
modal run    tools/modal_tinker_server.py::smoke                  # before first deploy
modal deploy tools/modal_tinker_server.py
```

Client config:

```bash
export TINKER_API_KEY=tml-...
export TINKER_BASE_URL=https://<workspace>--skyrl-tinker-server.modal.run
```

Memory split (both processes share the GPU):

| | Fraction | Purpose |
|---|---|---|
| Trainer | 0.45 | weights + 8 LoRAs + activations |
| Sampler | 0.40 | weights + LoRA + KV cache |
| Free | 0.15 | XLA command buffers (exhaustion → exit 134) |

Operations:

```bash
modal app list                               # is it up?
curl -s $TINKER_BASE_URL/_status             # boot stage (unauthenticated by design)
modal app logs skyrl-tinker
modal app stop --yes skyrl-tinker            # --yes required (no tty → abort)
```

- **Bills continuously** (`min_containers=1`) until explicitly stopped.
- **Known failure:** the vLLM sampler can wedge silently; clients then die after the SDK's
  7200 s no-progress timeout. Monitor step progress, not process liveness.
- Checkpoints off the Volume: `tools/export_tinker_checkpoints.py` (read its docstring first).
  Training checkpoints are Orbax **directories** named `<id>.tar.gz`, not archives.
  If the checkpoint DB and Volume drift, `modal run tools/modal_tinker_server.py::seed_checkpoint_index`
  rebuilds the index. To import an external PEFT adapter, see `tools/import_peft_adapter.py`.

---

## Performance notes

Local 9B, TP=2, measured 2026-08-11 unless stated.

- **Trainer**: ~1,675 trained tok/s per `forward_backward` request at 4096 tokens with the
  quick-start config (was ~1,546 before the `loss_chunk_size`/`linear_attention_chunk_size`
  retune — +8%). Per-execution efficiency was already near this card's practical roofline;
  the win came from removing overhead, not kernels.
- **Sampler**: ~2,370 gen tok/s at 128 concurrent seqs, ~2,730 at 256; KV cache 845k tokens.
  The sampler is far from saturated at production concurrency — raising `num_samples` or
  prompt concurrency is close to free.
- **Biggest end-to-end lever is client-side**: async pipelining (train batch *i* while
  sampling batch *i+1*) recovers the ~35% duty-cycle loss of a synchronous RL loop
  (measured 2026-08-10). Server-side coalescing would not help: compute is per-sequence.
- Sampler weight publish (`save_weights_and_get_sampling_client`): 2–4 s.
- Trainer co-residency with the sampler (historical): MEM_FRAC 0.90 → 0.65 costs 4.1% on
  `forward_backward`; co-residency itself adds nothing beyond the smaller pool.

Rejected (measured, do not re-add):

| Setting | Result |
|---|---|
| `loss_chunk_size: 64` | −7% tok/s vs 128, no memory benefit at these shapes |
| `linear_attention_chunk_size: 32` | −5% tok/s **and** +0.6 GiB peak |
| `--max-num-batched-tokens 2048` | engine death on prompt-logprob requests (both 0.85 and 0.90 util) |
| `--gpu-memory-utilization 0.85` | −10.6% KV vs 0.90 for zero measured safety benefit on 0.25.1 |
| `--max-num-seqs 128` | −13% throughput at 256-seq offered load |
| MTP spec-decode (4B/9B) | −46% at 128 seqs (compute-bound) |
| `--enable-prefix-caching` | 0% hit rate, −15% KV |
| `sample_max_num_sequences` | no effect with external inference |
| `forwarding_inference_max_connections` | no effect with external inference (only read by the megatron/fsdp forwarding client) |
