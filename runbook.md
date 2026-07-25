# Runbook: Qwen3.5 Tinker (JAX) + vLLM on the local 4× RTX-5000 Ada box

Two long-lived servers for Tinker RL training. Hardware: 4× RTX-5000 Ada, 32 GB
each, sm_89, driver 580.126.09 / CUDA 13.0, 576 GB/s per card.

| Service | GPUs | Port | Process |
|---------|------|------|---------|
| Tinker training server (JAX backend) | 0, 1 | 8001 | `skyrl.tinker.api` |
| vLLM sampling server | 2 or 3 | 8000 | `vllm serve` |

Clients point the Tinker SDK at `http://127.0.0.1:8001`. Sampling is forwarded
to vLLM internally; clients never call port 8000.

§11 covers two other environments: an H100 box, and an unrelated 27B server that
also wants a GPU on this box.

**Neither server has authentication.** Anyone who can reach port 8001 can create
models, run training steps, and write files under `--checkpoints-base`. Both
launches below pass `--host 127.0.0.1`; keep it. For remote access use
`ssh -N -L 8001:127.0.0.1:8001 <host>` rather than rebinding.

---

## 1. Variables

Each block below re-declares these so it can be pasted standalone.

```bash
REPO=/scr1/michael/SkyRL                 # set to your SkyRL checkout
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
PG_BIN=$REPO/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin
MODEL=/scr1/public_models/huggingface/Qwen/Qwen3.5-4B-Base
TAG=4b                                   # 2b | 4b | 9b — names logs and state dirs
```

- **`STATE_DIR`** holds durable state: pgdata, checkpoints, the JAX compilation
  cache, XLA dumps. It must resolve to the same path on every launch, or the
  trainer will point at a different database. `/storage_slow` is root-owned — use
  the per-user subdirectory. If pgdata was created under `/tmp` and
  `/storage_slow/$USER` appears later, re-run §2 setup at the new location.
- **Model paths:** `/scr1/public_models/huggingface/Qwen/Qwen3.5-{2B,4B,9B}-Base`.
- **Sampler LoRA adapters go on tmpfs** (`/dev/shm/$USER/...`), not `STATE_DIR`.
  The JAX backend writes each adapter as a plain directory that vLLM loads in
  place, avoiding a tar round-trip through slow disk. vLLM never unloads them, so
  the directory grows until vLLM restarts — `rm -rf` it when you bounce vLLM
  (§7).
- **Trainer virtualenv:** the trainer needs the `tinker` and `jax` extras. If
  your `.venv` lacks them (`ModuleNotFoundError: sqlalchemy`), either point
  `UV_PROJECT_ENVIRONMENT` at a venv that has them or use `--isolated`.

---

## 2. PostgreSQL

Tinker runs on PostgreSQL, not SQLite: concurrent `asample` calls collide with
SQLite's writer queue during `save_weights` and leave `PENDING` futures that
block the SDK for its full 7200 s timeout (`database is locked`).

Postgres comes from the `pgserver` PyPI package, **binaries only**. Do not run
any Python that imports `pgserver` against this pgdata while Postgres is up:
`pgserver` installs an `atexit` handler that SIGTERMs the postmaster on
interpreter exit, even with `cleanup_mode=None`.

```text
PGDATA:  $STATE_DIR/skyrl_tinker_pg_data
DBNAME:  skyrl_tinker
PG_URL:  postgresql://postgres@/skyrl_tinker?host=$STATE_DIR/skyrl_tinker_pg_data
```

One-time, only if pgdata does not exist:

```bash
cd $REPO && uv pip install pgserver
mkdir -p $STATE_DIR/skyrl_tinker_pg_data
$PG_BIN/initdb -D $STATE_DIR/skyrl_tinker_pg_data -U postgres --auth=trust
```

Start and ensure the database exists. Idempotent — safe to run before every
launch. Tinker runs `SQLModel.metadata.create_all` at startup, so there is no
migration step.

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

Postgres is shared across all model variants and stays up across server
restarts. Do not stop it in §7.

---

## 3. Launch the sampler (vLLM)

### 4B — use this one

vLLM 0.25.1 in a throwaway env, text-only. This is 27% faster than the 0.20.2
launch below; §9 has the measurements and §10.6 explains why the version differs.
The sampler is a plain HTTP server that the Tinker engine reaches over
`--external-inference-url`, so its vLLM version is independent of the repo's
`vllm==0.20.2` pin.

```bash
REPO=/scr1/michael/SkyRL
MODEL=/scr1/public_models/huggingface/Qwen/Qwen3.5-4B-Base
cd $REPO && mkdir -p logs

CUDA_VISIBLE_DEVICES=3 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
setsid uv run --isolated \
  --with "vllm==0.25.1" --index https://wheels.vllm.ai/0.25.1/cu130 \
  vllm serve $MODEL \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 \
  --enable-lora --max-lora-rank 64 --max-loras 2 \
  --max-model-len 4096 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-num-seqs 128 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  > logs/qwen35_4b_vllm.log 2>&1 < /dev/null &

until curl -fsS -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done
echo vllm_ok
```

Confirm text-only mode engaged — the whole speedup is this line:

```text
All limits of multimodal modalities supported by the model are set to 0, running in text-only mode.
```

Two flags to know about:

- **`--gpu-memory-utilization 0.90`** — every GB not given to vLLM becomes KV
  cache, which is the throughput constraint here (§9). This exact value is
  untested: the §9 measurements were taken at 0.82 and 0.74 because the card was
  shared at the time, so the KV figures there are conservative. If the server
  fails to start, something is still resident on the GPU — see §10.7.
- **`--max-num-seqs 128`** — the measured throughput peak (§9). Raising it gains
  nothing and costs startup memory, since vLLM warms up with one dummy request
  per slot.

### 2B and 9B

These still use the repo's pinned vLLM 0.20.2. The 4B launch above has not been
tried with them.

| Model | Extra flags |
|-------|-------------|
| 2B | `--max-loras 2 --max-model-len 4096` |
| 9B | `--max-loras 4 --max-model-len 4096 --limit-mm-per-prompt '{"image":0,"video":0}'` |

```bash
REPO=/scr1/michael/SkyRL
TAG=9b
MODEL=/scr1/public_models/huggingface/Qwen/Qwen3.5-9B-Base
VLLM_ARGS='--max-loras 4 --max-model-len 4096 --limit-mm-per-prompt {"image":0,"video":0}'
cd $REPO && mkdir -p logs

CUDA_VISIBLE_DEVICES=3 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
setsid uv run --no-sync --extra fsdp vllm serve $MODEL \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 \
  --enable-lora --max-lora-rank 64 --dtype bfloat16 \
  $VLLM_ARGS \
  > logs/qwen35_${TAG}_vllm.log 2>&1 < /dev/null &

until curl -fsS -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done
echo vllm_ok
```

`--max-loras` only needs to be `>=` the trainer's `max_lora_adapters`.

---

## 4. Launch the trainer (Tinker, JAX backend)

`BACKEND_CONFIG` per model — §8 explains each value:

| Model | `MEM_FRAC` | `BACKEND_CONFIG` |
|-------|-----------|------------------|
| 2B | 0.95 | `{"max_lora_adapters":2,"max_lora_rank":64,"tensor_parallel_size":1,"fully_sharded_data_parallel_size":2,"train_micro_batch_size":1,"gradient_checkpointing":true}` |
| 4B | 0.90 | `{"max_lora_adapters":2,"max_lora_rank":64,"tensor_parallel_size":2,"fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,"gradient_checkpointing":true,"loss_chunk_size":128,"train_pad_seq_len_to":4096,"train_bucket_seq_len_per_micro_batch":true}` |
| 9B | 0.95 | `{"max_lora_adapters":4,"max_lora_rank":64,"tensor_parallel_size":2,"fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,"gradient_checkpointing":true,"loss_chunk_size":128}` |

```bash
REPO=/scr1/michael/SkyRL
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
TAG=4b
MODEL=/scr1/public_models/huggingface/Qwen/Qwen3.5-4B-Base
MEM_FRAC=0.90
BACKEND_CONFIG='{"max_lora_adapters":2,"max_lora_rank":64,"tensor_parallel_size":2,"fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,"gradient_checkpointing":true,"loss_chunk_size":128,"train_pad_seq_len_to":4096,"train_bucket_seq_len_per_micro_batch":true}'

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

Environment variables, all four required:

- **`XLA_PYTHON_CLIENT_PREALLOCATE=true`** — reserves `MEM_FRAC` as one
  contiguous slab at startup. With `false` the pool grows as separate regions
  that cannot coalesce, and the ~10 GiB `forward_backward` workspace eventually
  fails to find contiguous space. Safe because GPUs 0,1 are the trainer's alone.
- **`XLA_PYTHON_CLIENT_MEM_FRACTION`** — see the table; §8 explains why the 4B
  uses 0.90.
- **`JAX_COMPILATION_CACHE_DIR`** — each sequence-length bucket costs ~165-175 s
  of XLA compilation, and trainer restarts are routine (§7), so without the cache
  every restart re-pays the full ladder. The cache key includes the JAX version,
  GPU type and XLA flags, so changing any of those simply misses and recompiles.
  Safe to `rm -rf` at any time. Do not put it on `/dev/shm` — it must survive
  reboots to be useful.
- **`NCCL_NET=Socket`** — works around NCCL 2.28.9's net-transport auto-probe
  (`ncclNetPluginInit`) segfaulting on any 2-GPU collective on this box.
  Reproduces in a bare torch allreduce, so it is not SkyRL's stack.
  `NCCL_IB_DISABLE=1` works equally well; `NCCL_NET_PLUGIN=none` does not.
  Also needed if vLLM is ever run with TP=2 here.

---

## 5. Verify

```bash
REPO=/scr1/michael/SkyRL
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
PG_BIN=$REPO/.venv/lib/python3.12/site-packages/pgserver/pginstall/bin

pgrep -af '[s]kyrl\.tinker|[v]llm serve /scr1/public_models'
$PG_BIN/pg_ctl -D $STATE_DIR/skyrl_tinker_pg_data status | head -1
curl -fsS -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo vllm_ok
curl -fsS -m 5 http://127.0.0.1:8001/api/v1/get_server_capabilities >/dev/null && echo tinker_ok
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Must print 127.0.0.1, never 0.0.0.0. Both servers default to 0.0.0.0, and the
# health checks above pass either way, so a wrong bind is silent.
ss -ltnp | grep -E ':(8000|8001)\b'
```

Sampler KV cache size, worth checking after any flag change — it is the
throughput constraint (§9):

```bash
grep -oE "GPU KV cache size: [0-9,]+ tokens" logs/qwen35_4b_vllm.log | tail -1
```

---

## 6. Reset state

Stop the trainer first (§7) so nothing holds database connections.

```bash
STATE_DIR=$([ -w /storage_slow/$USER ] && echo /storage_slow/$USER || echo /tmp)
TAG=4b

rm -rf /dev/shm/$USER/qwen35_${TAG}_lora_models
rm -rf $STATE_DIR/skyrl_qwen35_${TAG}_checkpoints
```

All model variants share the one `skyrl_tinker` database, so dropping it clears
every variant's sessions, futures and checkpoint records. On-disk artifacts are
per-`TAG`.

```bash
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "DROP DATABASE IF EXISTS skyrl_tinker;"
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d postgres \
  -c "CREATE DATABASE skyrl_tinker;"
```

To unblock a client stuck on a `PENDING` future without dropping the database,
mark the futures failed:

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

## 7. Stop and restart

Escape the dots and use the `[x]` bracket trick. An unescaped `skyrl.tinker`
also matches the postgres command line (`skyrl_tinker_pg_data`) and would kill
the database; an unbracketed pattern matches the killing shell itself.

```bash
pkill -f '[s]kyrl\.tinker' || true
pkill -f '[v]llm serve /scr1/public_models' || true

for i in $(seq 1 30); do
  pgrep -f '[s]kyrl\.tinker|[v]llm serve /scr1/public_models' >/dev/null || break
  sleep 1
done
```

Do not stop Postgres.

When to restart what:

- **Trainer** — after any JAX OOM. The allocator high-water does not drop on its
  own, and a JAX OOM returns HTTP 400 without killing the process, so every
  subsequent training op fails until a bounce. A restart also clears leaked
  adapter slots (§10.1).
- **vLLM** — periodically on long runs, to drop accumulated sampler adapters
  (§10.2), and whenever sampling stalls. `rm -rf` the `/dev/shm/$USER/...`
  adapter directory at the same time.

---

## 8. Settings reference

### 2B

FSDP=2 / TP=1 fits. The values in §3 and §4 need no adjustment.

### 4B

- **TP=2 is required; do not use FSDP.** GPUs 0 and 1 are PCIe-connected
  (`PIX`, not NVLink). Under FSDP=2/TP=1 the tied 248K-vocab `lm_head` logits are
  replicated per card instead of vocab-sharded, which doubles the
  `forward_backward` transient and OOMs.
- **`MEM_FRAC=0.90`, not 0.95.** CUDA-graph and XLA command-buffer
  instantiations are driver allocations *outside* the JAX BFC pool, living in
  whatever memory `MEM_FRAC` did not reserve. At 0.95 that headroom (~1.6 GiB) is
  too thin: over a long run accumulated command buffers exhaust it, a graph
  re-instantiation OOMs one TP rank, the TP=2 collective desyncs, and the process
  aborts with exit 134 (`Retry CUDA graph instantiation after OOM`, then a
  rendezvous termination).
- **Maximum trainable sequence length is 4096 tokens.** Buckets 1024-4096
  compile and run; 6144 needs 14.55 GiB and OOMs on both cards even on a fresh
  start. The next bucket above 4096 is 6144 (see `round_up_seq_len`), so cap
  client sequences at 4096 raw tokens.
- **Do not enable the XLA latency-hiding scheduler.**
  `--xla_gpu_enable_latency_hiding_scheduler` was tried to overlap the PCIe TP
  all-reduces; it extends buffer live ranges and inflated startup high-water to
  ~16 GiB/card before any step ran, versus ~8.5 GiB without.
- **`train_pad_seq_len_to: 4096`** pads every batch to 4096 so there is one
  train executable and one allocation pattern, making step memory invariant to
  the number of concurrent runs.
- **`train_bucket_seq_len_per_micro_batch: true`** overrides that pin per
  micro-batch and is worth 1.42x on `forward_backward` (§9.1). Enabled in the §4
  config. It costs GPU headroom; §9.1 lists the trade-offs.
- **`max_lora_adapters: 2`** gives one trainable adapter — slot 0 is reserved for
  the base model. `1` rejects client creation outright.
- **Concurrent runs need the shared `AdamState` optimizer** (fixed 2026-07-13 in
  `skyrl/backends/jax.py`). Previously each created model got its own
  `nnx.Optimizer` holding fp32 Adam moments over the entire
  `max_lora_adapters`-stacked LoRA tree, ~5-6 GiB/GPU per model at 5 slots ×
  rank 64. One model trained for hours, but a second client's model made the next
  step OOM. The fix allocates mu/nu once and has `optim_step` update only the
  stepping model's slot, so optimizer memory is constant in the number of models;
  4 concurrent rank-64 runs at 4096 pass. Full-state checkpoints written before
  that fix use the old `optimizer_state` layout and will not reload. Sampler
  weight checkpoints are unaffected.
- Do not enable offloading unless everything above still produces a real OOM.

### 9B

- `Qwen3.5-9B-Base` requires `--limit-mm-per-prompt '{"image":0,"video":0}'` on
  vLLM 0.20.2. Without it the vision encoder buffer pushes the process to ~31 GB
  during profiling and OOMs the card even at `--max-loras 2`.
- With multimodal disabled, `--max-loras 4` at rank 64 fits alongside the 19.1 GB
  of weights (~29.3 GB total). `--max-model-len 4096`, no `--enforce-eager`, no
  reduced `--gpu-memory-utilization` needed.

### Attention backend (all models)

Qwen3.5 uses `head_dim=256`, which exceeds cuDNN flash attention's head-dim cap
of 128 on Ampere and Ada. On this box `dot_product_attention` therefore falls off
the cuDNN path: causal prefill and training use Pallas/Triton, non-causal decode
uses XLA.

The cap is read from `SKYRL_CUDNN_MAX_HEAD_DIM` by
`skyrl/tx/layers/attention.py`, default 128. There is no auto-detection — set it
by GPU generation:

- **Ada or older (≤ sm_89), including this box:** leave it unset. Setting 256 is
  rejected by JAX with `NotImplementedError: head dim must be <= 128`.
- **Hopper (sm_90) or newer:** set `SKYRL_CUDNN_MAX_HEAD_DIM=256` on the trainer
  launch to get the native cuDNN kernel.

### Multimodal weights

All three Qwen3.5 Base checkpoints (2B, 4B, 9B) carry a `vision_config` and a
vision tower — for the 4B, 297 of 738 tensors. All three also ship 15 unused
`mtp.*` tensors and declare `mtp_num_hidden_layers: 1`. For text-only workloads
the vision tower is dead weight; see §9.2 for what disabling it is worth and
§10.6 for the version constraint.

---

## 9. Measured performance

All figures 2026-07-24 on this box, 4B model.

### 9.1 Trainer: per-micro-batch sequence-length bucketing (1.42x)

`train_bucket_seq_len_per_micro_batch: true` pads each micro-batch to
`round_up_seq_len` of its own longest sequence instead of the whole batch's
maximum.

The problem it solves: with `train_pad_seq_len_to: 4096`, `forward_backward`
costs a flat ~2.0 s per sequence regardless of that sequence's real length — a
500-token sequence costs the same as a 4095-token one. Measured across production
requests of 2, 17, 32, 47 and 158 sequences: 1.98-2.04 s/seq every time. Those
448 sequences averaged 2454 tokens (median 2238), so **40% of all
`forward_backward` compute was padding.**

Lowering `train_pad_seq_len_to` does not fix this. The pad length is the maximum
over the whole request batch, and 10 of 11 production requests contained at least
one ~4095-token sequence, so the batch pads to 4096 regardless. The fix has to be
per-micro-batch.

Measured, same server, `train_micro_batch_size=1`, N=32, warm:

| Workload | s/seq |
|----------|-------|
| every sequence 4095 tokens (equals the old pinned cost) | 2.08 |
| real length distribution (mean 2358) | **1.47** |

**1.42x.** Replaying the real length histogram predicted 1.49x; short passes are
slightly less efficient per token, so the histogram figure is an upper bound.

Gradients are unchanged by construction: the loss divides by `loss_mask.sum()`,
which counts real tokens only, and `pad_batch` zero-fills, so padding cancels.
Verified — logprobs for 512/1000/2048/3000/4095-token sequences were
bit-identical between bucketed and pinned at 4 of the 5 lengths; the 512 case
differed 1.25e-4 relative, far below bf16's ~4e-3 epsilon.

Costs:

- **GPU headroom.** 7 executables instead of 1 raised GPU 0,1 usage from
  29344 MiB to 30786 MiB of 32760, leaving ~1.97 GiB. Those are command buffers,
  which live outside the BFC pool — the same headroom whose exhaustion causes the
  exit-134 crash described in §8. There is room at `MEM_FRAC=0.90`, but watch it
  on long runs. If it crashes with `Retry CUDA graph instantiation after OOM`,
  lower `MEM_FRAC` to 0.87 rather than disabling bucketing.
- **First-run compilation.** One compile per bucket, ~165-175 s each. The
  production distribution needs 7 buckets (512, 768, 1024, 1536, 2048, 3072,
  4096), about 15 minutes once. `JAX_COMPILATION_CACHE_DIR` makes restarts free,
  but a client whose sequences reach an eighth bucket pays another ~170 s
  mid-run. All 7 are cached on this box.
- **A coarser bucket ladder trades speed for headroom.** From the histogram:
  7 buckets = 1.49x ideal, a fixed {2048, 3072, 4096} ladder = 1.39x with 3
  executables, powers of two only = 1.34x with 4. Not currently exposed as a
  knob.

To roll back, set the flag to `false`; `train_pad_seq_len_to` then applies as
before.

### 9.2 Sampler: text-only mode (+27%)

Workload measured from 27,975 production sampling requests: 341-token prompts,
`num_samples=16`, `max_tokens` averaging 3753.

| Configuration | KV memory | KV tokens | Peak output |
|---------------|-----------|-----------|-------------|
| 0.20.2, multimodal enabled | 6.83 GiB | 157,509 | 1714 tok/s at 64 seqs |
| 0.25.1, text-only | **14.15 GiB** | **326,562** | **2170 tok/s at 128 seqs** |

**+107% KV cache, +27% peak throughput, +31% at a matched 128 sequences.**

The handicap ran against the winning side: the 0.20.2 baseline had the card to
itself at the default utilization, while the 0.25.1 figures were taken at 0.82
with ~5 GB resident. The advantage is not an artifact of the memory budget — a
matched-budget run at 0.74 (288,954 KV tokens) still returned 2168 tok/s at 128
sequences, against the baseline's 1652 at the same concurrency. On a free card at
0.90 both sides gain KV cache and text-only gains more of it.

With multimodal enabled, vLLM reserves a 114,688-token encoder cache and profiles
with a video item; the resulting activation peak crowds out KV cache. In the
production run this left **13 GB of the card unused while KV cache sat at 99-100%
with 442 requests queued.**

Throughput by concurrency, 0.25.1 text-only, 1024 `max_tokens`: 32 seqs 1209 ·
64 seqs 1718 · **128 seqs 2170** · 192 seqs 1996 · 256 seqs 2109. The old
configuration peaked at 64 and degraded past it because KV ran out.

The Tinker LoRA path is unaffected. 0.25.1 renamed some response fields, so this
was checked end-to-end (real trainer, `optim_step`, then
`save_weights_and_get_sampling_client`, then n=16 sampling):
`POST /v1/load_lora_adapter` works, sync took 5.2 s; `return_token_ids` →
`choice["token_ids"]` and `logprobs.token_logprobs` are unchanged, which is what
`skyrl/tinker/extra/external_inference.py` reads; 128 LoRA sequences ran at
1908 tok/s with logprobs aligned for all 128. LoRA sampling costs ~12% versus
base-model sampling.

**How much KV cache is enough.** At 1024 `max_tokens`, 288,954 and 326,562 KV
tokens performed identically (2168 vs 2170 tok/s at 128 sequences) — past ~289k,
KV stops binding *at that length*. The win above comes from clearing the
starvation threshold: the old 157,509 tokens could not hold
128 × (341+1024) = 174,720. At production length this changes; measured with
288,954 KV tokens:

| Sequences | 1024 max_tokens | 3072 max_tokens |
|-----------|-----------------|-----------------|
| 64 | 1712 tok/s | 1590 tok/s |
| 128 | 2168 tok/s | 1664 tok/s, KV 99.4%, requests queueing |

Sizing rule: concurrent capacity is roughly
`KV_tokens / (prompt + max_tokens)`. At 3413 tokens per sequence the old
configuration holds 46 concurrent sequences and text-only holds 84, so the +27%
above is a **floor** for production-length work. That 1.8x capacity ratio is
arithmetic, not a measurement — the two configurations were never benchmarked
head-to-head at 3072. Worth doing on a free card if the exact figure matters.

### 9.3 Settings that were measured and rejected

- **MTP speculative decoding** — a large regression for this workload. A/B at the
  same `--gpu-memory-utilization 0.74`:

  | | KV tokens | 64 seqs | 128 seqs |
  |--|-----------|---------|----------|
  | no MTP | 288,954 | 1712 tok/s | **2168 tok/s** |
  | MTP, `num_speculative_tokens: 1` | 191,341 (−34%) | 1136 (−34%) | **1172 (−46%)** |

  The 4B ships the same 15 `mtp.*` tensors that give the 27B +56% (§11.2), but
  the regime is opposite. The 27B runs at ~95% of memory bandwidth, where
  spending compute to get ~2 tokens per weight read is nearly free. The 4B at RL
  concurrency is compute-bound — ~31% of the 576 GB/s ceiling at 32+ sequences —
  so the draft forward is added cost that acceptance does not repay. MTP also
  gives up 34% of KV capacity because GDN/mamba state is allocated per
  speculative token. Check where a model sits against memory bandwidth before
  assuming MTP helps it.
- **`--enable-prefix-caching`** — 86% of production sampling prompts repeat
  (27,983 requests, 3,932 distinct), which sounds promising, but with
  `num_samples=16` the 341-token prefill happens once and is shared by all 16
  samples. Prefill is therefore ~1% of tokens processed against ~33,600 decode
  tokens. Saving ~1% does not pay for the 15% of KV cache it costs on this hybrid
  architecture (§11.2), and KV is the binding constraint.
- **Raising `sample_max_num_sequences`** — has no effect on this configuration;
  see §10.4.

---

## 10. Troubleshooting

### 10.1 `Maximum number of LoRA adapters (N) reached`

Crashed or exited clients do not release trainer adapter slots. A dead client
leaves its adapter registered, so each crash burns a slot permanently. Session
expiry does not reclaim it — verified 2026-07-24: a session that had been expired
for over 5 minutes still held its slot.

Restart the trainer. With `max_lora_adapters: 2` there is exactly one trainable
slot, so a single crashed client blocks all new clients.

### 10.2 Sampling stalls, or vLLM memory creeps up

vLLM never unloads sampler adapters. A fresh ephemeral LoRA is loaded per
checkpoint and `/unload` is never called, so they accumulate (25 observed in one
run). Restart vLLM and `rm -rf` the `/dev/shm/$USER/...` adapter directory.

### 10.3 Diagnosing a trainer OOM

Environment variables must be set at process start, so stop the trainer, relaunch
with the additions below, and re-run the workload that failed. Keep the same
`BACKEND_CONFIG` as the failing run so it reproduces — only the variables and
dump directory are additions.

```bash
rm -rf $STATE_DIR/skyrl_xla_dump && mkdir -p $STATE_DIR/skyrl_xla_dump

# add to the §4 launch environment:
TF_CPP_MIN_LOG_LEVEL=0 \
TF_CPP_VMODULE=bfc_allocator=2 \
JAX_TRACEBACK_FILTERING=off \
XLA_FLAGS="--xla_dump_to=$STATE_DIR/skyrl_xla_dump --xla_dump_hlo_as_long_text --xla_dump_hlo_as_text" \
```

The allocator summary goes to the trainer log; the XLA buffer assignment goes to
the dump directory. After it OOMs:

```bash
# confirm the failing allocation size
grep -nE "ran out of memory|requested by op|Sum Total|Bin \(" \
  logs/qwen35_${TAG}_tinker.log | tail -40

# the largest module is the forward_backward graph
ls -S $STATE_DIR/skyrl_xla_dump/*-memory-usage-report.txt | head

# largest buffers with shapes. f32[6144,248320] is logits over the 248K vocab;
# f32[2,8,3072,3072] is attention scores.
head -15 "$(ls -S $STATE_DIR/skyrl_xla_dump/*-memory-usage-report.txt | head -1)"

# map a size back to an op and source line
BA="$(ls -S $STATE_DIR/skyrl_xla_dump/*-buffer-assignment.txt | head -1)"
grep -nE "size 2[0-9]{10}" "$BA" | head    # adjust to the failing size
```

### 10.4 `sample_max_num_sequences` has no effect with external inference

With `--external-inference-url` set, sample requests are stored as
`RequestType.EXTERNAL` (`skyrl/tinker/api.py:1080`) and dispatched immediately as
fire-and-forget asyncio tasks (`api.py:1096`). `sample_max_num_sequences` is only
applied in the engine's `find_batchable_sample`, which queries
`RequestType.SAMPLE`, and `engine.py:460` explicitly excludes `EXTERNAL` from
engine processing. All 984 sample requests of the 2026-07-24 run were `EXTERNAL`.

Sampling concurrency is therefore set entirely by how many `asample` calls the
client keeps in flight, bounded only by `forwarding_inference_max_connections`
(default `None`, unlimited). To throttle the sampler, use that setting, cap
`--max-num-seqs` on vLLM, or throttle client-side. Changing
`sample_max_num_sequences` will do nothing, so it is omitted from the §4 configs.

The H100 configuration in §11.1 previously credited this knob with fixing a vLLM
stall. That attribution cannot be correct for the external-inference path.

### 10.5 `--max-model-len 4096` versus what the client asks for

The 2026-07-24 run requested `max_tokens` averaging 3753 (maximum 3818) on
341-token prompts, which sums to ~4094 — flush against the 4096 ceiling, and the
longest requests exceed it.

Raising the sampler's limit alone does not help, because the trainer cannot
consume sequences longer than its 4096 bucket (§8). Either cap the client at
`4096 - len(prompt)`, or accept that completions past 4096 will be generated and
then discarded by training.

### 10.6 Do not bump the repo's dependencies to get newer vLLM

The sampler runs 0.25.1 out-of-tree, which is exactly why no repo change is
needed. Checked with `uv lock --upgrade --dry-run` (463 packages resolved):

- **`uv lock --upgrade` does not move vLLM.** `vllm==0.20.2` is an exact pin with
  a matching `wheels.vllm.ai/0.20.2/cu129` index, so a blanket upgrade changes
  ~250 other packages and leaves vLLM where it is.
- **It moves torch 2.11.0 → 2.13.0, which breaks three hand-built wheels** pinned
  to the 2.11 ABI at `pyproject.toml:285-287`: `flash-attn 2.8.3`
  (`v2.8.3-torch2.11-clean`), `causal-conv1d` (`v1.6.1.post4-torch2.11`) and
  `mamba-ssm` (`v2.3.1-torch2.11`). No torch-2.13 builds exist at those URLs, and
  `causal-conv1d`/`mamba-ssm` are the kernels Qwen3.5 linear attention needs.
- It also moves jax 0.9.2 → 0.11.0, the `tinker` SDK 0.16.1 → 0.23.4, and majors
  across `datasets` 4→5, `opencv` 4→5, `starlette` 0.52→1.3, `wrapt` 1→2,
  `numpy` 2.2→2.4. `transformers` does not move; it is capped `<=5.8.0`.

Separately, `--limit-mm-per-prompt '{"image":0,"video":0}'` **crashes vLLM
0.20.2** for the 4B: `AttributeError: 'NoneType' object has no attribute 'size'`
at `qwen3_next.py:495`, raised from `_dummy_run` during CUDA-graph capture,
because the multimodal embedding input is `None` and the 0.20.2 forward does not
handle that. 0.25.1 handles it and skips the vision tower rather than merely
shrinking its cache. The 9B does pass this flag on 0.20.2 successfully (§8), so
do not generalise either behaviour to the other model without testing it.

If you do want one vLLM version everywhere, that is a separate task: bump the pin
and the wheel index to cu130 while holding torch at 2.11.0, then run a
weight-sync test pass, because SkyRL imports unstable vLLM internals
(`distributed.weight_transfer.nccl_engine`,
`model_executor.model_loader.reload`, `config.WeightTransferConfig`, `renderers`,
`v1.metrics.ray_wrappers`). This is untested as of 2026-07-24.

### 10.7 vLLM will not start

All three of these mean the GPU is not as free as the launch assumes — usually a
previous server that did not exit. Check
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`, stop the
stray process (§7), and retry before changing any flag.

- `Free memory on device cuda:N (X/31.47 GiB) on startup is less than desired GPU
  memory utilization` — vLLM's own pre-flight check.
- `CUDA out of memory occurred when warming up sampler with N dummy requests`.
- An OOM inside `torch.cuda.graph` / `capture_end` — CUDA-graph capture needs
  memory outside the KV pool.

If the GPU really is yours and it still fails, lower
`--gpu-memory-utilization` in steps of 0.04. Lowering `--max-num-seqs` alone does
not help: it moves the failure from sampler warmup to graph capture.

Do not grep the log for readiness — benign warnings match naive error patterns
and fire before the port binds. Poll `/v1/models`.

---

## 11. Other environments

### 11.1 9B on the H100 box

Configuration that converged running the InjecAgent matrix (up to 4 concurrent
client runs, Qwen3.5-9B-Base attacker) on a 3× H100 80 GB modal box.

- GPUs 0,1 = JAX trainer (TP=2); GPU 2 = vLLM (TP=1).
- sm_90, so set `SKYRL_CUDNN_MAX_HEAD_DIM=256` on the trainer (§8).
- Environment quirks baked into the launch scripts: `TMPDIR=/root/tmp`,
  `UV_PROJECT_ENVIRONMENT=.venv-jax` for the trainer, `VLLM_USE_DEEP_GEMM=0`
  (vLLM 0.20.2 otherwise crashes in DeepGEMM warmup),
  `HF_HUB_ENABLE_HF_TRANSFER=1`.
- Launch via `/root/launch_tinker.sh` and `/root/launch_vllm.sh`, each of which
  kills its predecessor and re-execs. Add `JAX_COMPILATION_CACHE_DIR` there too;
  restarts are frequent on that box.
- Cap client concurrency at 4.

Trainer `BACKEND_CONFIG`:

```json
{"max_lora_adapters":5,"max_lora_rank":64,"tensor_parallel_size":2,
 "fully_sharded_data_parallel_size":1,"train_micro_batch_size":1,
 "gradient_checkpointing":true,"loss_chunk_size":64}
```

vLLM: `--max-loras 9 --max-num-seqs 512 --max-num-batched-tokens 16384
--max-model-len 4096 --limit-mm-per-prompt '{"image":0,"video":0}'`.

Why these values:

- **`train_micro_batch_size: 1`** — at 4, padded InjecAgent sequences made the
  backward pass need 93.82 GiB on one card and OOM. At 1 it peaks ~23 GiB.
- **`max_lora_adapters: 5`** (4 usable; slot 0 is the base model). At 9, two
  things went wrong: the JAX backend stores LoRA gradients stacked over all slots
  (`accumulated_grads = zeros_like(lora_params)`), so gradient and accumulation
  memory scale with slot count and pinned ~66 GiB/GPU, OOMing 4-concurrent; and
  `forward_backward` slowed to ~83-91 s versus ~17-25 s at 5.
- **`loss_chunk_size: 64`** — halves the logits chunk versus 128, fitting 4
  concurrent `forward_backward` under the ~76 GiB/GPU usable cap. Drop to 32 if
  it OOMs again.

### 11.2 Qwen3.6-27B-NVFP4 standalone server

Not part of the Tinker stack — a plain OpenAI-compatible endpoint on port 8002.
**It occupies one GPU, so it and the §3 sampler cannot share a card.** Text-only.

Model: `/scr1/public_models/huggingface/nvidia/Qwen3.6-27B-NVFP4` (21 GB,
`Qwen3_5ForConditionalGeneration`, 64 layers = 48 GDN linear-attention + 16 full
attention, `mtp_num_hidden_layers: 1`).

```bash
CUDA_VISIBLE_DEVICES=3 setsid uv run --isolated \
  --with "vllm==0.25.1" --index https://wheels.vllm.ai/0.25.1/cu130 \
  vllm serve /scr1/public_models/huggingface/nvidia/Qwen3.6-27B-NVFP4 \
  --served-model-name Qwen3.6-27B-NVFP4 \
  --host 127.0.0.1 --port 8002 \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-prefix-caching --async-scheduling \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","model":"/scr1/public_models/huggingface/nvidia/Qwen3.6-27B-NVFP4","num_speculative_tokens":1}' \
  > logs/qwen36_27b_nvfp4_vllm.log 2>&1 < /dev/null &

until curl -fsS -m 5 http://127.0.0.1:8002/v1/models >/dev/null; do sleep 5; done
echo vllm_ok
```

Cold start ~6 min (two `torch.compile` passes: main model and MTP draft). Warm
restarts hit the AOT cache and take ~2 min. Changing any flag that alters the
graph hash, notably `--speculative-config`, forces a full recompile.

**Measured** (256-token completions, thinking off). Weights 19.96 GiB, KV cache
94,208 tokens, or 110,832 without prefix caching.

| Concurrency | No MTP | With MTP |
|-------------|--------|----------|
| 1 | 26.8 tok/s | **43.7** |
| 8 | 188.7 tok/s | **298.6** |
| 16 | — | **497.1** |
| 32 | — | **490.7** |

Prefill is flat at ~2,520 tok/s (TTFT 1.65 s at 4k, 3.21 s at 8k, 5.69 s at 14k).

This model is at the hardware limit. Non-speculative decode of 26.8 tok/s ×
~20.5 GB of weights is 549 GB/s against the card's 576 GB/s, about 95% of memory
bandwidth. MTP is the only way past that wall — it does not read weights faster,
it gets ~2 tokens per read. At concurrency 16 the system drops to ~67% of
bandwidth, i.e. compute-bound on the Marlin FP4 dequant path, which is the cost
of running NVFP4 on Ada rather than Blackwell. Further gains need different
hardware or a different checkpoint, not configuration.

**Why 0.25.1** — vLLM 0.20.2 cannot serve this checkpoint at all. Three
independent blockers, each fixed in a different release:

1. `W4A16_NVFP4` is never dispatched before 0.22.0. In 0.20.2,
   `ModelOptMixedPrecisionConfig.get_quant_method` handles only `FP8` and
   `NVFP4`. This checkpoint is `MIXED_PRECISION` with 208 FP8 and 193
   `W4A16_NVFP4` layers (all 64 MLPs plus `lm_head`); the latter fall through to
   `UnquantizedLinearMethod` and load as bf16, turning 9.5 GB of packed MLP
   weights into ~34 GB. The signature is a 170 MiB failing allocation
   (`17408 × 5120 × 2`, an MLP `down_proj` at bf16) with the traceback ending in
   `linear.py`.
2. A quantized `lm_head` needs 0.23.0. It is NVFP4 on disk, but `ParallelLMHead`
   is a `VocabParallelEmbedding` rather than a `LinearBase`, so 0.22.0 fails with
   `There is no module or parameter named 'lm_head.input_scale'`.
3. Use the `/cu130` index, not `/cu129`. With cu129, uv installs
   `nvidia-cuda-nvrtc-cu12` while torch resolves to `2.11.0+cu130`; `import vllm`
   succeeds and the failure only appears during Marlin scale preparation as
   `nvrtc: error: failed to open libnvrtc-builtins.so.13.0`.

**Rejected settings**, all measured:

- `num_speculative_tokens: 2` — legal, and vLLM reruns the MTP layer
  autoregressively, but it saturates earlier: concurrency 1 rises to 53.2 tok/s
  (+21%) while concurrency 16 falls to 350.0 (−29%), and KV drops to 90,931. Use
  1 for aggregate throughput; consider 2 only for a single interactive stream.
- `--max-num-batched-tokens 8192` — OOMs during memory profiling, 594 MiB short,
  because profiling activations scale with this value rather than with
  `--max-model-len`. Pointless anyway: prefill is flat at ~2,520 tok/s, so 2048
  already saturates the GPU.
- `--max-num-seqs` above 32 — throughput plateaus from concurrency 16 onward.

**Flag notes:**

- `--enable-prefix-caching` costs 15% of KV cache (110,832 → 94,208 tokens)
  because it forces `Mamba cache mode = 'align'` for this hybrid architecture,
  and is throughput-neutral on unique prompts. Its benefit on shared prefixes was
  never measured here. If prompts do not share a system prompt, few-shot block or
  chat history, drop it and reclaim the tokens.
- `--async-scheduling` measured neutral.
- `--gpu-memory-utilization 0.92` — 19.96 GiB of weights on a 32 GB card leaves
  ~11 GiB. Do not raise to 0.95; the 594 MiB profiling OOM above lands in exactly
  that reserve.

**API notes** (these differ from older vLLM):

- Thinking is on by default. Disable per request with
  `"chat_template_kwargs": {"enable_thinking": false}`, and budget `max_tokens`
  accordingly — an 80-token cap is consumed entirely by the reasoning trace.
- The trace returns as `message.reasoning`, not `message.reasoning_content`
  (renamed in 0.25.x). The final answer stays in `message.content`.
- Parser names come from the 0.25.1 registries: reasoning `qwen3`, tool
  `qwen3_xml` (an exact alias of `qwen3_coder` — same class, no behavioural
  difference). The chat template emits
  `<tool_call><function=…><parameter=…>`.
- MTP disables `min_p` and `logit_bias`; vLLM warns at startup. If a client needs
  either, drop `--speculative-config` and accept 26.8 tok/s single-stream.
