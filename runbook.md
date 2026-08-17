# Tinker Server Runbook

Launching **attacker Tinker servers** (Qwen3.5 LoRA training, JAX backend + external vLLM
sampling) across four environments:

| Environment | Hardware | Topology | Slots | Section |
|---|---|---|---|---|
| **Local** | 4× RTX 5000 Ada (32 GB, sm_89, PCIe) | trainer TP=2 (GPU 0,1) + sampler TP=2 (GPU 2,3) | 1 (rank 64) | [Local](#local-4-rtx-5000-ada) |
| **RunPod** | 2× RTX PRO 6000 Blackwell (96 GB, sm_120, PCIe) | trainer TP=2 + sampler TP=2, co-located on both GPUs | 1 (rank 64) | [RunPod](#runpod-2-rtx-pro-6000-blackwell) |
| **H100 box** | 3× H100 (80 GB, sm_90) | trainer TP=2 (GPU 0,1) + sampler TP=1 (GPU 2) | 4 | [H100](#h100-box-3-h100-80-gb) |
| **Modal** | 1× B200 (192 GB) | trainer + sampler co-located, one container | 8 | [Modal](#modal-b200-one-192-gb-gpu-single-container) |

Every flag and config below is load-bearing: verified against the code and re-measured
(local 9B stack 2026-08-11; RunPod stack 2026-08-16). Optimizations with no measured benefit
were removed. Facts that hold only on specific hardware are tagged with the environment.

In every environment the client connects to the **trainer** (`skyrl.tinker.api`, port 8001);
sampling is forwarded internally to vLLM (port 8000), which clients never touch.

---

## Invariants (every environment)

**Security.** The Tinker API has **no authentication** and can write files anywhere under
`--checkpoints-base`. Both servers must bind `127.0.0.1`:

> **`--host 127.0.0.1` is load-bearing — never omit it.** `skyrl.tinker.api` defaults to
> `0.0.0.0`. The `ss -ltnp` check in each verify step catches a launch that didn't pass it.
> Remote access is always an SSH tunnel; the only deployment safe to expose is Modal, which
> fronts the API with an `X-API-Key` proxy and keeps the API itself on loopback.

**Shared machines.** Before any launch or kill, confirm the target GPUs/PIDs are yours:
`nvidia-smi --query-compute-apps=pid --format=csv,noheader`, then `ps -o user,cmd -p <pid>`.
Never kill a PID you have not identified as your own.

**Launch rules baked into the code:**

- **Must be launched via `uv run … -m skyrl.tinker.api`** — the API reconstructs its uv flags
  from the parent command line to spawn the engine subprocess, and refuses to start otherwise.
  The GPU backend comes from `--extra gpu` (it carries `jax[cuda12]`), not `--extra jax`.
- **`--port 8001` is not optional**: the API's default port is 8000 — the sampler's port.
- **`--checkpoints-base` and `--external-inference-lora-base` default to `/tmp`** — forgetting
  either silently writes state to `/tmp`.
- **`--backend-config` keys are `extra="forbid"`**: a typo'd key is a hard engine-startup
  failure, which takes the API down with it (engine and API live or die together).
- The API and engine are separate processes sharing one database. Under concurrent
  clients SQLite stalls on its 30 s busy-timeout (`asample` + `save_weights`) — use Postgres
  for multi-client servers (local quick start). A single-run server is fine on SQLite (RunPod).

**LoRA adapter slots.** `max_lora_adapters` = number of trainable adapters. With
`--external-inference-url` (always, in this runbook) all N slots serve user models; without
it, slot 0 is reserved for the base model (and `max_lora_adapters: 1` refuses to start). The
startup log prints the usable-slot count — grep `adapter slots`. Each rank-64 slot costs
1.68 GiB/GPU (9B, TP=2: stacked LoRA weights + fp32 grad/mu/nu); cost scales with rank.

**Session expiry frees dead clients' slots.** A dead client pins its slot until session
expiry: `--session-timeout-sec` (default 300 s) + up to `--session-cleanup-interval-sec`
(default 60 s; the sweep runs between engine batches, so a long `forward_backward` delays
it). Lower the timeout instead of restarting the trainer if slot churn is frequent. A client
retrying `create_lora_training_client` in a tight loop against a full server can livelock
with expiry; retry with ≥30 s backoff.

> **Fixed 2026-08-16** (`skyrl/tinker/engine.py`): a client that died before sending its
> first heartbeat left `last_heartbeat_at` NULL, which the stale-session query silently
> excluded — the session never expired and its model pinned the slot **forever** (reproduced
> on RunPod by short-lived smoke-test scripts; the only cure was a trainer restart). The
> cleanup now falls back to `created_at` when no heartbeat was ever recorded.

**vLLM ↔ trainer contract.** The trainer publishes each adapter as a directory under
`--external-inference-lora-base` and references it **by name**; vLLM loads it on demand via
the filesystem resolver. Both processes must therefore share a filesystem, and the vLLM
launch must carry all three of: `--enable-lora`,
`VLLM_LORA_RESOLVER_CACHE_DIR=<same dir>`, `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`
(the last attaches `/v1/unload_lora_adapter`, which the trainer's quiet-adapter sweep calls;
without it adapters accumulate until vLLM restarts — 25 observed in one run). A missing
resolver dir means **every LoRA sample 404s** with "model does not exist" while base-model
sampling works — reproduced on RunPod 2026-08-16.

---

## Local (4× RTX 5000 Ada)

32 GB cards, sm_89, PCIe (no NVLink). 9B primary; [2B/4B variants](#2b--4b-variants).
Run steps 1–4 in order; steps 2 and 3 can start concurrently (different GPU pairs).

### 1. Start PostgreSQL

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

Why each non-default flag is there (all re-verified 2026-08-11 on vLLM 0.25.1; the LoRA
env-var trio is explained under [Invariants](#invariants-every-environment)):

| Flag / env var | Reason |
|---|---|
| `NCCL_NET=Socket` | **Local box only.** NCCL 2.28.9 segfaults on any 2-GPU collective on this box (re-verified: removal = SIGSEGV). Needed at TP=2. Not needed on RunPod's Blackwell hosts. |
| `--max-num-batched-tokens 1024` | **32 GB cards.** Caps the prompt-logprobs fp32 transient (chunk × 248,320 vocab × 4 B). At 2048 a single 4k-token `prompt_logprobs` request allocates 1.89 GiB and **kills the engine** (reproduced). At 1024 the same load survives; throughput cost vs 2048 is ~1%. Do not raise it here. 96 GB cards absorb the transient at vLLM's default chunking (RunPod sets no override). |
| `--gpu-memory-utilization 0.90` | +10.6% KV cache vs 0.85 (845k vs 764k tokens) at identical throughput; survived 24 concurrent 4k-token logprob probes under decode load. (vLLM ≥0.21 counts CUDA-graph memory inside this budget, so 0.90 today ≈ 0.87 in pre-0.21 terms.) |
| `--max-num-seqs 256` | 128 loses ~13% throughput at 256-seq offered load; 256 measured 2,733 tok/s aggregate. |
| `--max-logprobs 128` | vLLM default is 20; client `topk_prompt_logprobs` above the cap = 400 per request. |
| `--max-model-len 5120` | Must exceed the trainer's 4096 cap: probing prompt logprobs on a max-length trajectory needs prompt + 1 token, so equal caps 400 every end-of-trajectory probe. (RunPod currently runs 4096 — see its gotchas.) |
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

### Stop, restart, and reset (local)

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
`skyrl.tinker.api` processes on fixed ports. Never stop Postgres.

| Restart | When |
|---------|------|
| **Trainer** | After any JAX OOM (allocator high-water never drops; the request itself returns HTTP 400 without killing the server). Also clears leaked adapter slots — though `--session-timeout-sec` handles those without a restart. |
| **vLLM** | When sampling stalls. Also `rm -rf /dev/shm/$USER/qwen35_${TAG}_lora_models` while it's down — published adapter directories are never garbage-collected. |

**Reset state** (stop the trainer first):

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

### 2B / 4B variants

Same commands as the 9B quick start with `TAG`, `MODEL`, `MEM_FRAC`, and `BACKEND_CONFIG`
swapped per the [configuration reference](#configuration-reference). The 2B sampler
additionally simplifies to a single card (boot-verified on 0.25.1, 1.36M KV tokens):

```bash
# 2B sampler: CUDA_VISIBLE_DEVICES=3, --tensor-parallel-size 1, --max-model-len 4096,
# drop NCCL_NET (single GPU) and --max-num-seqs/--max-num-batched-tokens overrides.
```

The old 2B path (`uv run --no-sync --extra fsdp vllm serve`, using the venv's vLLM) still
works but depends on the venv staying at its currently-installed vLLM — one `uv sync` breaks
it. Prefer the isolated launch.

---

## RunPod (2× RTX PRO 6000 Blackwell)

96 GB cards, sm_120, PCIe Gen5 x16 through host bridges (no NVLink, `nvidia-smi topo -m`
shows `NODE`). Everything is scripted under **`tools/runpod/`** — read its README for
day-to-day operations; this section is the reference for how and why.

| File | Purpose |
|---|---|
| `tools/runpod/env.sh` | Every knob: model, ports, TP degrees, memory split, backend config |
| `tools/runpod/bootstrap.sh` | Idempotent env setup on the pod: uv, two venvs, model download |
| `tools/runpod/run_sampler.sh`, `run_trainer.sh` | One role each; run as tmux window commands |
| `tools/runpod/launch.sh` | Starts both roles in tmux session `tinker`, waits for health |
| `tools/runpod/boot.sh` | Pod start-command hook: host-key restore + bootstrap + launch |
| `tools/runpod/ssh_proxy.sh` | Local `ProxyCommand`: resolves the pod's current IP/port per connection |

### Topology: co-located TP=2, and why

The rl-hammer RL loop is **strictly synchronous** — trainer and sampler never run at the
same time — so a one-GPU-per-role split caps utilization at 50% by construction. Both roles
therefore span both GPUs. TP=2 scales poorly over the PCIe host-bridge link (trainer 1.13×,
sampler 1.31×, measured 2026-08-16), but poor scaling of otherwise-idle GPU-seconds still
wins: **146 s/step vs 173 s/step** for the tuned split topology at production geometry
(578k trained + 131k sampled tokens/step), and 232 s/step for the original untuned split.

Alternatives measured and rejected:

- **DP=2 sampler** (two replicas): best sampler throughput (4,231 tok/s, 1.78×) but only 8%
  better end-to-end; needs a load balancer, and with 1 slot per replica the trainer's unload
  calls land on one replica only — the other accumulates adapters until it wedges.
- **1-GPU pod** (both roles on one 96 GB card): forces gradient checkpointing back on and
  halves KV → ~262 s/step. Half the cost, 1.8× slower.
- **4-GPU pod**: the trainer dominates step time and its TP scaling is already saturating at
  2 GPUs on this fabric.

Per-GPU memory split, sized against measured peaks (both fractions are of 96 GB):

| | Fraction | Purpose |
|---|---|---|
| Trainer (XLA) | 0.50 → 48 GiB pool | vs 42.2 GiB measured peak (TP=2, no ckpt, 4096-token seqs) |
| Sampler (vLLM) | 0.32 → 30.7 GiB | 8.8 GiB weights (TP=2) + KV: 918k tokens = 224 concurrent 4096-token seqs |
| Free | 0.18 → ~17 GiB | XLA command buffers (outside the pool; exhaustion = exit 134) + vLLM workspace |

The realistic production step needs ~670k KV tokens; the absolute worst case (256 × 4096)
exceeds 918k, in which case vLLM **queues** the overflow — degraded tail latency, not failure.

### Backend config (deltas from the local 9B quick start)

From `env.sh`, all measured 2026-08-16 on this hardware (sweep logs land in
`/workspace/bench/` on the pod):

| Delta | Effect |
|---|---|
| `gradient_checkpointing: false` | **+29%** (3,797 → 4,917 tok/s single-GPU). The ~33 GiB no-ckpt transient that forces checkpointing on 32 GB cards fits here (halved per GPU at TP=2). |
| `loss_chunk_size: 0`, `linear_attention_chunk_size: 128` | +9% and +2% — the local retune, reconfirmed on Blackwell. |
| No `NCCL_NET=Socket`, no `XLA_FLAGS` all-reduce override | Neither failure mode reproduces on these hosts — both are local-box workarounds. |
| No `--max-num-batched-tokens` override on vLLM | The 32 GB prompt-logprob transient cap doesn't apply at 96 GB. |
| vLLM `--max-logprobs 64`, `--max-model-len 4096` | Sized to the rl-hammer workload (top-64 prompt logprobs, 4096-token cap). **Caveat:** end-of-trajectory prompt-logprob probes on exactly-4096-token trajectories will 400 (needs prompt+1); raise to 5120 as on local if that path is ever exercised. |
| Database: SQLite on `/workspace` | Single-run server; see the Postgres note under Invariants before pointing multiple concurrent clients at it. |

### Creating a pod

Prereqs: `runpodctl` configured (`~/.runpod/config.toml` holds the API key, also used by
`ssh_proxy.sh`), and your SSH key registered on the RunPod account.

Create via the GraphQL API (`runpodctl` can't express the start command). The essentials:

```
mutation podFindAndDeployOnDemand(input: {
  cloudType: SECURE            # community had no 2x stock when checked
  gpuCount: 2
  gpuTypeId: "NVIDIA RTX PRO 6000 Blackwell Server Edition"
  imageName: "nvidia/cuda:13.0.3-devel-ubuntu24.04"   # devel: FlashInfer JITs via nvcc
  ports: "22/tcp"  supportPublicIp: true              # direct TCP SSH, not the HTTP proxy
  volumeInGb: 150  containerDiskInGb: 30  volumeMountPath: "/workspace"
  env: [{key: "EXTRA_PUBKEY", value: "<your ssh pubkey>"}]
  dockerArgs: "bash -c 'apt-get update && apt-get install -y --no-install-recommends \
    openssh-server rsync tmux curl git ninja-build build-essential; \
    mkdir -p /run/sshd /root/.ssh; chmod 700 /root/.ssh; \
    printf \"%s\\n\" \"$PUBLIC_KEY\" \"$EXTRA_PUBKEY\" >> /root/.ssh/authorized_keys; \
    chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; \
    [ -x /workspace/SkyRL/tools/runpod/boot.sh ] && \
      nohup /workspace/SkyRL/tools/runpod/boot.sh > /workspace/boot.log 2>&1 & \
    sleep infinity'"
})
```

If deployment fails with a resources error, shrink the disks before shrinking anything else —
150/30 GB was the binding constraint in practice. ~$4.18/hr for 2 GPUs on secure cloud
(2026-08 pricing; spot bids offered no discount over on-demand when checked).

First boot on a fresh volume, after the pod is RUNNING:

```bash
# point the local ssh config at the new pod id (see Access below), then:
rsync -rlptz --omit-dir-times --exclude='.git' --exclude='.venv' --exclude='logs' \
  --exclude='__pycache__' --exclude='*.egg-info' --exclude='node_modules' \
  $REPO/ runpod-tinker:/workspace/SkyRL/
ssh runpod-tinker 'setsid nohup /workspace/SkyRL/tools/runpod/boot.sh \
  > /workspace/boot.log 2>&1 < /dev/null &'
```

~25–35 min: venvs + model download + cold FlashInfer JIT. Every later start is ~5 min
(everything persistent — venvs, model, caches, state — lives on `/workspace`).

### Access: tunnel + self-resolving SSH

Nothing is exposed publicly except SSH. Local plumbing, set up once:

**`~/.ssh/config`** — no fixed address; the ProxyCommand asks the RunPod API for the pod's
current IP/port on every connection, so stop/start cycles need no config edits:

```
Host runpod-tinker
  User root
  IdentityFile ~/.ssh/id_ed25519
  ProxyCommand /storage_slow/ajoe/code/SkyRL/tools/runpod/ssh_proxy.sh <pod-id>
  StrictHostKeyChecking accept-new
  LocalForward 18001 127.0.0.1:8001    # trainer  -> base_url http://localhost:18001
  LocalForward 18000 127.0.0.1:8000    # sampler (debugging)
  ServerAliveInterval 30
  ServerAliveCountMax 4
```

Local ports are 18000/18001 (not 8000/8001) so `pytest tests/tinker/` can still bind its
fixed ports. The pod's host keys are persisted in `/workspace/ssh_host_keys` and restored by
`boot.sh`, so restarts don't trip host-key verification.

**Tunnel as a systemd user service** (`~/.config/systemd/user/runpod-tinker-tunnel.service`):

```ini
[Unit]
Description=SSH tunnel to RunPod Tinker server
StartLimitIntervalSec=0
[Service]
ExecStart=/usr/bin/ssh -N -o ExitOnForwardFailure=yes -o BatchMode=yes runpod-tinker
Restart=always
RestartSec=10
[Install]
WantedBy=default.target
```

`systemctl --user enable --now runpod-tinker-tunnel && loginctl enable-linger`. It retries
every 10 s, self-heals across drops and pod restarts, and idles harmlessly while the pod is
stopped. Don't use RunPod's HTTP proxy instead: it times requests out around 100 s and
`retrieve_future` long-polls up to 300 s.

**Client config**: `TINKER_BASE_URL=http://localhost:18001`, and `TINKER_API_KEY=tml-<anything>`
— the server ignores the key (the tunnel is the auth), but the SDK refuses keys without the
`tml-` prefix client-side.

### Operating

```bash
runpodctl pod list
runpodctl pod stop <id>          # ~$0.13/day for disk; GPUs released
runpodctl pod start <id>         # boot.sh relaunches everything unattended
runpodctl pod remove <id>        # DESTROYS /workspace
ssh runpod-tinker                # then: tmux attach -t tinker (windows: sampler, trainer)
                                 # logs also tee to /workspace/logs/{sampler,trainer}.log
/workspace/bin/uvx nvitop        # GPU monitor (uv lives at /workspace/bin on the pod)
```

**RunPod-specific gotchas** (each one hit in practice, 2026-08-16):

- **A stopped pod can lose its GPUs.** The host may reallocate them; `pod start` then fails
  with "not enough free GPUs" and the volume cannot migrate to another host. Recovery =
  create a fresh pod and re-run first-boot (~30 min). Budget for this before stopping a pod
  you need again soon.
- **`setsid` for anything launched over ssh.** A bare `nohup … &` in an ssh command dies with
  the session — a launch can silently do nothing. (`boot.sh` via the pod start command is
  not affected.)
- **The container overlay is wiped on every stop/start** — apt packages, `/root`, everything
  outside `/workspace`. `boot.sh` rebuilds it; anything you add by hand must be re-added or
  moved to `/workspace`.
- **`/workspace` is network-backed (MooseFS) and rejects `chown`** — rsync with `-rlptz`
  (not `-a`), or every transfer exits 23.
- **Restarting only one role**: `tmux respawn-window -k -t tinker:trainer` (or `:sampler`) —
  each window runs its `run_*.sh` directly. Note `launch.sh`'s health-check short-circuit:
  it does nothing while the trainer answers healthz, so it cannot restart a dead sampler
  beside a live trainer — use `respawn-window` for that case too.

**Teardown checklist** (order matters; nothing survives except `tools/runpod/` in git):

```bash
runpodctl pod remove <id>                                  # pod + volume
systemctl --user disable --now runpod-tinker-tunnel
rm ~/.config/systemd/user/runpod-tinker-tunnel.service && systemctl --user daemon-reload
# remove the Host runpod-tinker block from ~/.ssh/config
ssh-keygen -R runpod-tinker
loginctl disable-linger                                    # if enabled only for the tunnel
runpodctl pod list                                         # must be empty
```

---

## H100 box (3× H100 80 GB)

GPUs 0,1 = trainer (TP=2); GPU 2 = vLLM (TP=1). Up to 4 concurrent clients.
Differences from local (not re-benchmarked; last verified on that box):

- `SKYRL_CUDNN_MAX_HEAD_DIM=256` (sm_90 native cuDNN attention — works there;
  **does not transfer to Blackwell**, see [Attention backend](#attention-backend))
- `TMPDIR=/root/tmp`, `UV_PROJECT_ENVIRONMENT=.venv-jax`, `VLLM_USE_DEEP_GEMM=0`
- Trainer: `max_lora_adapters: 4`, `loss_chunk_size: 64` (sized for 4 concurrent
  `forward_backward`; the local chunking measurements don't transfer directly — re-measure
  before changing it there)
- vLLM: `--max-loras 9 --max-num-seqs 512 --max-num-batched-tokens 16384 --max-model-len 4096
  --limit-mm-per-prompt '{"image":0,"video":0}'` (80 GB cards tolerate the larger prefill
  chunk; the sm_89 logprob-transient math does not apply at these margins)

---

## Modal B200 (one 192 GB GPU, single container)

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

## Configuration reference

### Backend configs by model (local box)

| Model | MEM_FRAC | BACKEND_CONFIG deltas from the local 9B quick start |
|-------|----------|-----------------------------------------------|
| 2B | 0.95 | `"tensor_parallel_size":1,"fully_sharded_data_parallel_size":2`; drop `loss_chunk_size`/`linear_attention_chunk_size`/bucketing keys |
| 4B | 0.90 | none (same shape as 9B) |
| 9B | 0.90 | — (the quick-start config) |
| 9B, 2 LoRAs | 0.90 | `"max_lora_adapters":2,"max_lora_rank":32,"loss_chunk_size":256` (measured 1,538 tok/s, peak 26.6–27.1 GiB of the 28.3 GiB pool — fits, verified) |

Model paths: `/scr1/public_models/huggingface/Qwen/Qwen3.5-{2B,4B,9B}-Base` (local);
`/workspace/models/Qwen/Qwen3.5-9B-Base` (RunPod). RunPod's config lives in
`tools/runpod/env.sh` — see [its section](#backend-config-deltas-from-the-local-9b-quick-start).

### Measured tuning (9B; tok/s = trained tokens/s per request)

| Knob | Setting | Why |
|---|---|---|
| `loss_chunk_size` | **0 (disabled)** for 1 slot | +9% over the old 128 (local 2026-08-11, reconfirmed RunPod 2026-08-16); peak memory +0.6 GiB, irrelevant at 1 slot. Use **256** for the 2-slot config where memory is tight (chunk 64 costs 7%). |
| `linear_attention_chunk_size` | **128** (default 64) | +2% tok/s **and** −1.0 GiB peak. 32 is worse on both axes. |
| `train_bucket_seq_len_per_micro_batch` | true | 1.42× on production mixed-length batches (2026-07-24). Costs ~7 XLA executables (~170 s each to compile, cached) + ~1.4 GiB command buffers. |
| `gradient_checkpointing` | **hardware-dependent** | 32 GB cards: **true**, required — without it the 4096-token backward tries a 33 GiB allocation (reproduced). 96 GB cards: **false**, +29% (RunPod, measured; peak 74.6 GiB solo / 42.2 GiB per GPU at TP=2). |
| `train_micro_batch_size` | 1 | Compute is per-sequence; larger values only change padding behavior. |

### Trainer environment variables

| Variable | Scope | Why |
|----------|-------|-----|
| `XLA_PYTHON_CLIENT_PREALLOCATE=true` | all | One contiguous slab. Perf-neutral, but fragmented regions OOM the ~10 GiB `forward_backward` workspace on long runs. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | all | Local: 0.90 (4B/9B), 0.95 (2B). RunPod co-located: 0.50. Never so high that XLA command buffers (**outside** the pool, ~3 GiB) starve — the process aborts with exit 134. |
| `JAX_COMPILATION_CACHE_DIR` | all | Each sequence-length bucket costs ~170 s to compile; the cache makes restarts cheap. Safe to `rm -rf`. |
| `XLA_FLAGS=--xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false` | **local only** | Removal reproduces `INVALID_ARGUMENT: Unsupported AllReduce kernel` on every TP=2 `forward_backward` (sm_89, no NVLink). Not needed on Blackwell (verified 2026-08-16). |
| `NCCL_NET=Socket` | **local only** | NCCL 2.28.9 segfaults on any 2-GPU collective on that box (removal = SIGSEGV, reproduced). Needed by trainer and sampler there; not on RunPod. |

JAX CUDA plugin (local venv): `jax`/`jaxlib` 0.10.2 with `jax-cuda12-{plugin,pjrt}` **0.10.2**
(upgraded 2026-08-11 to match; perf-neutral, removes the boot warning about a 0.9.2 plugin).
If a jax upgrade ever reintroduces that warning, reinstall matching versions:
`uv pip install --python .venv/bin/python "jax-cuda12-plugin[with-cuda]==<jaxlib version>" "jax-cuda12-pjrt==<jaxlib version>"`.

### Hard constraints (each one reproduced, not theoretical)

- **TP for 4B/9B; never FSDP** — the 248K-vocab `lm_head` logits replicate under FSDP,
  doubling the transient and OOMing (reproduced on 32 GB; untested headroom on 96 GB, but TP
  is the proven path there too).
- **Max trainable sequence length: 4096** on 32 GB cards — the 6144 bucket OOMs (16.8 GiB
  temp-arena allocation failure, reproduced 2026-08-11). Cap client sequences accordingly.
  (RunPod also runs a 4096 cap, by workload choice; larger buckets untested there.)
- **2 LoRA slots at rank 64 OOM at 4096 tokens on 32 GB** (16.71 GiB temp arena, reproduced)
  — use rank 32 for 2 slots there. Prefer lowering `max_lora_rank` over losing sequence length.
- **Do not enable the XLA latency-hiding scheduler** — inflates startup memory ~8.5 → ~16 GiB/card.
- **Do not use `--enable-prefix-caching`** — 0% hit rate on this workload (prefill is shared
  across `num_samples` already), costs 15% KV. It also shifted logprobs by up to 6.6e-2 on
  this hybrid model, and the RL loss uses sampling logprobs for the importance ratio.
- **Do not use MTP speculative decoding on the 4B/9B sampler** — −46% at production
  concurrency (compute-bound regime).

### Attention backend

Qwen3.5's `head_dim=256` exceeds cuDNN's 128 cap on **both** Ada (sm_89) and Blackwell
(sm_120 — `SKYRL_CUDNN_MAX_HEAD_DIM=256` reproduces `Num hidden_dim should be less than or
equal to 128` there, 2026-08-16): training falls back to Pallas/Triton automatically. Leave
`SKYRL_CUDNN_MAX_HEAD_DIM` unset except on H100 (sm_90), where the native kernel works.

---

## Troubleshooting

### `Maximum number of LoRA adapters (N) reached`

Usually the cap doing its job: `max_lora_adapters: 1` allows exactly one concurrent client.
A dead client's slot frees at session expiry (see Invariants — including the fixed
NULL-heartbeat case, which before 2026-08-16 pinned slots forever). For faster turnaround,
launch with `--session-timeout-sec 60`. To clear immediately, restart the trainer — but first
check for stale queued requests that would replay on startup:

```bash
$PG_BIN/psql -h $STATE_DIR/skyrl_tinker_pg_data -U postgres -d skyrl_tinker \
  -c "SELECT request_type, status, count(*) FROM futures WHERE status='PENDING' GROUP BY 1,2;"
```

If any `CREATE_MODEL` is `PENDING`, mark it failed (the `UPDATE futures` command in the local
reset section) before relaunching. (On RunPod's SQLite: same query via
`/workspace/SkyRL/.venv/bin/python` + `sqlite3` against `/workspace/state/tinker.db`.)

### LoRA samples 404 with "model `<name>` does not exist"

The vLLM launch is missing the filesystem-resolver env vars — see the vLLM ↔ trainer
contract under Invariants. Base-model sampling working while adapter sampling 404s is the
signature (reproduced on RunPod 2026-08-16).

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
(local: `/dev/shm/$USER/..._lora_models`; RunPod: `/workspace/state/lora_models`) are what
accumulate on disk (the server never deletes them) — `rm -rf` when vLLM is down.

### Samples fail with a 400 from `/v1/completions`

The error detail now includes vLLM's reason. Known triggers: `topk_prompt_logprobs` above
`--max-logprobs`, `temperature=0` with `num_samples>1`, prompt+max_tokens over `--max-model-len`.
Note each forwarded sample also has a hard 300 s timeout (10 s connect).

### vLLM engine dies on prompt-logprob requests (32 GB cards)

`prompt_logprobs` materializes full-vocab fp32 logits for every prompt position in the prefill
chunk: chunk 2048 × 248,320 vocab × 4 B = 1.89 GiB in one allocation — reproduced killing the
engine at `--max-num-batched-tokens 2048` regardless of memory utilization. The local launch's
`1024` is the fix (verified surviving 24 concurrent probes); if it ever recurs, drop to 512
(costs ~3% throughput). Not applicable at 80–96 GB margins.

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

The sampler already runs 0.25.1 out-of-tree (local: `uv run --isolated`; RunPod: its own venv
at `/workspace/venv-vllm`). The repo pins `vllm==0.23.0` exactly, and `uv lock --upgrade`
would move torch off 2.11, breaking four hand-built wheels pinned to the torch-2.11 ABI
(`flash-attn`, `causal-conv1d`, `mamba-ssm`, `transformer-engine-torch`).

---

## Performance notes

### Local (9B, TP=2, 32 GB Ada, measured 2026-08-11)

- **Trainer**: ~1,675 trained tok/s per `forward_backward` request at 4096 tokens with the
  quick-start config (was ~1,546 before the `loss_chunk_size`/`linear_attention_chunk_size`
  retune — +8%). Per-execution efficiency was already near this card's practical roofline;
  the win came from removing overhead, not kernels.
- **Sampler**: ~2,370 gen tok/s at 128 concurrent seqs, ~2,730 at 256; KV cache 845k tokens.
  The sampler is far from saturated at production concurrency — raising `num_samples` or
  prompt concurrency is close to free.
- Sampler weight publish (`save_weights_and_get_sampling_client`): 2–4 s.
- Trainer co-residency with the sampler (historical): MEM_FRAC 0.90 → 0.65 costs 4.1% on
  `forward_backward`; co-residency itself adds nothing beyond the smaller pool.

### RunPod (9B, 96 GB Blackwell, measured 2026-08-16)

Trainer, warm `forward_backward`, 32 × 4096-token seqs, 1 rank-64 slot:

| Config | tok/s | peak |
|---|---|---|
| Modal-derived baseline (ckpt on, `loss_chunk 128`), 1 GPU | 3,271 | 41.3 GiB |
| + `loss_chunk 0`, `linear_attention_chunk_size 128` | 3,797 | 36.3 GiB |
| + `gradient_checkpointing false` | 4,917 | 74.6 GiB |
| same, TP=2 **(deployed)** | 5,577 | 42.2 GiB/GPU |

Sampler, production wave (16 prompts × n=16, 2k-token prompts, 512 gen, logprobs, LoRA):
2,380 tok/s on 1 GPU → 3,120 at TP=2 (deployed) → 4,231 at DP=2 (rejected — see topology).
Through the full API: warm fb ~5,200 tok/s (≈6% engine/API overhead), `optim_step` <0.5 s,
weight publish 2–3 s.

Step time at production geometry (578k trained + 131k sampled tokens): 232 s original →
173 s tuned-split → **146 s deployed co-located TP=2** (1.59×).

### Both environments

- **Biggest end-to-end lever is client-side**: async pipelining (train batch *i* while
  sampling batch *i+1*) recovers the ~35% duty-cycle loss of a synchronous RL loop
  (measured 2026-08-10). Server-side coalescing would not help: compute is per-sequence.
  With an API-based defender (e.g. gpt-4o-mini), all GPUs additionally idle during
  defender calls.

### Rejected (measured, do not re-add)

| Setting | Result |
|---|---|
| `loss_chunk_size: 64` | −7% tok/s vs 128, no memory benefit at these shapes |
| `linear_attention_chunk_size: 32` | −5% tok/s **and** +0.6 GiB peak |
| `--max-num-batched-tokens 2048` (32 GB) | engine death on prompt-logprob requests (both 0.85 and 0.90 util) |
| `--gpu-memory-utilization 0.85` (solo card) | −10.6% KV vs 0.90 for zero measured safety benefit on 0.25.1 |
| `--max-num-seqs 128` | −13% throughput at 256-seq offered load |
| MTP spec-decode (4B/9B) | −46% at 128 seqs (compute-bound) |
| `--enable-prefix-caching` | 0% hit rate, −15% KV, logprob drift up to 6.6e-2 on this hybrid model |
| `SKYRL_CUDNN_MAX_HEAD_DIM=256` (sm_89/sm_120) | cuDNN rejects head_dim 256; H100 (sm_90) only |
| Trainer TP=2 with ckpt on (96 GB) | 4,209 tok/s — no gain over 1-GPU no-ckpt (4,917) |
| DP=2 sampler (96 GB) | +8% end-to-end, but needs an LB and per-replica adapter unloads diverge |
| `sample_max_num_sequences` | no effect with external inference |
| `forwarding_inference_max_connections` | no effect with external inference (only read by the megatron/fsdp forwarding client) |
