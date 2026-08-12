"""Tinker API server on Modal: Qwen3.5-9B-Base, 8 trainable LoRAs, 4096 tokens.

Topology (one container, one 192 GB B200 shared by both roles):

    GPU 0   Tinker/JAX trainer, TP=1, 8 trainable LoRA slots     127.0.0.1:8001
    GPU 0   vLLM sampler, TP=1, --max-loras 8                    127.0.0.1:8000

The memory split between the co-resident processes is defined below
(TRAINER_MEM_FRACTION / SAMPLER_MEM_UTILIZATION).

Both roles live in one container on purpose. The trainer hands vLLM a filesystem
*path* to each adapter (skyrl/tinker/extra/external_inference.py:145-157), so the
two processes must share a filesystem. Splitting them across two Modal Functions
would mean round-tripping every adapter through a Volume commit/reload.

The only thing published to the internet is a reverse proxy that requires
`X-API-Key` -- the header the Tinker SDK sends (tinker/_client.py:163). Neither
the Tinker API nor vLLM authenticates, and the Tinker API can write files under
--checkpoints-base, so they stay bound to loopback inside the container.

Usage:

    modal run   tools/modal_tinker_server.py::download_model   # once, ~18 GB
    modal deploy tools/modal_tinker_server.py

Then locally:

    export TINKER_API_KEY=tml-...            # value of the skyrl-tinker-key secret
    export TINKER_BASE_URL=https://<workspace>--skyrl-tinker-server.modal.run
    python -c "import tinker; print(tinker.ServiceClient().get_server_capabilities())"
"""

import hmac
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

# --- knobs ---------------------------------------------------------------
APP_NAME = "skyrl-tinker"
MODEL_REPO = "Qwen/Qwen3.5-9B-Base"
# `modal run` validates every Function in the App before running any of them, so a
# GPU your account can't schedule blocks even the CPU-only download. The override
# also lets you exercise the whole pipeline on cheaper hardware.
GPU_SPEC = os.environ.get("SKYRL_MODAL_GPU", "B200")
MAX_LORA_ADAPTERS = 8

# One 192 GB B200 hosts both roles, so the two processes must split it explicitly.
# Both knobs are fractions of *total* memory, not free memory, so they compose as
# long as they sum below 1.0 -- but only if each process is actually held to its
# share, which is why the trainer preallocates.
#
#   trainer  0.45 -> ~86 GiB   vs ~18.2 weights + 8 x 3.37 LoRA + ~29 activations
#   sampler  0.40 -> ~77 GiB   vs ~17.6 weights + ~1.3 LoRA, rest is KV
#   free     0.15 -> ~29 GiB   XLA command buffers, which live outside the BFC pool
#
# That last line is not slack. Exhausting it aborts the process with exit 134
# rather than raising, and on a shared card there's no second GPU to absorb it.
TRAINER_MEM_FRACTION = "0.45"
SAMPLER_MEM_UTILIZATION = "0.40"
MAX_LORA_RANK = 64
MAX_SEQ_LEN = 4096
# vLLM's concurrency ceiling. 256 is not memory-bound here: Qwen3.5-9B is hybrid
# (8 of 32 layers full attention, the rest constant-state GDN), so KV costs only
# 32 KB/token and 256 x 4096 tokens needs 32 GiB against a ~52 GiB pool. Raising
# this spends KV that is otherwise unusable, which is why it is a knob.
MAX_NUM_SEQS = int(os.environ.get("SKYRL_MAX_NUM_SEQS", "256"))
VLLM_VERSION = "0.25.1"
VLLM_INDEX = f"https://wheels.vllm.ai/{VLLM_VERSION}/cu130"

# Container-internal paths.
REPO = "/root/SkyRL"
MODEL_DIR = f"/models/{MODEL_REPO}"
# Trainer writes, vLLM reads -- so this must be a path both processes can see, which
# is why they share a container. On the state Volume rather than container-local disk:
# for the JAX backend with external inference, save_weights_for_sampler writes the
# adapter *here* and not under checkpoints_base (engine.py:683-685), so keeping it on
# the overlay meant every sampler sync died with the container.
LORA_DIR = "/state/lora_models"
CKPT_DIR = "/state/checkpoints"
JAX_CACHE = "/cache/jax_compilation_cache"
# On the state Volume, NOT container-local. This is not "just session state": the
# models table and the checkpoints index live here, and validate_checkpoint
# (api.py:1175) refuses any load_weights whose checkpoint row is missing. Keeping
# it on the container overlay meant `modal app stop` silently destroyed the only
# record of every checkpoint, so weights survived on the Volume but became
# unloadable -- resume 404s with "Model not found".
DB_PATH = "/state/tinker.db"

TRAINER_PORT = 8001
SAMPLER_PORT = 8000

# 8 trainable adapters, not 7 + a base placeholder: engine.py:258-259 sets
# backend_config.external_inference whenever --external-inference-url is given,
# which releases the reserved slot 0 (jax.py:725).
BACKEND_CONFIG = {
    "max_lora_adapters": MAX_LORA_ADAPTERS,
    "max_lora_rank": MAX_LORA_RANK,
    "tensor_parallel_size": 1,
    "fully_sharded_data_parallel_size": 1,
    "train_micro_batch_size": 1,
    "gradient_checkpointing": True,
    "loss_chunk_size": 128,
    "train_bucket_seq_len_per_micro_batch": True,
}

# Overlay for A/B deploys: SKYRL_BACKEND_CONFIG_OVERRIDE is JSON merged over the
# defaults above, so the two arms of a benchmark differ only by an env var at
# deploy time rather than by an edit to this file. Read at import, which is when
# Modal serialises the config into the deployed Function.
_override = os.environ.get("SKYRL_BACKEND_CONFIG_OVERRIDE", "").strip()
if _override:
    BACKEND_CONFIG = {**BACKEND_CONFIG, **json.loads(_override)}
    # max_lora_adapters feeds vLLM's --max-loras too; keep them consistent or the
    # trainer can register an adapter the sampler will refuse to schedule.
    MAX_LORA_ADAPTERS = BACKEND_CONFIG["max_lora_adapters"]

# --- volumes -------------------------------------------------------------
models_vol = modal.Volume.from_name("skyrl-tinker-models", create_if_missing=True)
state_vol = modal.Volume.from_name("skyrl-tinker-state", create_if_missing=True)
cache_vol = modal.Volume.from_name("skyrl-tinker-cache", create_if_missing=True)
VOLUMES = {"/models": models_vol, "/state": state_vol, "/cache": cache_vol}

api_key_secret = modal.Secret.from_name("skyrl-tinker-key")

# --- image ---------------------------------------------------------------
# Two prebaked venvs. The jax and vllm dependency trees conflict (CLAUDE.md), and
# resolving vLLM from a custom wheel index at container start would add minutes to
# every cold start.
image = (
    # CUDA *devel*, not runtime, and not debian_slim: FlashInfer's TRTLLM attention and
    # sampling kernels are JIT-compiled with nvcc at first use. The pip wheels ship only
    # CUDA runtime libs, so without a toolchain vLLM dies with "Could not find nvcc".
    # 13.0.x matches torch 2.11.0+cu130.
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    # ninja-build is not optional: FlashInfer's JIT shells out to `ninja` by name to
    # build its kernels, so without it on PATH the engine dies with FileNotFoundError
    # even though nvcc is present.
    .apt_install("git", "curl", "build-essential", "ninja-build")
    .pip_install("uv==0.9.26", "httpx", "starlette", "huggingface_hub[hf_transfer]")
    .env(
        {
            "UV_LINK_MODE": "copy",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "CUDA_HOME": "/usr/local/cuda",
            # Keep JIT output on the cache Volume so kernels compile once ever, not
            # once per container. Modal background-commits Volumes, so it persists.
            "FLASHINFER_CACHE_DIR": "/cache/flashinfer",
            "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer",
        }
    )
    .add_local_dir(
        Path(__file__).resolve().parent.parent,
        remote_path=REPO,
        # Patterns are dockerignore-style, so bare names only match at the root.
        # `**/__pycache__` matters: Modal aborts the build if a mounted file changes
        # mid-upload, and importing this script writes tools/__pycache__/*.pyc.
        ignore=[
            "**/.venv",
            "**/.venv-*",
            "**/.git",
            "**/logs",
            "**/*.pyc",
            "**/__pycache__",
            "**/*.egg-info",
            "**/.pytest_cache",
            "**/node_modules",
        ],
        copy=True,
    )
    .run_commands(
        f"cd {REPO} && uv sync --extra gpu --extra tinker --extra jax",
        "uv venv /opt/vllm --python 3.12",
        f"VIRTUAL_ENV=/opt/vllm uv pip install 'vllm=={VLLM_VERSION}' --index {VLLM_INDEX}",
    )
)

# The weight download needs neither GPU deps nor the repo, so it gets its own thin
# image -- otherwise a broken build of the big image would also block the download.
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(APP_NAME)


@app.function(
    image=download_image,
    volumes={"/models": models_vol},
    secrets=[modal.Secret.from_name("huggingface", required_keys=[])],
    timeout=3600,
)
def download_model():
    """One-time: pull the base weights onto the models Volume."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR, max_workers=8)
    models_vol.commit()
    print(f"downloaded {MODEL_REPO} -> {MODEL_DIR}")
    print(sorted(p.name for p in Path(MODEL_DIR).iterdir()))


@app.function(image=image, volumes=VOLUMES, timeout=900)
def seed_checkpoint_index(model_id: str, checkpoint_ids: str, rank: int = 64, seed: int = 0):
    """Re-register orphaned checkpoints so `load_weights` can resolve them again.

    For checkpoints written while DB_PATH was container-local: the Orbax
    directories survive under /state/checkpoints/<model_id>/, but the rows
    validate_checkpoint looks up died with the container. Optimizer state is not
    reconstructed -- it was never lost, only unindexed.

    No GPU: pure SQLite work against the Volume. The real work runs in the repo
    venv, since sqlmodel and skyrl are installed only there and not in Modal's
    system interpreter.

        modal run tools/modal_tinker_server.py::seed_checkpoint_index \
            --model-id model_705230e0 --checkpoint-ids atk_000128
    """
    proc = subprocess.run(
        [
            f"{REPO}/.venv/bin/python",
            f"{REPO}/tools/seed_checkpoint_index.py",
            "--db-path",
            DB_PATH,
            "--ckpt-dir",
            CKPT_DIR,
            "--base-model",
            MODEL_DIR,
            "--model-id",
            model_id,
            "--checkpoint-ids",
            checkpoint_ids,
            "--rank",
            str(rank),
            "--seed",
            str(seed),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "VIRTUAL_ENV": f"{REPO}/.venv"},
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
        raise RuntimeError(f"seed failed rc={proc.returncode}")
    state_vol.commit()
    return proc.stdout


FLASHINFER_PROBE = r"""
import time, traceback, torch, flashinfer
print("flashinfer", flashinfer.__version__, "torch", torch.__version__,
      "cap", torch.cuda.get_device_capability())

# Drive the exact JIT paths vLLM hits on Blackwell. Each build() shells out to nvcc
# and ninja, so a missing toolchain component raises here -- and, unlike a server
# boot, every failure is reported in one pass instead of one per deploy.
targets = []
try:
    from flashinfer.decode import get_trtllm_gen_fmha_module
    targets.append(("trtllm_gen_fmha (attention)", get_trtllm_gen_fmha_module))
except Exception:
    traceback.print_exc()
try:
    from flashinfer.sampling import get_sampling_module
    targets.append(("sampling", get_sampling_module))
except Exception:
    traceback.print_exc()

failures = []
for name, fn in targets:
    t0 = time.time()
    try:
        fn()
        print(f"JIT OK   {name}  ({time.time() - t0:.1f}s)")
    except Exception as exc:
        failures.append(name)
        print(f"JIT FAIL {name}  ({time.time() - t0:.1f}s)")
        traceback.print_exc()

print("FLASHINFER_FAILURES=" + (",".join(failures) if failures else "none"))
"""


@app.function(image=image, gpu=GPU_SPEC, volumes=VOLUMES, timeout=3600)
def smoke():
    """Prove the toolchain before paying for a server boot.

    Two things get checked: that JAX has kernels for this GPU's compute capability,
    and that FlashInfer can actually JIT-compile. The second matters because vLLM
    builds those kernels lazily during startup profiling, so a missing build
    dependency only surfaces ~10 minutes into a deploy, one package at a time.
    """
    for cmd in (["nvidia-smi", "-L"], ["nvcc", "--version"], ["ninja", "--version"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f"$ {' '.join(cmd)} (rc={r.returncode})\n{r.stdout.strip() or r.stderr.strip()}\n")

    jax_probe = (
        "import jax;"
        "print('jax', jax.__version__, jax.devices());"
        "import jax.numpy as jnp;"
        "x=jnp.ones((2048,2048));"
        "print('matmul ok', float((x@x).sum()))"
    )
    vllm_probe = (
        "import torch,vllm;"
        "print('vllm',vllm.__version__,'torch',torch.__version__,"
        "'cap',torch.cuda.get_device_capability())"
    )
    checks = [
        (f"{REPO}/.venv", f"{REPO}/.venv/bin/python", jax_probe, True),
        ("/opt/vllm", "/opt/vllm/bin/python", vllm_probe, True),
        ("/opt/vllm flashinfer JIT", "/opt/vllm/bin/python", FLASHINFER_PROBE, False),
    ]
    hard_failed = []
    for label, python, code, fatal in checks:
        r = subprocess.run([python, "-c", code], capture_output=True, text=True)
        print(f"--- {label} (rc={r.returncode}) ---\n{r.stdout}{r.stderr[-4000:]}")
        if r.returncode != 0 and fatal:
            hard_failed.append(label)
    if hard_failed:
        raise RuntimeError(f"probe failed: {hard_failed}")


# --- server --------------------------------------------------------------
# "stage" is the single source of truth for boot progress: "ready" and "failed" are
# terminal, everything else means still starting. "error" carries the detail for "failed".
_state = {"error": None, "stage": "cold"}
_procs: list[subprocess.Popen] = []


def _spawn(name: str, argv: list[str], env: dict) -> subprocess.Popen:
    """Run a child, mirroring its output to both a file and this container's stdout.

    The file is what `modal container exec ... tail /root/<name>.log` reads; stdout is
    what `modal app logs` shows. Writing only to the file makes a 20-minute boot
    completely opaque from outside the container.
    """
    log = open(f"/root/{name}.log", "wb")
    proc = subprocess.Popen(argv, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def pump() -> None:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(f"[{name}] {line.decode('utf-8', 'replace')}")
            sys.stdout.flush()

    threading.Thread(target=pump, daemon=True).start()
    _procs.append(proc)
    print(f"[{name}] pid={proc.pid}")
    return proc


def _wait_http(url: str, label: str, timeout: int) -> None:
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        for proc in _procs:
            if proc.poll() is not None:
                raise RuntimeError(f"{label}: a subprocess exited with {proc.returncode} before ready")
        try:
            if httpx.get(url, timeout=5.0).status_code < 500:
                print(f"[{label}] ready after {int(timeout - (deadline - time.time()))}s")
                return
        except Exception:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"{label} not ready within {timeout}s")


def _boot() -> None:
    """Start sampler then trainer, mirroring the loopback topology used locally."""
    for d in (LORA_DIR, CKPT_DIR, JAX_CACHE):
        Path(d).mkdir(parents=True, exist_ok=True)

    # --- sampler, GPU 1 ---
    # --max-loras must be >= max_lora_adapters or the trainer can register an
    # adapter the sampler will silently refuse to schedule.
    # No --enable-prefix-caching: on this hybrid model it shifted logprobs by up to
    # 6.6e-2, and the RL loss uses sampling logprobs for the importance ratio.
    _state["stage"] = "starting sampler"
    _spawn(
        "sampler",
        [
            "/opt/vllm/bin/vllm",
            "serve",
            MODEL_DIR,
            "--served-model-name",
            MODEL_REPO,
            "--host",
            "127.0.0.1",
            "--port",
            str(SAMPLER_PORT),
            "--tensor-parallel-size",
            "1",
            "--enable-lora",
            "--max-lora-rank",
            str(MAX_LORA_RANK),
            "--max-loras",
            str(MAX_LORA_ADAPTERS),
            "--max-model-len",
            str(MAX_SEQ_LEN),
            "--dtype",
            "bfloat16",
            # No --attention-backend pin: let vLLM auto-select, which on Blackwell means
            # FLASHINFER with TRTLLM kernels -- the fast path. The devel image supplies
            # the nvcc they JIT through.
            # Overlaps scheduling with the forward pass. Measured worthwhile locally.
            "--async-scheduling",
            "--gpu-memory-utilization",
            SAMPLER_MEM_UTILIZATION,
            "--max-num-seqs",
            str(MAX_NUM_SEQS),
            "--limit-mm-per-prompt",
            '{"image":0,"video":0}',
        ],
        {
            # Same physical GPU as the trainer -- see the memory split above.
            "CUDA_VISIBLE_DEVICES": "0",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
            # FlashInfer sampling left enabled (the default) now that nvcc exists.
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "TRITON_CACHE_DIR": "/cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor",
        },
    )

    # --- trainer, GPU 0 ---
    # MEM_FRAC 0.90 not 0.95: XLA command buffers live outside the BFC pool and
    # exhausting that headroom aborts the process with exit 134.
    _state["stage"] = "starting trainer"
    _spawn(
        "trainer",
        [
            "uv",
            "run",
            "--active",
            "--no-sync",
            "--extra",
            "gpu",
            "--extra",
            "tinker",
            "--extra",
            "jax",
            "-m",
            "skyrl.tinker.api",
            "--base-model",
            MODEL_DIR,
            "--backend",
            "jax",
            "--host",
            "127.0.0.1",
            "--port",
            str(TRAINER_PORT),
            "--external-inference-url",
            f"http://127.0.0.1:{SAMPLER_PORT}",
            "--external-inference-lora-base",
            LORA_DIR,
            "--checkpoints-base",
            CKPT_DIR,
            "--database-url",
            f"sqlite:///{DB_PATH}",
            "--backend-config",
            json.dumps(BACKEND_CONFIG),
        ],
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "VIRTUAL_ENV": f"{REPO}/.venv",
            "PATH": f"{REPO}/.venv/bin:" + os.environ["PATH"],
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": TRAINER_MEM_FRACTION,
            "JAX_COMPILATION_CACHE_DIR": JAX_CACHE,
        },
    )

    _state["stage"] = "waiting for sampler"
    _wait_http(f"http://127.0.0.1:{SAMPLER_PORT}/v1/models", "sampler", timeout=1800)
    _state["stage"] = "waiting for trainer"
    _wait_http(f"http://127.0.0.1:{TRAINER_PORT}/api/v1/healthz", "trainer", timeout=1800)
    _state["stage"] = "ready"


def _boot_guarded() -> None:
    try:
        _boot()
    except Exception as exc:  # surfaced via GET /_status
        _state["error"] = f"{type(exc).__name__}: {exc}"
        _state["stage"] = "failed"
        print(f"boot failed: {_state['error']}")


@app.function(
    image=image,
    gpu=GPU_SPEC,
    volumes=VOLUMES,
    secrets=[api_key_secret],
    timeout=86400,
    min_containers=1,
    scaledown_window=1800,
    max_containers=1,
)
@modal.concurrent(max_inputs=256)
@modal.asgi_app()
def server():
    import httpx
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    api_key = os.environ["TINKER_API_KEY"]
    upstream = httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{TRAINER_PORT}",
        # /api/v1/retrieve_future long-polls for up to 300s (api.py:1121). Modal's
        # edge caps a single request at 150s and then issues a 303 the client
        # follows; the Tinker SDK sets follow_redirects=True, so that resolves
        # transparently. This timeout only has to exceed the upstream's own.
        timeout=httpx.Timeout(360.0, connect=10.0),
        limits=httpx.Limits(max_connections=256),
    )
    # Hop-by-hop headers plus anything the ASGI server recomputes.
    DROP = {"host", "content-length", "transfer-encoding", "connection", "content-encoding", "keep-alive"}

    threading.Thread(target=_boot_guarded, daemon=True).start()

    async def status(request):
        return JSONResponse({**_state, "model": MODEL_REPO, "max_lora_adapters": MAX_LORA_ADAPTERS})

    async def proxy(request):
        # compare_digest, not !=: this is the only authentication control in front of a
        # service that can write under --checkpoints-base and delete any tenant's
        # checkpoints, so the key comparison should not short-circuit on first mismatch.
        if not hmac.compare_digest(request.headers.get("x-api-key") or "", api_key):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        if _state["stage"] != "ready":
            if _state["error"]:
                return JSONResponse({"detail": f"server failed to start: {_state['error']}"}, status_code=500)
            return JSONResponse({"detail": f"server still starting: {_state['stage']}"}, status_code=503)
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP}
        r = await upstream.request(
            request.method, request.url.path, params=request.query_params, content=body, headers=headers
        )
        out = {k: v for k, v in r.headers.items() if k.lower() not in DROP}
        return Response(content=r.content, status_code=r.status_code, headers=out)

    return Starlette(
        routes=[
            Route("/_status", status, methods=["GET"]),
            Route("/{path:path}", proxy, methods=["GET", "POST", "PUT", "PATCH", "DELETE"]),
        ]
    )
