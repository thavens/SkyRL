#!/usr/bin/env bash
# One-time (but idempotent) environment setup on the pod. Safe to re-run; each
# step is guarded so a post-restart boot only redoes what the container wipe
# actually destroyed.
set -euo pipefail
source "$(dirname "$0")/env.sh"

mkdir -p "$LORA_DIR" "$CKPT_DIR" "$JAX_CACHE" "$LOG_DIR" /workspace/bin \
    /workspace/cache/vllm /workspace/cache/triton /workspace/cache/inductor \
    "$FLASHINFER_CACHE_DIR"

if ! command -v uv >/dev/null; then
    echo "== installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/workspace/bin sh
fi

# Two venvs because the jax and vllm dependency trees conflict (CLAUDE.md).
echo "== syncing trainer venv (jax)"
cd "$REPO"
uv sync --extra gpu --extra tinker --extra jax

if [ ! -x "$VLLM_VENV/bin/vllm" ]; then
    echo "== building sampler venv (vllm $VLLM_VERSION)"
    uv venv "$VLLM_VENV" --python 3.12
    VIRTUAL_ENV="$VLLM_VENV" uv pip install "vllm==${VLLM_VERSION}" --index "$VLLM_INDEX"
fi

# Marker file, not just config.json: an interrupted snapshot_download leaves a
# partial tree that looks plausible but fails at load time.
if [ ! -f "$MODEL_DIR/.download_complete" ]; then
    echo "== downloading $MODEL_REPO (~18 GB)"
    uv run --with "huggingface_hub[hf_transfer]" python - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${MODEL_REPO}", local_dir="${MODEL_DIR}", max_workers=8)
PYEOF
    touch "$MODEL_DIR/.download_complete"
fi

echo "== bootstrap complete"
