#!/usr/bin/env bash
# Tinker API + JAX trainer, TP=TRAINER_TP on TRAINER_GPUS. Run directly or as
# the command of the tinker:trainer tmux window (restart it with
# `tmux respawn-window -k`).
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO"

export CUDA_VISIBLE_DEVICES=$TRAINER_GPUS
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=$TRAINER_MEM_FRACTION
export JAX_COMPILATION_CACHE_DIR=$JAX_CACHE

exec > >(tee -a "$LOG_DIR/trainer.log") 2>&1
exec uv run --no-sync --extra gpu --extra tinker --extra jax -m skyrl.tinker.api \
    --base-model "$MODEL_DIR" \
    --backend jax \
    --host 127.0.0.1 --port "$TRAINER_PORT" \
    --external-inference-url "http://127.0.0.1:$SAMPLER_PORT" \
    --external-inference-lora-base "$LORA_DIR" \
    --checkpoints-base "$CKPT_DIR" \
    --database-url "sqlite:///$DB_PATH" \
    --backend-config "$BACKEND_CONFIG"
