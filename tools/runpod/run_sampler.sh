#!/usr/bin/env bash
# vLLM sampler, TP=SAMPLER_TP on SAMPLER_GPUS. Run directly or as the command
# of the tinker:sampler tmux window (restart it with `tmux respawn-window -k`).
#
# --max-loras must be >= max_lora_adapters or the trainer can register an
# adapter the sampler will silently refuse to schedule.
# The resolver env-var trio is the trainer contract: adapters are published as
# directories under LORA_DIR and loaded by name on demand (runbook.md).
# No --enable-prefix-caching: on this hybrid model it shifted logprobs by up
# to 6.6e-2, and the RL loss uses sampling logprobs for the importance ratio.
# No --attention-backend pin: on Blackwell vLLM auto-selects FLASHINFER with
# TRTLLM kernels; the devel image supplies the nvcc they JIT through.
set -uo pipefail
source "$(dirname "$0")/env.sh"

export CUDA_VISIBLE_DEVICES=$SAMPLER_GPUS
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_PLUGINS=lora_filesystem_resolver
export VLLM_LORA_RESOLVER_CACHE_DIR=$LORA_DIR
export VLLM_CACHE_ROOT=/workspace/cache/vllm
export TRITON_CACHE_DIR=/workspace/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/workspace/cache/inductor

exec > >(tee -a "$LOG_DIR/sampler.log") 2>&1
exec "$VLLM_VENV/bin/vllm" serve "$MODEL_DIR" \
    --served-model-name "$MODEL_REPO" \
    --host 127.0.0.1 --port "$SAMPLER_PORT" \
    --tensor-parallel-size "$SAMPLER_TP" \
    --enable-lora --max-lora-rank "$MAX_LORA_RANK" --max-loras "$MAX_LORA_ADAPTERS" \
    --max-model-len "$MAX_SEQ_LEN" \
    --max-logprobs "$MAX_LOGPROBS" \
    --dtype bfloat16 \
    --async-scheduling \
    --gpu-memory-utilization "$SAMPLER_MEM_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --limit-mm-per-prompt '{"image":0,"video":0}'
