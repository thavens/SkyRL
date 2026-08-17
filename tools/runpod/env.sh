# Shared knobs for the RunPod Tinker server. Sourced by bootstrap.sh and launch.sh.
#
# Topology (2x RTX PRO 6000 Blackwell 96GB): BOTH roles span BOTH GPUs, TP=2.
#
#   GPU 0+1   Tinker/JAX trainer, TP=2, 1 trainable LoRA slot   127.0.0.1:8001
#   GPU 0+1   vLLM sampler, TP=2, --max-loras 1                 127.0.0.1:8000
#
# Co-location is the right call *because* this workload's RL loop is strictly
# synchronous -- the trainer and sampler never run at the same time, so a
# one-GPU-per-role split caps utilization at 50% by construction. TP=2 scales
# poorly over this host's PCIe-bridge link (trainer 1.13x, sampler 1.31x,
# measured 2026-08-16), but poor scaling of otherwise-idle seconds still wins:
# 146 s/step vs 173 s/step for the split topology at production geometry.
# A DP=2 sampler (1.78x) would shave another 8% but needs an LB and leaves
# per-replica adapter unloads unbalanced -- rejected.
#
# To revert to the one-GPU-per-role split: TP=1/1, GPUS 0 / 1, fractions
# 0.90 / 0.90, and gradient_checkpointing stays false (peak 74.6 GiB fits a
# solo 86 GiB pool).
#
# Both servers bind loopback only; the pod exposes nothing but SSH. Reach the
# API through an SSH tunnel (see tools/runpod/README.md).
#
# Everything that must survive a pod stop/start lives under /workspace (the
# pod's volume disk). The container overlay -- including /root and every apt
# package -- is wiped on stop, which is why boot.sh reinstalls the toolchain
# and why uv itself is pinned into /workspace/bin.

export MODEL_REPO="Qwen/Qwen3.5-9B-Base"
export MAX_LORA_ADAPTERS=1
export MAX_LORA_RANK=64
export MAX_SEQ_LEN=4096
export MAX_NUM_SEQS=256
# vLLM caps per-request logprobs at --max-logprobs (default 20). Tinker
# requests with topk_prompt_logprobs up to this value pass through
# external_inference.py as prompt_logprobs=k, so the cap must cover them.
export MAX_LOGPROBS=64
export VLLM_VERSION="0.25.1"
export VLLM_INDEX="https://wheels.vllm.ai/${VLLM_VERSION}/cu130"

export REPO=/workspace/SkyRL
export MODEL_DIR="/workspace/models/${MODEL_REPO}"
export LORA_DIR=/workspace/state/lora_models
export CKPT_DIR=/workspace/state/checkpoints
export DB_PATH=/workspace/state/tinker.db
export JAX_CACHE=/workspace/cache/jax_compilation_cache
export VLLM_VENV=/workspace/venv-vllm
export LOG_DIR=/workspace/logs

export TRAINER_PORT=8001
export SAMPLER_PORT=8000

export TRAINER_TP=2
export SAMPLER_TP=2
export TRAINER_GPUS=0,1
export SAMPLER_GPUS=0,1

# Co-resident memory split, per GPU, against measured peaks (2026-08-16):
#   trainer 0.50 -> 48.0 GiB pool  vs 42.2 GiB peak (TP=2, no ckpt, 4096 tok)
#   sampler 0.32 -> 30.7 GiB       vs 8.8 weights + ~18 KV (1.1M tokens, TP=2
#                                     shards KV; worst case 256x4096 = 1.05M)
#   free    0.18 -> ~17 GiB        XLA command buffers (outside the BFC pool,
#                                     exit 134 if exhausted) + vLLM workspace
export TRAINER_MEM_FRACTION=0.50
export SAMPLER_MEM_UTILIZATION=0.32

# uv + managed python + caches, all on the volume disk.
export UV_INSTALL_DIR=/workspace/bin
export UV_CACHE_DIR=/workspace/cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/uv-python
export UV_LINK_MODE=copy
export PATH="/workspace/bin:$PATH"

export CUDA_HOME=/usr/local/cuda
export HF_HUB_ENABLE_HF_TRANSFER=1
# FlashInfer JIT-compiles kernels with nvcc at first use; keep the output on
# the volume so they compile once ever, not once per container.
export FLASHINFER_CACHE_DIR=/workspace/cache/flashinfer
export FLASHINFER_WORKSPACE_BASE=/workspace/cache/flashinfer

# Tuned 2026-08-16 on this hardware (bench logs: /workspace/bench/, sweep
# summary in tools/runpod/README.md). Deltas from the old Modal-derived config:
#   gradient_checkpointing false  +29% -- the 33 GiB no-ckpt transient only
#                                 fits on 96 GB cards (halved per GPU at TP=2)
#   loss_chunk_size 0             +9% at 1 slot (runbook retune, reconfirmed)
#   linear_attention_chunk_size 128  +2% and -1 GiB peak (runbook retune)
# Rejected on measurement: SKYRL_CUDNN_MAX_HEAD_DIM=256 (cuDNN rejects
# head_dim 256 on sm_120 exactly as on Ada -- Pallas fallback stays).
BACKEND_CONFIG=$(cat <<EOF
{
  "max_lora_adapters": ${MAX_LORA_ADAPTERS},
  "max_lora_rank": ${MAX_LORA_RANK},
  "tensor_parallel_size": ${TRAINER_TP},
  "fully_sharded_data_parallel_size": 1,
  "train_micro_batch_size": 1,
  "gradient_checkpointing": false,
  "loss_chunk_size": 0,
  "linear_attention_chunk_size": 128,
  "train_bucket_seq_len_per_micro_batch": true
}
EOF
)
export BACKEND_CONFIG
