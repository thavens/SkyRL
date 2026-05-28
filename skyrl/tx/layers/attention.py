"""Shared attention utilities for transformer models."""

import os

import jax
import jax.numpy as jnp

# cuDNN flash attention supported dtypes/head dimensions
# https://github.com/jax-ml/jax/blob/8b1f782540f71fbe230a2dccd331975faafc6c83/jax/_src/cudnn/fused_attention_stablehlo.py#L290
_CUDNN_SUPPORTED_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2)
# cuDNN's head-dim cap is hardware dependent: 256 on Hopper (sm_90)+, 128 on
# Ampere/Ada. Set SKYRL_CUDNN_MAX_HEAD_DIM=256 on Hopper+ so head_dim=256 models
# (e.g. Qwen3.5) use cuDNN instead of the Pallas/XLA fallback. See process4.md §9.
_CUDNN_MAX_HEAD_DIM = int(os.environ.get("SKYRL_CUDNN_MAX_HEAD_DIM", "128"))

# Pallas/Triton flash attention (jax.experimental) is used for head dims that
# cuDNN rejects (e.g. Qwen3.5's head_dim=256). Importing the GPU ops pulls in
# Triton, so guard it for CPU/TPU-only environments.
try:
    from jax.experimental.pallas.ops.gpu import attention as _pl_gpu_attn
except Exception:  # pragma: no cover - import only succeeds with the GPU/Triton stack
    _pl_gpu_attn = None

_PALLAS_SUPPORTED_DTYPES = (jnp.float16, jnp.bfloat16)
# Small blocks keep the kernel's shared-memory footprint within the ~100KB/SM
# budget on Ada (sm_89) at head_dim=256; the kernel defaults (128 fwd / 64 bwd)
# overflow and hard-crash. Backward blocks must be set explicitly or the kernel
# raises "Backward block sizes must all be set."
_PALLAS_BLOCK = 64
_PALLAS_BLOCK_BWD = 32


def _pallas_flash_attention(q: jax.Array, k: jax.Array, v: jax.Array, scale: float) -> jax.Array:
    """Causal flash attention via the Pallas/Triton GPU kernel.

    The kernel only supports MHA, so GQA inputs are expanded by repeating the
    key/value heads. Sequences are right-padded to a multiple of the block size
    (required by the kernel); under causal masking the padding does not affect
    the outputs at valid positions.
    """
    _, q_len, num_heads, _ = q.shape
    num_kv_heads = k.shape[2]
    if num_heads != num_kv_heads:
        n_rep = num_heads // num_kv_heads
        k = jnp.repeat(k, n_rep, axis=2)
        v = jnp.repeat(v, n_rep, axis=2)

    pad = (-q_len) % _PALLAS_BLOCK
    if pad:
        pad_width = ((0, 0), (0, pad), (0, 0), (0, 0))
        q = jnp.pad(q, pad_width)
        k = jnp.pad(k, pad_width)
        v = jnp.pad(v, pad_width)

    out = _pl_gpu_attn.mha(
        q,
        k,
        v,
        segment_ids=None,
        sm_scale=scale,
        causal=True,
        block_sizes=_pl_gpu_attn.BlockSizes(
            block_q=_PALLAS_BLOCK,
            block_k=_PALLAS_BLOCK,
            block_q_dkv=_PALLAS_BLOCK_BWD,
            block_kv_dkv=_PALLAS_BLOCK_BWD,
            block_q_dq=_PALLAS_BLOCK_BWD,
            block_kv_dq=_PALLAS_BLOCK_BWD,
        ),
        num_stages=1,
    )
    return out[:, :q_len]


def dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_mask: jax.Array,
    is_causal: bool,
    head_dim: int,
) -> jax.Array:
    """Compute dot-product attention with automatic backend selection.

    Uses cuDNN on GPU for memory-efficient attention. Falls back to XLA for CPU/TPU.

    Args:
        q: Query tensor of shape [batch, q_len, num_heads, head_dim]
        k: Key tensor of shape [batch, kv_len, num_kv_heads, head_dim]
        v: Value tensor of shape [batch, kv_len, num_kv_heads, head_dim]
        attention_mask: Mask of shape [batch, kv_len] where 1 = valid, 0 = masked.
            Sequences must be right-padded (valid tokens first, then padding).
        is_causal: Whether to apply causal masking (for prefill/training)
        head_dim: Dimension of each attention head (for scaling)

    Returns:
        Attention output of shape [batch, q_len, num_heads, head_dim]
    """
    scale = 1.0 / head_dim**0.5

    is_gpu = jax.default_backend() == "gpu"
    cudnn_supported_head_dim = head_dim <= _CUDNN_MAX_HEAD_DIM and head_dim % 8 == 0
    if is_gpu and q.dtype in _CUDNN_SUPPORTED_DTYPES and cudnn_supported_head_dim:
        kv_seq_lengths = attention_mask.sum(axis=1).astype(jnp.int32)
        q_seq_lengths = jnp.minimum(kv_seq_lengths, q.shape[1])
        return jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            is_causal=is_causal,
            query_seq_lengths=q_seq_lengths,
            key_value_seq_lengths=kv_seq_lengths,
            implementation="cudnn",
        )

    # Pallas flash attention for the causal prefill/training path when head_dim
    # exceeds cuDNN's cap (e.g. Qwen3.5's 256). The non-causal decode path keeps
    # the XLA fallback below, which handles explicit kv-padding masking.
    if is_gpu and is_causal and q.dtype in _PALLAS_SUPPORTED_DTYPES and not cudnn_supported_head_dim and _pl_gpu_attn is not None:
        return _pallas_flash_attention(q, k, v, scale)

    # CPU/TPU fallback, and GPU fallback for models whose head size is not
    # accepted by cuDNN flash attention.
    return jax.nn.dot_product_attention(
        q,
        k,
        v,
        scale=scale,
        mask=attention_mask[:, None, None, :].astype(bool),
        is_causal=is_causal,
        implementation="xla",
    )
