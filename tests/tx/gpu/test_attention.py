"""GPU tests for flash attention.

These tests require a GPU and verify that cuDNN flash attention produces
numerically equivalent results to the mask-based implementation.
"""

import jax
import jax.numpy as jnp
import pytest

from skyrl.tx.layers.attention import dot_product_attention

# Skip all tests if not on GPU
pytestmark = pytest.mark.skipif(jax.default_backend() != "gpu", reason="GPU tests require CUDA")


def make_qkv(batch, seq_len, num_heads, head_dim, num_kv_heads=None, dtype=jnp.bfloat16):
    """Create random Q, K, V tensors."""
    if num_kv_heads is None:
        num_kv_heads = num_heads
    q = jax.random.normal(jax.random.key(0), (batch, seq_len, num_heads, head_dim), dtype=dtype)
    k = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim), dtype=dtype)
    v = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim), dtype=dtype)
    return q, k, v


def make_right_padded_mask(batch, seq_len, seq_lengths):
    """Create right-padded mask: [1,1,1,...,0,0]."""
    seq_lengths = jnp.array(seq_lengths)
    return (jnp.arange(seq_len)[None, :] < seq_lengths[:, None]).astype(jnp.float32)


def assert_attention_match(q, k, v, mask, is_causal, head_dim, seq_lengths=None):
    """Run both attention implementations and assert they match.

    Args:
        seq_lengths: If provided, only compare valid positions per batch element.
                    If None, compare all positions.
    """
    scale = 1.0 / head_dim**0.5
    result = dot_product_attention(q, k, v, mask, is_causal=is_causal, head_dim=head_dim)
    expected = jax.nn.dot_product_attention(
        q, k, v, scale=scale, mask=mask[:, None, None, :].astype(bool), is_causal=is_causal
    )

    # bfloat16 has ~7 bits of mantissa (epsilon ≈ 2^-7 = 0.0078)
    # Attention chains multiple ops, so errors compound to ~2^-6 = 0.0156
    atol = 0.02

    if seq_lengths is None:
        assert jnp.allclose(result, expected, atol=atol)
    else:
        for b, length in enumerate(seq_lengths):
            assert jnp.allclose(result[b, :length], expected[b, :length], atol=atol), f"Mismatch at batch {b}"


class TestFlashAttention:
    """Verify cuDNN flash attention matches mask-based attention."""

    @pytest.mark.parametrize("seq_len", [32, 128, 512])
    def test_padded_equivalence(self, seq_len):
        """cuDNN matches mask-based for right-padded sequences."""
        batch, num_heads, head_dim = 2, 4, 64
        q, k, v = make_qkv(batch, seq_len, num_heads, head_dim)
        seq_lengths = [seq_len - 4, seq_len - 8]
        mask = make_right_padded_mask(batch, seq_len, seq_lengths)
        assert_attention_match(q, k, v, mask, is_causal=True, head_dim=head_dim, seq_lengths=seq_lengths)

    def test_no_padding(self):
        """Full sequences (no padding) work correctly."""
        batch, seq_len, num_heads, head_dim = 2, 64, 4, 64
        q, k, v = make_qkv(batch, seq_len, num_heads, head_dim)
        mask = jnp.ones((batch, seq_len))
        assert_attention_match(q, k, v, mask, is_causal=True, head_dim=head_dim)

    @pytest.mark.parametrize(
        "seq_lengths",
        [
            [128, 96, 64, 32],  # decreasing lengths
            [32, 64, 96, 128],  # increasing lengths
            [128, 128, 128, 128],  # all full (no padding)
            [1, 1, 1, 1],  # minimal valid sequences
        ],
    )
    def test_mixed_seq_lengths(self, seq_lengths):
        """Batch with varying sequence lengths."""
        batch, seq_len, num_heads, head_dim = 4, 128, 4, 64
        q, k, v = make_qkv(batch, seq_len, num_heads, head_dim)
        mask = make_right_padded_mask(batch, seq_len, seq_lengths)
        assert_attention_match(q, k, v, mask, is_causal=True, head_dim=head_dim, seq_lengths=seq_lengths)

    def test_decode(self):
        """Decode mode (is_causal=False, single query token)."""
        batch, kv_len, num_heads, head_dim = 2, 128, 4, 64
        q = jax.random.normal(jax.random.key(0), (batch, 1, num_heads, head_dim), dtype=jnp.bfloat16)
        k = jax.random.normal(jax.random.key(1), (batch, kv_len, num_heads, head_dim), dtype=jnp.bfloat16)
        v = jax.random.normal(jax.random.key(2), (batch, kv_len, num_heads, head_dim), dtype=jnp.bfloat16)
        mask = make_right_padded_mask(batch, kv_len, [100, 80])
        assert_attention_match(q, k, v, mask, is_causal=False, head_dim=head_dim)

    def test_float32_fallback(self):
        """float32 (unsupported by cuDNN) uses mask-based fallback."""
        batch, seq_len, num_heads, head_dim = 2, 64, 4, 64
        q, k, v = make_qkv(batch, seq_len, num_heads, head_dim, dtype=jnp.float32)
        mask = jnp.ones((batch, seq_len))
        assert_attention_match(q, k, v, mask, is_causal=True, head_dim=head_dim)

    def test_gqa_decode(self):
        """GQA decode mode (8 Q heads, 2 KV heads, single query token)."""
        batch, kv_len, num_heads, num_kv_heads, head_dim = 2, 128, 8, 2, 64
        q = jax.random.normal(jax.random.key(0), (batch, 1, num_heads, head_dim), dtype=jnp.bfloat16)
        k = jax.random.normal(jax.random.key(1), (batch, kv_len, num_kv_heads, head_dim), dtype=jnp.bfloat16)
        v = jax.random.normal(jax.random.key(2), (batch, kv_len, num_kv_heads, head_dim), dtype=jnp.bfloat16)
        mask = make_right_padded_mask(batch, kv_len, [100, 80])
        assert_attention_match(q, k, v, mask, is_causal=False, head_dim=head_dim)

    def test_gqa_prefill(self):
        """GQA prefill mode with right-padded sequences (8 Q heads, 2 KV heads)."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 128, 8, 2, 64
        q, k, v = make_qkv(batch, seq_len, num_heads, head_dim, num_kv_heads)
        seq_lengths = [100, 80]
        mask = make_right_padded_mask(batch, seq_len, seq_lengths)
        assert_attention_match(q, k, v, mask, is_causal=True, head_dim=head_dim, seq_lengths=seq_lengths)


class TestPallasFlashAttention:
    """Pallas path (head_dim=256, > cuDNN's 128 cap) matches XLA, fwd and bwd."""

    @pytest.mark.parametrize("seq_len", [64, 128, 256, 200, 513])
    def test_gqa_prefill_head_dim_256(self, seq_len):
        """Large-head-dim GQA causal prefill, including non-block-multiple lengths."""
        batch, num_heads, num_kv_heads, head_dim = 2, 8, 2, 256
        q, k, v = make_qkv(batch, seq_len, num_heads, head_dim, num_kv_heads)
        seq_lengths = [seq_len, max(1, seq_len - 17)]
        mask = make_right_padded_mask(batch, seq_len, seq_lengths)
        assert_attention_match(q, k, v, mask, is_causal=True, head_dim=head_dim, seq_lengths=seq_lengths)

    def test_mha_prefill_head_dim_256(self):
        """Large-head-dim MHA (no GQA expansion) causal prefill."""
        batch, seq_len, num_heads, head_dim = 2, 128, 8, 256
        q, k, v = make_qkv(batch, seq_len, num_heads, head_dim)
        seq_lengths = [128, 100]
        mask = make_right_padded_mask(batch, seq_len, seq_lengths)
        assert_attention_match(q, k, v, mask, is_causal=True, head_dim=head_dim, seq_lengths=seq_lengths)

    @pytest.mark.parametrize("seq_len", [128, 256])
    def test_gqa_backward_head_dim_256(self, seq_len):
        """Gradients through the Pallas path match XLA (exercises the backward kernel)."""
        batch, num_heads, num_kv_heads, head_dim = 2, 8, 2, 256
        q, k, v = make_qkv(batch, seq_len, num_heads, head_dim, num_kv_heads)
        mask = jnp.ones((batch, seq_len))

        def loss(fn):
            def _loss(q, k, v):
                return fn(q, k, v, mask, is_causal=True, head_dim=head_dim).astype(jnp.float32).sum()

            return _loss

        ref = lambda q, k, v, m, is_causal, head_dim: jax.nn.dot_product_attention(  # noqa: E731
            q, k, v, scale=1.0 / head_dim**0.5, mask=m[:, None, None, :].astype(bool), is_causal=is_causal
        )
        g_pallas = jax.grad(loss(dot_product_attention), argnums=(0, 1, 2))(q, k, v)
        g_ref = jax.grad(loss(ref), argnums=(0, 1, 2))(q, k, v)
        for name, gp, gr in zip("qkv", g_pallas, g_ref):
            denom = jnp.max(jnp.abs(gr.astype(jnp.float32))) + 1e-6
            rel = jnp.max(jnp.abs(gp.astype(jnp.float32) - gr.astype(jnp.float32))) / denom
            assert rel < 0.05, f"grad_{name} rel error {rel} too high"
