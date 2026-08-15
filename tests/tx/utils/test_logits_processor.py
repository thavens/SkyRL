"""Unit tests for LogitsProcessorMixin chunked logprobs computation."""

import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tx.utils.logits_processor import LogitsProcessorMixin
from tests.tx.utils.test_generator import DummyModel


def assert_chunked_matches_nonchunked(
    hidden_states: jnp.ndarray,
    target_ids: jnp.ndarray,
    chunk_size: int,
    adapter_indices: jnp.ndarray | None = None,
    vocab_size: int = 16,
):
    """Assert chunked and non-chunked paths produce identical results."""
    model_chunked = DummyModel(vocab_size=vocab_size, loss_chunk_size=chunk_size)
    model_nonchunked = DummyModel(vocab_size=vocab_size, loss_chunk_size=0)

    logprobs_chunked = model_chunked.compute_logprobs(hidden_states, target_ids, adapter_indices)
    logprobs_nonchunked = model_nonchunked.compute_logprobs(hidden_states, target_ids, adapter_indices)

    B, T = target_ids.shape
    assert logprobs_chunked.shape == (B, T)
    assert logprobs_nonchunked.shape == (B, T)

    np.testing.assert_allclose(
        np.asarray(logprobs_chunked),
        np.asarray(logprobs_nonchunked),
        rtol=1e-5,
        atol=1e-5,
    )


class TestChunkedLogprobs:
    """Tests for chunked vs non-chunked logprobs computation."""

    @pytest.mark.parametrize(
        "B,T,chunk_size",
        [
            (2, 4, 3),  # chunk doesn't divide evenly, needs padding
            (2, 4, 8),  # chunk equals B*T exactly
            (2, 4, 16),  # chunk larger than B*T
            (1, 8, 3),  # single batch element
            (4, 1, 2),  # single token per sequence
            (1, 1, 1),  # minimal case
        ],
    )
    def test_chunk_boundary_cases(self, B, T, chunk_size):
        """Test various chunk size vs total token relationships."""
        V = 16  # vocab_size = hidden_size for identity lm_head
        hidden_states = jnp.arange(B * T * V, dtype=jnp.float32).reshape(B, T, V) / (B * T * V)
        target_ids = jnp.arange(B * T, dtype=jnp.int32).reshape(B, T) % V

        assert_chunked_matches_nonchunked(hidden_states, target_ids, chunk_size, vocab_size=V)

    @pytest.mark.parametrize(
        "B,T,chunk_size,adapter_indices",
        [
            (2, 4, 3, None),  # no adapters
            (2, 4, 3, "arange"),  # different adapter per batch, chunk spans boundary
            (3, 4, 5, "arange"),  # chunk spans multiple batches
            (4, 2, 3, "zeros"),  # all same adapter
        ],
    )
    def test_adapter_indices_handling(self, B, T, chunk_size, adapter_indices):
        """Test adapter indices are correctly mapped across chunk boundaries."""
        V = 16
        hidden_states = jnp.arange(B * T * V, dtype=jnp.float32).reshape(B, T, V) / (B * T * V)
        target_ids = jnp.arange(B * T, dtype=jnp.int32).reshape(B, T) % V

        if adapter_indices == "arange":
            adapter_indices = jnp.arange(B, dtype=jnp.int32)
        elif adapter_indices == "zeros":
            adapter_indices = jnp.zeros(B, dtype=jnp.int32)

        assert_chunked_matches_nonchunked(hidden_states, target_ids, chunk_size, adapter_indices, vocab_size=V)

    def test_gradient_checkpointing_flag(self):
        """Gradient checkpointing should not affect forward pass results."""
        B, T, V, chunk_size = 2, 4, 16, 3
        hidden_states = jnp.arange(B * T * V, dtype=jnp.float32).reshape(B, T, V) / (B * T * V)
        target_ids = jnp.arange(B * T, dtype=jnp.int32).reshape(B, T) % V

        model_no_ckpt = DummyModel(vocab_size=V, loss_chunk_size=chunk_size)
        model_no_ckpt.config.gradient_checkpointing = False

        model_ckpt = DummyModel(vocab_size=V, loss_chunk_size=chunk_size)
        model_ckpt.config.gradient_checkpointing = True

        logprobs_no_ckpt = model_no_ckpt.compute_logprobs(hidden_states, target_ids)
        logprobs_ckpt = model_ckpt.compute_logprobs(hidden_states, target_ids)

        np.testing.assert_allclose(
            np.asarray(logprobs_no_ckpt),
            np.asarray(logprobs_ckpt),
            rtol=1e-5,
            atol=1e-5,
        )


class TestLogprobNormalizerPrecision:
    """The vocab-sized log-normalizer must not be accumulated in the logits' own dtype.

    Qwen3.5 pairs a bf16 lm_head with a 248k-entry vocabulary. Summing exp() over that
    many bf16 terms loses ~0.03 nats per token (p99 ~0.08), which surfaces directly as
    sampler/trainer logprob mismatch -- and therefore as PPO clipping -- in RL, since the
    vLLM sampler normalizes in fp32. The existing chunked-vs-nonchunked tests run at
    vocab_size=16 in fp32 and cannot see this.
    """

    @staticmethod
    def _reference_logprobs(logits_bf16: jnp.ndarray, targets: np.ndarray) -> np.ndarray:
        """log_softmax of the *same* bf16-rounded logits, accumulated in float64.

        Using the bf16 values as the reference input isolates the precision of the
        reduction from the precision of the logits themselves.
        """
        x = np.asarray(logits_bf16, dtype=np.float64)
        m = x.max(axis=-1, keepdims=True)
        lse = m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True))
        return (np.take_along_axis(x, targets[:, None], axis=-1) - lse).squeeze(-1)

    def test_bf16_logits_are_normalized_in_fp32(self):
        # Large enough that a bf16 accumulator visibly drifts; a few hundred entries would not.
        B, V = 8, 200_000
        rng = np.random.default_rng(0)
        logits = (rng.standard_normal((B, V)) * 2.0).astype(np.float32)
        # A few dominant logits, as in a trained LM head.
        logits[np.arange(B), rng.integers(0, V, B)] += 8.0
        targets = rng.integers(0, V, B)

        logits_bf16 = jnp.asarray(logits, jnp.bfloat16)
        got = np.asarray(
            LogitsProcessorMixin.logits_to_logprobs(logits_bf16, jnp.asarray(targets)),
            dtype=np.float64,
        )

        # Returned in fp32: a bf16 logprob cannot represent the gap it is about to be
        # differenced against on the sampler side.
        assert LogitsProcessorMixin.logits_to_logprobs(logits_bf16, jnp.asarray(targets)).dtype == jnp.float32

        # An fp32 reduction lands within ~1e-4 of the float64 result; a bf16 one is ~1e-2 off.
        np.testing.assert_allclose(got, self._reference_logprobs(logits_bf16, targets), atol=1e-4)
