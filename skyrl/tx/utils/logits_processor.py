"""Mixin for logits computation in causal language models."""

from abc import ABC, abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp

from skyrl.tx.models.configs import ModelConfig

# lm_head: (hidden_states, adapter_indices) -> logits
LMHead = Callable[[jax.Array, jax.Array | None], jax.Array]


class LogitsProcessorMixin(ABC):
    """Mixin providing logits/logprobs computation for causal language models."""

    @abstractmethod
    def get_model_config(self) -> ModelConfig:
        """Return the model configuration."""
        ...

    @abstractmethod
    def get_lm_head(self) -> LMHead:
        """Return the lm_head callable for logits computation."""
        ...

    def compute_logits(
        self,
        hidden_states: jax.Array,
        adapter_indices: jax.Array | None = None,
    ) -> jax.Array:
        """Compute logits from hidden states. For sampling.

        Args:
            hidden_states: Hidden states from model forward [B, T, H].
            adapter_indices: Optional adapter indices for LoRA.

        Returns:
            Logits [B, T, V].
        """
        return self.get_lm_head()(hidden_states, adapter_indices)

    def compute_logprobs(
        self,
        hidden_states: jax.Array,
        target_ids: jax.Array,
        adapter_indices: jax.Array | None = None,
    ) -> jax.Array:
        """Compute logprobs from hidden states. For training and prompt logprobs.

        Args:
            hidden_states: Hidden states [B, T, H].
            target_ids: Target token IDs [B, T].
            adapter_indices: Adapter indices for LoRA on lm_head.

        Returns:
            Log probabilities for target tokens [B, T].
        """
        chunk_size = self.get_model_config().loss_chunk_size
        if chunk_size > 0:
            return self._compute_chunked_logprobs(hidden_states, target_ids, chunk_size, adapter_indices)
        else:
            logits = self.compute_logits(hidden_states, adapter_indices)
            return self.logits_to_logprobs(logits, target_ids)

    @staticmethod
    def logits_to_logprobs(logits: jax.Array, target_ids: jax.Array) -> jax.Array:
        """Convert logits to logprobs. For decode logprobs when logits already computed.

        Args:
            logits: Logits [B, T, V] or [B, V].
            target_ids: Target token IDs [B, T] or [B].

        Returns:
            Log probabilities for target tokens [B, T] or [B], in float32.
        """
        # The normalizer is a reduction over the whole vocabulary -- 248k entries for
        # Qwen3.5. Accumulating that in bf16 costs ~0.028 nats of error per token
        # (p99 ~0.081), which is the dominant term in the gap between these logprobs and
        # a vLLM sampler's, and on its own enough to push tokens outside a 0.2 PPO clip
        # band. In fp32 the same reduction costs ~0.002 nats. The lm_head output stays
        # bf16; only the reduction is widened.
        #
        # The upcast is applied to the logsumexp input rather than to `logits` as a whole
        # so XLA fuses the convert into the reduction: with loss_chunk_size=0 the full
        # [B, T, V] logits are ~1 GiB at training shapes and an fp32 copy of them would
        # not fit. The target gather stays on the bf16 tensor (a gather is exact) and is
        # widened afterwards, for the same reason.
        log_sum_exp = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1, keepdims=True)
        target_logits = jnp.take_along_axis(logits, target_ids[..., None], axis=-1).astype(jnp.float32)
        return (target_logits - log_sum_exp).squeeze(-1)

    def _compute_chunked_logprobs(
        self,
        hidden_states: jax.Array,
        target_ids: jax.Array,
        chunk_size: int,
        adapter_indices: jax.Array | None,
    ) -> jax.Array:
        """Compute log probabilities using chunked lm_head computation.

        This avoids materializing the full [B*T, V] logits tensor by computing
        lm_head and log probabilities for each chunk sequentially.
        """
        B, T, H = hidden_states.shape
        total_tokens = B * T
        lm_head = self.get_lm_head()

        # Flatten batch and sequence dimensions
        flat_hidden = hidden_states.reshape(-1, H)  # [B*T, H]
        flat_target_ids = target_ids.reshape(-1)  # [B*T]

        # Flatten and chunk adapter indices like hidden states and targets
        if adapter_indices is None:
            flat_adapter_indices = jnp.zeros(total_tokens, dtype=jnp.int32)
        else:
            flat_adapter_indices = jnp.repeat(adapter_indices, T)  # [B*T]

        # Pad to multiple of chunk_size for clean slicing
        num_chunks = (total_tokens + chunk_size - 1) // chunk_size
        pad_amount = num_chunks * chunk_size - total_tokens
        flat_hidden = jnp.pad(flat_hidden, ((0, pad_amount), (0, 0)))
        flat_target_ids = jnp.pad(flat_target_ids, (0, pad_amount))
        flat_adapter_indices = jnp.pad(flat_adapter_indices, (0, pad_amount))

        # Reshape into chunks: [num_chunks, chunk_size, ...]
        chunked_hidden = flat_hidden.reshape(num_chunks, chunk_size, H)
        chunked_targets = flat_target_ids.reshape(num_chunks, chunk_size)
        chunked_adapters = flat_adapter_indices.reshape(num_chunks, chunk_size)

        def compute_chunk_logprobs(args):
            """Compute lm_head and log probabilities for a chunk of tokens."""
            chunk_hidden, chunk_targets, chunk_adapters = args
            chunk_logits = lm_head(chunk_hidden, chunk_adapters)
            return LogitsProcessorMixin.logits_to_logprobs(chunk_logits, chunk_targets)

        if self.get_model_config().gradient_checkpointing:
            compute_chunk_logprobs = jax.checkpoint(compute_chunk_logprobs, policy=None)

        all_logprobs = jax.lax.map(compute_chunk_logprobs, (chunked_hidden, chunked_targets, chunked_adapters))
        return all_logprobs.reshape(-1)[:total_tokens].reshape(B, T)
