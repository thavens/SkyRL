import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from skyrl.tx.layers.util import (
    _loop_beats_densifying,
    _ragged_dot_grouped_grad,
    ragged_dot,
)


@pytest.mark.parametrize(
    "group_sizes,group_offset,g_local,expected_scale",
    [
        ([2, 2, 2], 1, 2, [0, 0, 1, 1, 2, 2]),  # middle shard
        ([2, 2, 2], 0, 2, [1, 1, 2, 2, 0, 0]),  # first shard
        ([2, 2, 2], 2, 1, [0, 0, 0, 0, 1, 1]),  # last shard
        ([6], 0, 1, [1, 1, 1, 1, 1, 1]),  # single group
        ([2, 0, 0, 4], 1, 2, [0, 0, 0, 0, 0, 0]),  # empty groups in shard
        ([1, 3, 2], 1, 2, [0, 1, 1, 1, 2, 2]),  # uneven sizes
    ],
)
def test_ragged_dot_with_group_offset(group_sizes, group_offset, g_local, expected_scale):
    """Test ragged_dot with group_offset for various edge cases."""
    group_sizes = jnp.array(group_sizes)
    m, d = 6, 2

    lhs = jnp.arange(m * d, dtype=jnp.float32).reshape(m, d)
    rhs = jnp.stack([(i + 1) * jnp.eye(d) for i in range(g_local)])  # 1*I, 2*I, ...

    result = jax.jit(ragged_dot)(lhs, rhs, group_sizes, group_offset=jnp.array([group_offset]))

    # expected_scale: 0 for masked tokens, else local_group_idx + 1
    scale = jnp.array(expected_scale, dtype=jnp.float32)[:, None]
    expected = lhs * scale

    assert jnp.allclose(result, expected), f"Got:\n{result}\nExpected:\n{expected}"


@pytest.mark.parametrize(
    "group_sizes",
    [
        [4, 5, 3, 2, 1],  # every group populated
        [9, 0, 6, 0, 0],  # empty groups -- the common LoRA case (one adapter per batch)
        [15, 0, 0, 0, 0],  # single populated group
    ],
)
def test_grouped_grad_matches_ragged_dot(group_sizes):
    """The memory-efficient VJP must agree with lax.ragged_dot's own gradients.

    _ragged_dot_grouped_grad replaces XLA's transpose with a hand-written one
    (_ragged_contract), so forward *and* both cotangents have to be checked -- a wrong
    weight gradient here silently mistrains every LoRA adapter.
    """
    gs = jnp.array(group_sizes, dtype=jnp.int32)
    m, k, n, g = int(gs.sum()), 4, 3, len(group_sizes)

    rng = np.random.default_rng(0)
    lhs = jnp.asarray(rng.normal(size=(m, k)), jnp.float32)
    rhs = jnp.asarray(rng.normal(size=(g, k, n)), jnp.float32)

    def ref(a, b):
        return lax.ragged_dot(a, b, gs).sum()

    def ours(a, b):
        return _ragged_dot_grouped_grad(a, b, gs).sum()

    assert jnp.allclose(lax.ragged_dot(lhs, rhs, gs), _ragged_dot_grouped_grad(lhs, rhs, gs), atol=1e-5)

    ref_dl, ref_dr = jax.grad(ref, argnums=(0, 1))(lhs, rhs)
    our_dl, our_dr = jax.grad(ours, argnums=(0, 1))(lhs, rhs)
    assert jnp.allclose(ref_dl, our_dl, atol=1e-5), "lhs cotangent diverged"
    assert jnp.allclose(ref_dr, our_dr, atol=1e-5), "rhs (weight) cotangent diverged"

    # Empty groups must contribute exactly zero, not near-zero: _ragged_contract skips
    # them entirely rather than multiplying by a zero mask.
    for i, size in enumerate(group_sizes):
        if size == 0:
            assert jnp.all(our_dr[i] == 0), f"empty group {i} got a non-zero weight gradient"


def test_loop_path_only_when_densifying_would_be_large():
    """The while-loop is gated on actually saving memory.

    Each _ragged_contract emits an HLO while-loop, and a full LoRA backward emits one per
    projection per layer. At small shapes that costs more than the densification it avoids
    (and has crashed the XLA compiler), so those must route to stock ragged_dot.
    """
    small_lhs, small_rhs = jnp.zeros((18, 1024), jnp.bfloat16), jnp.zeros((5, 1024, 32), jnp.bfloat16)
    big_lhs, big_rhs = jnp.zeros((8192, 4096), jnp.bfloat16), jnp.zeros((32, 4096, 32), jnp.bfloat16)

    assert not _loop_beats_densifying(small_lhs, small_rhs)
    assert _loop_beats_densifying(big_lhs, big_rhs)


def test_gated_paths_agree():
    """Whichever path the gate picks, the answer is the same."""
    gs = jnp.array([4, 5, 3, 2, 1], dtype=jnp.int32)
    rng = np.random.default_rng(1)
    lhs = jnp.asarray(rng.normal(size=(15, 4)), jnp.float32)
    rhs = jnp.asarray(rng.normal(size=(5, 4, 3)), jnp.float32)

    gated = jax.jit(ragged_dot)(lhs, rhs, gs)
    assert jnp.allclose(gated, lax.ragged_dot(lhs, rhs, gs), atol=1e-5)
