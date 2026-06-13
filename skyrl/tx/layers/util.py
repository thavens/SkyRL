import jax
from flax import nnx
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec, get_abstract_mesh


def _ragged_contract(a: jax.Array, b: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """Grouped contraction over the (sorted, contiguous) token dimension — the LoRA weight gradient.

    ``out[g, p, q] = sum_{t in group g} a[t, p] * b[t, q]``; ``a:[m, p]``, ``b:[m, q]`` -> ``[G, p, q]``.

    The obvious spelling — ``ragged_dot_general`` with the token dim as the *contracting*
    ragged dim — is exactly what XLA's autodiff transpose of ``ragged_dot`` already emits,
    and on GPU it lowers to a ``[G, m, p]`` densification (one masked copy of the activations
    per group/adapter slot) that dominates LoRA fwd/bwd memory. Instead we ``fori_loop`` over
    the (few) groups, masking ``a`` to one group per iteration and reusing a single ``[m, p]``
    buffer. The loop lowers to an HLO while-loop, which XLA does not vectorize back into
    ``[G, m, p]`` — so peak backward memory no longer scales with the adapter count.
    """
    num_groups = group_sizes.shape[0]
    m, p = a.shape
    q = b.shape[1]
    bounds = jnp.cumulative_sum(group_sizes, include_initial=True)
    tok = jnp.arange(m)

    def body(g, acc):
        mask = ((tok >= bounds[g]) & (tok < bounds[g + 1])).astype(a.dtype)
        return acc.at[g].set((a * mask[:, None]).T @ b)

    return lax.fori_loop(0, num_groups, body, jnp.zeros((num_groups, p, q), a.dtype))


@jax.custom_vjp
def _ragged_dot_grouped_grad(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """``lax.ragged_dot`` whose backward keeps the weight gradient ragged.

    Forward is identical to ``lax.ragged_dot``. The difference is the VJP: XLA's default
    transpose of ``ragged_dot`` w.r.t. the grouped ``rhs`` densifies to
    ``[num_groups, num_tokens, features]`` — for LoRA, one masked copy of the activations
    *per adapter slot*, which dominates fwd/bwd memory and makes it scale with
    ``max_lora_adapters``. Here the weight gradient goes through ``_ragged_contract`` instead,
    so backward memory no longer scales with the adapter count; ``d_lhs`` stays a cheap ragged_dot.

    ``group_sizes`` is a traced integer array (bincount of adapter indices), so it must be a
    regular arg (not ``nondiff_argnums``); bwd returns a ``None`` cotangent for it.
    """
    return lax.ragged_dot(lhs, rhs, group_sizes)


def _ragged_dot_grouped_grad_fwd(lhs, rhs, group_sizes):
    return lax.ragged_dot(lhs, rhs, group_sizes), (lhs, rhs, group_sizes)


def _ragged_dot_grouped_grad_bwd(res, g):
    lhs, rhs, group_sizes = res  # lhs:[m, k], rhs:[G, k, n], cotangent g:[m, n]
    d_lhs = lax.ragged_dot(g, jnp.swapaxes(rhs, -1, -2), group_sizes)  # [m, k] (cheap ragged_dot)
    d_rhs = _ragged_contract(lhs, g, group_sizes)  # [G, k, n] (no per-group densification)
    return d_lhs, d_rhs, None  # None cotangent for the integer group_sizes


_ragged_dot_grouped_grad.defvjp(_ragged_dot_grouped_grad_fwd, _ragged_dot_grouped_grad_bwd)


def ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    precision=None,
    preferred_element_type=None,
    group_offset: jax.Array | None = None,
) -> jax.Array:
    """Ragged dot product with group_offset support.

    When group_offset is specified, rhs contains groups [offset, offset + g_local).
    Tokens outside this range are routed to boundary groups and masked to zero.
    """
    if group_offset is None:
        # Memory-efficient backward (no per-group densification) for the common LoRA path.
        # precision / preferred_element_type are unused by the LoRA callers; if ever set,
        # fall back to plain ragged_dot (whose autodiff transpose densifies).
        if precision is None and preferred_element_type is None:
            return _ragged_dot_grouped_grad(lhs, rhs, group_sizes)
        return lax.ragged_dot(
            lhs,
            rhs,
            group_sizes,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

    assert group_offset.shape == (1,), "group_offset must have shape (1,)"
    offset = group_offset[0]
    m = lhs.shape[0]
    g_local = rhs.shape[0]

    assert g_local > 0, "rhs must have at least one group"

    # Compute token boundaries for local groups
    cumsum = jnp.cumulative_sum(group_sizes, include_initial=True)
    shard_start = cumsum[offset]
    shard_end = cumsum[offset + g_local]

    # Valid mask for tokens in local groups
    token_idx = jnp.arange(m)
    valid_mask = (token_idx >= shard_start) & (token_idx < shard_end)

    # Adjust group sizes: absorb extra tokens at boundaries
    local_group_sizes = lax.dynamic_slice_in_dim(group_sizes, offset, g_local, axis=0)
    adjusted_group_sizes = local_group_sizes.at[0].add(shard_start).at[-1].add(m - shard_end)

    # Call ragged_dot - extra tokens use boundary groups but get masked out
    result = lax.ragged_dot(
        lhs,
        rhs,
        adjusted_group_sizes,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    return jnp.where(valid_mask[:, None], result, 0)


def Param(*shape: int, dtype: jnp.dtype, kernel_init: nnx.Initializer, rngs: nnx.Rngs):
    return nnx.Param(kernel_init(rngs.param(), shape, dtype))


def prepare_routing(
    tokens: jax.Array, indices: jax.Array, num_groups: int, adapter_indices: jax.Array | None = None
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Prepare inputs for ragged_dot operations by sorting tokens by group.

    Args:
        tokens: Array of shape (num_tokens, ...) to be sorted by group
        indices: Array of shape (num_tokens,) indicating group assignment for each token
        num_groups: Total number of groups
        adapter_indices: Optional array of shape (num_tokens,) to be sorted together with tokens

    Returns:
        sorted_tokens: Tokens sorted by group index
        group_sizes: Number of tokens in each group
        unsort_indices: Indices to restore original order after ragged operations
    """
    sort_indices = jnp.argsort(indices)
    sorted_tokens = tokens[sort_indices]
    sorted_adapter_indices = None if adapter_indices is None else adapter_indices[sort_indices]
    group_sizes = jnp.bincount(indices, length=num_groups)
    unsort_indices = jnp.argsort(sort_indices)
    return sorted_tokens, group_sizes, unsort_indices, sorted_adapter_indices


def shard_map_ep(module: nnx.Module, func, *args):
    """Apply shard_map over the 'ep' axis for a stateful nnx.Module.

    Args:
        module: The NNX module (will be split into graph/state).
        func: Function to run inside shard_map. Signature: (module, *args).
        *args: Arguments to pass to func (replicated across shards).
    """
    graphdef, state = nnx.split(module)

    def make_ep_spec(spec, value):
        """Create a PartitionSpec with only 'ep' dims, truncated to match tensor rank."""
        if not isinstance(spec, PartitionSpec):
            return spec
        arr = value[...] if isinstance(value, nnx.Variable) else value
        # Stacked-layer extraction (e.g. x[layer_idx]) can drop a leading tensor dim,
        # while PartitionSpec metadata still includes it. Trim from the left to match.
        truncated = spec[-arr.ndim :] if arr.ndim else ()
        # Extract only 'ep' dims from PartitionSpecs, replacing others with None.
        return PartitionSpec(*(p if p == "ep" else None for p in truncated))

    partition_specs = nnx.get_partition_spec(state)
    state_specs = jax.tree.map(make_ep_spec, partition_specs, state, is_leaf=lambda x: isinstance(x, PartitionSpec))
    in_specs = (state_specs,) + (PartitionSpec(),) * len(args)

    @jax.shard_map(mesh=get_abstract_mesh(), in_specs=in_specs, out_specs=PartitionSpec(), axis_names={"ep"})
    def _body(state, *fn_args):
        module_shard = nnx.merge(graphdef, state)
        return func(module_shard, *fn_args)

    return _body(state, *args)
