import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Integer


def histogram(data: Float[Array, "n"], max_bins: int = 100):
    data = jnp.sort(data)

    breaks = jnp.max(data) * jnp.ones(max_bins + 1)
    breaks = breaks.at[0].set(jnp.min(data))

    break_idxs = (len(data) - 1) * jnp.ones(max_bins + 1, dtype=jnp.int32)
    break_idxs = break_idxs.at[0].set(0)

    def step_histogram(break_idxs: Integer[Array, "max_bins"], x):
        breaks = jnp.take(data, break_idxs, indices_are_sorted=True)
        break_costs, new_breaks, new_break_idxs = jax.vmap(break_cost, in_axes=(None, None, 0, 0))(
            breaks, break_idxs, data[1:], jnp.arange(1, len(data)))
        best_break_idx = jnp.nanargmax(break_costs)
        break_idxs = new_break_idxs[best_break_idx]
        return break_idxs, None

    break_idxs, _ = jax.lax.scan(f=step_histogram, init=break_idxs, length=max_bins - 1)
    breaks = jnp.take(data, break_idxs, indices_are_sorted=True)

    return breaks, break_idxs


def break_cost(breaks: Float[Array, "n"], break_idxs: Integer[Array, "n"], 
        new_break: Float[Array, ""], new_break_idx: Integer[Array, ""]):
    insert_idx = jnp.searchsorted(breaks, new_break)

    new_breaks = jnp.insert(breaks, insert_idx, new_break)[:-1]
    new_break_idxs = jnp.insert(break_idxs, insert_idx, new_break_idx)[:-1]

    N = jnp.diff(new_break_idxs)
    T = jnp.diff(new_breaks)
    N = N.at[0].set(N[0] + 1)

    bin_costs = N * (jnp.log(N) - jnp.log(T))

    return jnp.nansum(bin_costs), new_breaks, new_break_idxs