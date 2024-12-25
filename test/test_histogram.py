from src.genspn.histogram import break_cost, histogram
import jax.numpy as jnp


def test_break_cost():
    breaks = jnp.array([0, 10, 50, 50])
    break_idxs = jnp.array([0, 1, 5, 5])
    new_break = jnp.array(20)
    new_break_idx = jnp.array(4)

    cost = break_cost(breaks, break_idxs, new_break, new_break_idx)

    new_breaks = jnp.array([0, 10, 20, 50])
    new_break_idxs = jnp.array([0, 1, 4, 5])

    T0 = 10
    T1 = 10
    T2 = 30

    N0 = 2
    N1 = 3
    N2 = 1

    true_cost = N0 * (jnp.log(N0) - jnp.log(T0)) + \
        N1 * (jnp.log(N1) - jnp.log(T1)) + \
        N2 * (jnp.log(N2) - jnp.log(T2))

    assert cost[0] == true_cost

def test_histogram():
    data = jnp.array([0, 0.1, 0.2, 10, 10.1, 10.2])
    breaks, break_idxs = histogram(data, max_bins=3)

    assert jnp.all(breaks == jnp.array([0, 0.2, 10, 10.2]))
    assert jnp.all(break_idxs == jnp.array([0, 2, 3, 5]))
