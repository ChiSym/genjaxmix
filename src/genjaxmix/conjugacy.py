import jax.numpy as jnp
import jax
from .dpmm import K, L_num

def posterior_normal_inverse_gamma(assignments:jax.Array, x: jax.Array, mu_0:float=0.0, v_0:float=1.0, a_0:float=1.0, b_0:float=1.0):
    """Compute the posterior parameters of a normal-inverse-gamma distribution given the assignments and data.

    See Section 6 of https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf by Kevin P. Murphy.

    Args:
        assignments: an array of integers in [0, K) representing the cluster assignments
        x: an array of floats representing the data
        mu_0: the prior mean
        v_0: the prior precision
        a_0: the prior shape
        b_0: the
    """
    counts = jnp.bincount(assignments, length=K)
    sum_x = jax.ops.segment_sum(x, assignments, K)
    sum_x_sq = jax.ops.segment_sum(x**2, assignments, K)

    v_n_inv = 1/v_0 + counts
    m = (1/v_0 * mu_0 + sum_x) / v_n_inv
    a = a_0 + counts / 2
    b = b_0 + 0.5 * (sum_x_sq + 1/v_0*mu_0**2 - v_n_inv * m ** 2)
    return m, 1/v_n_inv, a, b

def posterior_dirichlet(assignments:jax.Array, x:jax.Array):
    """Computes the posterior parameters of a Dirichlet distribution for a multinomial likelihood.

    Args:
        assignments: an array of integers in [0, K) representing the cluster assignments
        x: an array of integers in [0, L_num) representing the data
    """
    one_hot_c = jax.nn.one_hot(assignments, K)
    one_hot_y = jax.nn.one_hot(x, L_num)
    frequency_matrix = one_hot_c.T @ one_hot_y
    return frequency_matrix