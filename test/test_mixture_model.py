import jax.numpy as jnp
import jax
import numpy as np
from genspn.smc import q_split
from genspn.distributions import (Normal, Categorical, Dirichlet, NormalInverseGamma, Mixed,
 MixedConjugate, posterior, sample, logpdf, DirichletPiecewiseUniform)
from functools import partial

# def test_categorical_mixture_model():
#     iters = 1000
#     alpha = 1
#     key = jax.random.PRNGKey(1234)
#     keys = jax.random.split(key, 5)

#     keys_c0 = jax.random.split(keys[0], 100)
#     keys_c1 = jax.random.split(keys[1], 100)
#     c_data0 = jax.vmap(jax.random.categorical, in_axes=(0, None))(keys_c0, jnp.log(jnp.array([0.9, 0.1])))
#     c_data1 = jax.vmap(jax.random.categorical, in_axes=(0, None))(keys_c1, jnp.log(jnp.array([0.1, 0.9])))

#     data = np.concatenate([c_data0, c_data1], axis=0)[:, None]
#     c = jnp.zeros(200)
#     g = Dirichlet(alpha=jnp.ones((1, 2)))

#     # step_jit = jax.jit(partial(step, gibbs_iters=iters))
#     trace = q_split(data, gibbs_iters=iters, max_clusters=2, key=keys[5], c0=c, g=g, alpha=alpha)
#     import ipdb; ipdb.set_trace()

#     assert jnp.all(trace.c[-1, :100] == trace.c[-1, 0])
#     assert jnp.all(trace.c[-1, 100:] == trace.c[-1, 100])
#     assert trace.c[-1, 0] != trace.c[-1, 100]

# def test_gaussian_mixture_model():
#     iters = 1000
#     alpha = 1
#     key = jax.random.PRNGKey(1234)
#     keys = jax.random.split(key, 6)
#     n_data0 = jax.random.normal(keys[0], (100, 2)) * .1
#     n_data1 = 10 + .1 * jax.random.normal(keys[1], (100, 2))

#     data = np.concatenate([n_data0, n_data1], axis=0)
#     c = jnp.zeros(200)
#     c = jnp.concatenate((jnp.zeros(100), 2*jnp.ones(100)))
#     g = NormalInverseGamma(m=jnp.zeros(2), l=jnp.ones(2), a=jnp.ones(2), b=jnp.ones(2))

#     # step_jit = jax.jit(partial(step, gibbs_iters=iters))
#     trace = q_split(data, gibbs_iters=iters, max_clusters=2, key=keys[5], c0=c, g=g, alpha=alpha)
#     import ipdb; ipdb.set_trace()

#     assert jnp.all(trace.c[-1, :100] == trace.c[-1, 0])
#     assert jnp.all(trace.c[-1, 100:] == trace.c[-1, 100])
#     assert trace.c[-1, 0] != trace.c[-1, 100]

def test_mixture_model():
    iters = 20
    alpha = 1
    key = jax.random.PRNGKey(1234)
    keys = jax.random.split(key, 6)
    n_data0 = jax.random.normal(keys[0], (100, 2)) * .1
    n_data1 = 1 + .1 * jax.random.normal(keys[1], (100, 2))

    keys_c0 = jax.random.split(keys[2], 100)
    keys_c1 = jax.random.split(keys[3], 100)
    c_data0 = jax.vmap(jax.random.categorical, in_axes=(0, None))(keys_c0, jnp.log(jnp.array([0.4, 0.6])))
    c_data1 = jax.vmap(jax.random.categorical, in_axes=(0, None))(keys_c1, jnp.log(jnp.array([0.4, 0.6])))

    data = np.concatenate([n_data0, n_data1], axis=0), np.concatenate([c_data0, c_data1], axis=0)[:, None]
    c = jnp.zeros(200)
    nig = NormalInverseGamma(m=jnp.zeros(2), l=jnp.ones(2), a=jnp.ones(2), b=jnp.ones(2))
    dirichlet = Dirichlet(alpha=jnp.ones((1, 2)))
    g = MixedConjugate(dists=(nig, dirichlet,))

    # step_jit = jax.jit(partial(step, gibbs_iters=iters))
    trace = q_split(data, gibbs_iters=iters, max_clusters=2, key=keys[5], c0=c, g=g, alpha=alpha)

    assert jnp.all(trace.c[-1, :100] == trace.c[-1, 0])
    assert jnp.all(trace.c[-1, 100:] == trace.c[-1, 100])
    assert trace.c[-1, 0] != trace.c[-1, 100]


def test_mixture_model_piecewise_uniform():
    iters = 20
    alpha = 1
    key = jax.random.PRNGKey(1234)
    keys = jax.random.split(key, 6)
    n_data0 = jax.random.normal(keys[0], (100, 2)) * .1
    n_data1 = 1 + .1 * jax.random.normal(keys[1], (100, 2))

    keys_c0 = jax.random.split(keys[2], 100)
    keys_c1 = jax.random.split(keys[3], 100)
    c_data0 = jax.vmap(jax.random.categorical, in_axes=(0, None))(keys_c0, jnp.log(jnp.array([0.4, 0.6])))
    c_data1 = jax.vmap(jax.random.categorical, in_axes=(0, None))(keys_c1, jnp.log(jnp.array([0.4, 0.6])))

    data = np.concatenate([n_data0, n_data1], axis=0), np.concatenate([c_data0, c_data1], axis=0)[:, None]
    c = jnp.zeros(200)
    nig = NormalInverseGamma(m=jnp.zeros(2), l=jnp.ones(2), a=jnp.ones(2), b=jnp.ones(2))
    breaks0, _ = histogram(data[0][:, 0])
    breaks1, _ = histogram(data[0][:, 1])

    min_val = np.min([np.min(b) for b in [breaks0, breaks1]]) - 1

    n_breaks = jnp.array([len(b) for b in [breaks0, breaks1]])
    max_n_breaks = jnp.max(n_breaks).astype(int)
    breaks_arr = min_val * np.ones((2, max_n_breaks)) 

    breaks_alpha = 1e-3 * jnp.ones((2, max_n_breaks))
    mask = jnp.tile(jnp.arange(max_n_breaks), (2, 1)) < n_breaks[:, None]
    breaks_alpha = jnp.where(mask, breaks_alpha, 1e-20)

    for i, b in enumerate([breaks0, breaks1]):
        breaks_arr[i, :len(b)] = b

    dirichlet_piecewise = DirichletPiecewiseUniform(breaks=breaks_arr, alpha=breaks_alpha)
    dirichlet = Dirichlet(alpha=jnp.ones((1, 2)))
    g = MixedConjugate(dists=(nig, dirichlet,))

    # step_jit = jax.jit(partial(step, gibbs_iters=iters))
    trace = q_split(data, gibbs_iters=iters, max_clusters=2, key=keys[5], c0=c, g=g, alpha=alpha)

    assert jnp.all(trace.c[-1, :100] == trace.c[-1, 0])
    assert jnp.all(trace.c[-1, 100:] == trace.c[-1, 100])
    assert trace.c[-1, 0] != trace.c[-1, 100]
