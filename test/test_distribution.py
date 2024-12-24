from genspn.distributions import (Normal, Categorical, Dirichlet, NormalInverseGamma, Mixed, 
    logpdf, posterior, sample, PiecewiseUniform, DirichletPiecewiseUniform)
import jax.numpy as jnp
import jax
from astropy.stats import bayesian_blocks
from scipy import stats
import numpy as np

def test_continuous_uniform():
    jax.config.update("jax_traceback_filtering", "off")
    N = 5000
    key = jax.random.PRNGKey(1234)
    x = jax.random.gamma(key, 2, shape=(N,))
    y = jax.random.normal(key, shape=(N,))

    breaks1 = bayesian_blocks(x, fitness='events')
    breaks2 = bayesian_blocks(y, fitness='events')

    max_length = max(len(breaks1), len(breaks2))

    breaks = -10000 * np.ones((2, max_length))
    breaks[0, :len(breaks1)] = breaks1
    breaks[1, :len(breaks2)] = breaks2
    breaks = jnp.array(breaks)

    xy = jnp.vstack((x, y))

    greater_than_x = xy[:, :, None] >= breaks[:, None, :-1]
    less_than_x = xy[:, :, None] <= breaks[:, None, 1:]

    in_interval = greater_than_x & less_than_x

    total_intervals = jnp.sum(in_interval, axis=-1)

    assert jnp.all(total_intervals == 1)
    probs = jnp.sum(in_interval, axis=-2) / xy.shape[-1]

    dist = PiecewiseUniform(breaks=breaks, logweights=jnp.log(probs))

    keys = jax.random.split(key, 1000)
    vals = jax.vmap(sample, in_axes=(0, None))(keys, dist)

    means = jnp.mean(vals, axis=0)
    xy_means = jnp.mean(xy, axis=-1)

    assert np.abs(means[0] - xy_means[0]) < 5e-2
    assert np.abs(means[1] - xy_means[1]) < 5e-2


    logpdfs = logpdf(dist, xy[:, 0])

    idx1 = np.where(breaks1 >= xy[0, 0])[0][0] - 1
    logpdf1 = dist.logweights[0][idx1] - np.log(breaks1[idx1+1] - breaks1[idx1])

    idx2 = np.where(breaks2 >= xy[1, 0])[0][0] - 1
    logpdf2 = dist.logweights[1][idx2] - np.log(breaks2[idx2+1] - breaks2[idx2])

    assert jnp.isclose(logpdfs, logpdf1 + logpdf2)

    prior = DirichletPiecewiseUniform(breaks=breaks, alpha=jnp.ones_like(dist.logweights))
    c = jnp.zeros(N, dtype=jnp.int32)
    piecewise_posterior = posterior(prior, xy.T, c)

    counts = jnp.sum(in_interval, axis=-2)

    assert jnp.all(piecewise_posterior.alpha == counts + 1)

def test_posterior_dirichlet():
    n_dim = 2
    k = 3
    N = 4
    alphas = jnp.array([[1.0, 2.0, 3.0], [3.0, 4.0, -jnp.inf]])
    dirichlet = Dirichlet(alphas)

    # x = jnp.array([[0, 1], [1, 0], [2, 0], [0, 0]])
    counts = jnp.array([
        [2, 1, 1],
        [3, 1, 0]
    ])
    # 0, 0, 1, 2 for the first dim
    # 0, 0, 0, 1 for the second dim
    dirichlet_posterior = posterior(dirichlet, counts)
    assert jnp.all(dirichlet_posterior.alpha[0] == jnp.array([1 + 2, 2 + 1, 3 + 1]))
    assert jnp.all(dirichlet_posterior.alpha[1] == jnp.array([3 + 3, 4 + 1, -jnp.inf]))


def test_posterior_nig():
    # adapted from cgpm https://github.com/probcomp/cgpm/blob/master/tests/test_teh_murphy.py
    n = jnp.array(100)
    n_dim = 4
    key = jax.random.PRNGKey(1234)
    x = jax.random.normal(key, shape=(n, n_dim))
    sum_x = jnp.sum(x, axis=0)
    sum_x_sq = jnp.sum(x**2, axis=0)

    all_m = jnp.array((1., 7., .43, 1.2))
    all_l = jnp.array((2., 18., 3., 11.))
    all_a = jnp.array((2., .3, 7., 4.))
    all_b = jnp.array((1., 3., 7.5, 22.5))

    h = NormalInverseGamma(all_m, all_l, all_a, all_b)
    h_prime = posterior(h, n, sum_x, sum_x_sq)

    def check_posterior(x, mu, l, a, b):
        xbar = jnp.mean(x)
        ln = l + n
        an = a + n/2.
        mun = (l*mu+n*xbar)/(l+n)
        bn = b + .5*jnp.sum((x-xbar)**2) + l*n*(xbar-mu)**2 / (2*(l+n))
        return mun, ln, an, bn

    mun, ln, an, bn = jax.vmap(check_posterior, in_axes=(1, 0, 0, 0, 0))(x, all_m, all_l, all_a, all_b)

    assert jnp.allclose(mun, h_prime.m)
    assert jnp.allclose(ln, h_prime.l)
    assert jnp.allclose(an, h_prime.a)
    assert jnp.allclose(bn, h_prime.b)

def test_posterior_nig_cluster():
    # adapted from cgpm https://github.com/probcomp/cgpm/blob/master/tests/test_teh_murphy.py
    n = jnp.array(100)
    n_dim = 4
    key = jax.random.PRNGKey(1234)
    x = jax.random.normal(key, shape=(n, n_dim))

    all_m = jnp.array((1., 7., .43, 1.2))
    all_l = jnp.array((2., 18., 3., 11.))
    all_a = jnp.array((2., .3, 7., 4.))
    all_b = jnp.array((1., 3., 7.5, 22.5))
    c = jnp.repeat(jnp.array([0, 1, 2, 3]), n)

    h = NormalInverseGamma(all_m, all_l, all_a, all_b)
    h_prime = jax.vmap(posterior, in_axes=(0, None, None))(h, x.T.reshape(-1, 1), c)

    def check_posterior(x, mu, l, a, b):
        xbar = jnp.mean(x)
        ln = l + n
        an = a + n/2.
        mun = (l*mu+n*xbar)/(l+n)
        bn = b + .5*jnp.sum((x-xbar)**2) + l*n*(xbar-mu)**2 / (2*(l+n))
        return mun, ln, an, bn

    mun, ln, an, bn = jax.vmap(check_posterior, in_axes=(1, 0, 0, 0, 0))(x, all_m, all_l, all_a, all_b)

    idxs = ((0, 1, 2, 3), (0, 1, 2, 3))
    assert jnp.allclose(mun, h_prime.m[idxs].ravel())
    assert jnp.allclose(ln, h_prime.l[idxs].ravel())
    assert jnp.allclose(an, h_prime.a[idxs].ravel())
    assert jnp.allclose(bn, h_prime.b[idxs].ravel())

def test_posterior_nig_bimodal():
    n = 10000
    n_dim = 2
    max_clusters = 10
    key = jax.random.PRNGKey(1234)
    keys = jax.random.split(key, 6)
    n_data0 = jax.random.normal(keys[0], (n, n_dim)) * 1e-5
    n_data1 = 1 + 1e-5 * jax.random.normal(keys[1], (n, n_dim))
    data = jnp.concatenate((n_data0, n_data1))

    c = jnp.tile(jnp.array([0, max_clusters]), n)
    nig = NormalInverseGamma(m=jnp.zeros(2), l=jnp.ones(2), a=jnp.ones(2), b=jnp.ones(2))
    h_prime = posterior(nig, data, c, 2*max_clusters)
    theta = sample(keys[2], h_prime)

    assert jnp.allclose(theta.std[0], .5, atol=1e-2)
    assert jnp.allclose(theta.std[10], .5, atol=1e-2)
    assert jnp.allclose(theta.mu[0], .5, atol=5e-2)
    assert jnp.allclose(theta.mu[10], .5, atol=5e-2)

def test_normal():
    mu = jnp.array([0.0, 1.0])
    std = jnp.array([1.0, 2.0])

    params = Normal(mu=mu, std=std)

    logp = logpdf(params, jnp.array([0.0, 0.0]))

    logp0 = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(1.0) - 0.5 * ((0.0 - 0.0) / 1.0) ** 2
    logp1 = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(2.0) - 0.5 * ((0.0 - 1.0) / 2.0) ** 2
    assert logp == logp0 + logp1

def test_categorical():
    probs = jnp.array([[0.4, 0.6, 0.0], [0.2, 0.3, 0.5]])
    logprobs = jnp.log(probs)

    params = Categorical(logprobs=logprobs)

    logp = logpdf(params, jnp.array([0, 1]))

    logp0 = jnp.log(0.4)
    logp1 = jnp.log(0.3)

    assert logp == logp0 + logp1

def test_mixed():
    mu = jnp.array([0.0, 1.0])
    std = jnp.array([1.0, 2.0])

    normal_params = Normal(mu=mu, std=std)

    probs = jnp.array([[0.4, 0.6, 0.0], [0.2, 0.3, 0.5]])
    logprobs = jnp.log(probs)

    categorical_params = Categorical(logprobs=logprobs)

    mixed_params = Mixed(dists=(normal_params, categorical_params,))

    logp = logpdf(mixed_params, (jnp.array([0.0, 0.0]), jnp.array([0, 1])))

    logp_n0 = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(1.0) - 0.5 * ((0.0 - 0.0) / 1.0) ** 2
    logp_n1 = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(2.0) - 0.5 * ((0.0 - 1.0) / 2.0) ** 2
    
    logp_c0 = jnp.log(0.4)
    logp_c1 = jnp.log(0.3) 

    assert logp == logp_n0 + logp_n1 + logp_c0 + logp_c1
