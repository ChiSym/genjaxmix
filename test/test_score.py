from genspn.distributions import (logpdf, GEM, Normal,
    NormalInverseGamma, Cluster)
from genspn.smc import score_trace_cluster, make_pi
import jax.numpy as jnp
import jax
import pytest

def test_make_pi():
    max_clusters = 3
    pi0 = jnp.array([1/2, 1/3, 0, 0, 0, 0])
    q_pi = jnp.array([1/2, 1/2, 3/4, 1/2, 0, 0])
    new_pi = make_pi(pi0, k=0, pi_split=q_pi, max_clusters=max_clusters)

    assert jnp.allclose(new_pi, jnp.array([1/3, 1/4, 1/4, 0, 0, 0]))

def test_score_stick_breaking():
    alpha = 1
    d = .1
    gem = GEM(alpha, d)
    K = jnp.array(2)
    pis0 = jnp.array([1/3, 1/4, 1/8])
    pis1 = jnp.array([1/2, 1/4, 1/8])

    logp0 = logpdf(gem, pis0, K)
    logp1 = logpdf(gem, pis1, K)

    logp0_expected = jax.scipy.stats.beta.logpdf(pis0[0], 1 - d, alpha + d) + jax.scipy.stats.beta.logpdf(pis0[1]/(1-pis0[0]), 1 - d, alpha + 2 * d)
    assert logp0 == logp0_expected 

    logp1_expected = jax.scipy.stats.beta.logpdf(pis1[0], 1 - d, alpha + d) + jax.scipy.stats.beta.logpdf(pis1[1]/(1-pis1[0]), 1 - d, alpha + 2 * d)

    assert logp1 == logp1_expected

def test_score_data():
    x = jnp.zeros((5, 6))
    c = jnp.array([0, 1, 0, 1, 0], dtype=int)

    dist  = Normal(
        mu=jnp.vstack((
            jnp.zeros(6),
            jnp.ones(6))),
        std=jnp.vstack((
            jnp.ones(6),
            jnp.ones(6),
        )))

    x_logpdfs = jax.vmap(logpdf, in_axes=(None, 0, 0))(dist, x, c)
    assert x_logpdfs[0] == x_logpdfs[2]
    assert x_logpdfs[2] == x_logpdfs[5]
    assert x_logpdfs[1] == x_logpdfs[3]
    assert x_logpdfs[0] != x_logpdfs[1]

def test_score_trace_cluster():
    """
    What is this testing? P(x | m_c), where m_c is hierarchical model (inverse-gamma
    AND normal) for each cluster c.

    So I guess, kinda P(x|mu, sigma) x P(mu| posterior-hypers) x P(sigma| posterior-hypers)
    """
    x = jnp.zeros((5, 2))
    c = jnp.array([0, 0, 0, 1, 1], dtype=int)
    pi = jnp.array([1/2, 1/2, jnp.nan])
    f = Normal(
        mu=jnp.array(
            [[0, 1],
            [1, 1],
            [10, 10]]
            ),
        std=jnp.ones((3,2)))
    g = NormalInverseGamma(
        m=jnp.zeros(2), b=jnp.ones(2), a=jnp.ones(2),
        l=jnp.ones(2))
    cluster = Cluster(c=c, pi=pi, f=f)
    max_clusters = 3

    logprobs = score_trace_cluster(x, g, cluster, max_clusters)
    # The next 3 lines compute 3 rows for the first cluster in two
    # columns and 2 rows for the second cluster.
    cluster0_x_prob = 3 * jax.scipy.stats.norm.logpdf(0) + \
        3 * jax.scipy.stats.norm.logpdf(0, loc=1)
    cluster1_x_prob = 4 * jax.scipy.stats.norm.logpdf(0, loc=1)

    # Running logpdf(dist: NormalInverseGamma, x: Normal)
    # scoring mu and sigma for each cluster, i.e.
    # P(mu| posterior-hypers) x P(sigma| posterior-hypers)
    cluster0_theta_prob = logpdf(g, f[0])
    cluster1_theta_prob = logpdf(g, f[1])

    # sum up to compute the full term.
    cluster0_prob = cluster0_x_prob + cluster0_theta_prob
    cluster1_prob = cluster1_x_prob + cluster1_theta_prob
    assert jnp.isclose(cluster0_prob, logprobs[0])
    assert jnp.isclose(cluster1_prob, logprobs[1])

@pytest.mark.xfail(reason="q_Z doesn't exist anymore - not yet sure what this should test")
def test_q_Z():
    x = jnp.zeros((5, 2))
    max_clusters = 2
    c = jnp.array([0, 0, 2, 1, 3], dtype=int)
    pi = jnp.array([1/2, 1/3, 1/2, 2/3])
    f = Normal(
        mu=jnp.array(
            [[0, 1],
            [1, 1],
            [10, 10],
            [0, 10]]
            ),
        std=jnp.ones((4,2)))

    cluster = Cluster(c=c, pi=pi, f=f)

    Z = q_Z(cluster, x, max_clusters)

    z0 = jnp.array([
        jnp.log(1/2) + jax.scipy.stats.norm.logpdf(0) + jax.scipy.stats.norm.logpdf(1),
        jnp.log(1/2) + jax.scipy.stats.norm.logpdf(10) + jax.scipy.stats.norm.logpdf(10),
    ])
    z0 = jnp.log(jnp.sum(3 * jnp.exp(z0)))


    z1 = jnp.array([
        jnp.log(1/3) + jax.scipy.stats.norm.logpdf(1) + jax.scipy.stats.norm.logpdf(1),
        jnp.log(2/3) + jax.scipy.stats.norm.logpdf(0) + jax.scipy.stats.norm.logpdf(10),
    ])
    z1 = jnp.log(jnp.sum(2 * jnp.exp(z1)))

    assert jnp.isclose(z0, Z[0])
    assert jnp.isclose(z1, Z[1])
