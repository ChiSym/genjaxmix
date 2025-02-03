import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

import genjax
from genjax import Pytree
from genjax._src.generative_functions.distributions.distribution import Distribution
from genjax.typing import PRNGKey, Array, Float

tfd = tfp.distributions

@Pytree.dataclass
class NormalInverseGamma(Distribution):
    def random_weighted(self, key: PRNGKey, mu, l, a, b):
        ig = tfd.InverseGamma(concentration=a, scale=b)
        key, subkey = jax.random.split(key)
        precision = ig.sample(seed=subkey)
        ig_logp = ig.log_prob(precision)

        normal = tfd.Normal(loc=mu, scale=precision / l)
        key, subkey = jax.random.split(key)
        mu = normal.sample(seed=subkey)
        mu_logp = normal.log_prob(mu)
        
        retval = jnp.stack([mu, precision], axis=1)
        inv_logp = -jnp.sum(ig_logp) - jnp.sum(mu_logp)
        return inv_logp, retval

    def estimate_logpdf(self, key: PRNGKey, x, mu, l, a, b):
        mu_sampled = x[:,0]
        precision = x[:,1]
        ig = tfd.InverseGamma(concentration=a, scale=b)
        ig_logp = ig.log_prob(precision)
        normal = tfd.Normal(loc=mu, scale= precision/l)
        mu_logp = normal.log_prob(mu_sampled)
        return jnp.sum(ig_logp) + jnp.sum(mu_logp)

@Pytree.dataclass
class Dirichlet(Distribution):
    def random_weighted(self, key:PRNGKey, alpha):
        dir = tfd.Dirichlet(concentration = alpha)
        probs = dir.sample(seed=key)
        inv_weight = -dir.log_prob(probs)
        return inv_weight, probs
    def estimate_logpdf(self, key:PRNGKey, x, alpha):
        dir = tfd.Dirichlet(concentration = alpha)
        return dir.log_prob(x)

"""
A class to store DP samples and the corresponding beta values. 

Used in GEM to avoid floating point error
"""
@Pytree.dataclass
class DPSample(Pytree):
    betas: Array
    pi: Array
    def __init__(self, betas, pi):
        self.betas = betas
        self.pi = pi

@Pytree.dataclass
class GEM(Distribution):
    C: int = Pytree.static(default=1)
    def __init__(self, C:int=10):
        self.C = jnp.asarray(C)
    def random_weighted(self, key: PRNGKey, alpha: Float):
        C = self.C
        sampler = tfd.Beta(concentration1 = jnp.array(alpha), concentration0=jnp.array(1.0))
        betas = sampler.sample(seed=key, sample_shape = C)
        inv_weight = -jnp.sum(sampler.log_prob(betas))
        betas_not = 1-betas

        betas = jnp.log(betas)
        betas_not = jnp.log(betas_not)
        # prefix sum of betas
        logpi = jnp.zeros(C)
        for i in range(1,C):
            logpi = logpi.at[i].set(jnp.sum(betas_not[:i]))
        for i in range(C):
            logpi = logpi.at[i].set(logpi[i] + betas[i])

        return inv_weight, logpi

    def estimate_logpdf(self, key: PRNGKey, pi, alpha: Float):
        # assumes dist.pi corresponds to dist.betas
        sampler = tfd.Beta(concentration1 = jnp.array(alpha), concentration0 = jnp.array(1.0))
        def unfold(carry, pi):
            logbeta = pi - carry
            return carry + jnp.log(-jnp.expm1(logbeta)) , jnp.exp(logbeta)

        _, betas = jax.lax.scan(unfold, 0.0, pi)
        weight = jnp.sum(sampler.log_prob(betas))
        return weight

nig = NormalInverseGamma()
dirichlet = Dirichlet()
@Pytree.dataclass
class MixtureModel(Distribution):
    def random_weighted(self, key, pi, categorical_probs):
        key_0, key_1 = jax.random.split(key, 2)
        cluster_dist = tfd.Categorical(pi)
        c = cluster_dist.sample(seed=key_0)
        c_logp = cluster_dist.log_prob(c)
        label_dist = tfd.Categorical(categorical_probs[c])
        y = label_dist.sample(seed=key_1)
        y_logp = label_dist.log_prob(y)
        return -c_logp-y_logp, (c,y)

    def estimate_logpdf(self, x, pi, categorical_probs):
        c, y = x
        cluster_dist = tfd.Categorical(pi)
        label_dist = tfd.Categorical(categorical_probs[c])
        logp = cluster_dist.log_prob(c) + label_dist.log_prob(y)
        return -logp

cmm = MixtureModel()

@genjax.repeat(n=100)
@gen 
def cluster(pi, probs):
    assignments = cmm(pi, probs) @ "assignments"
    return assignments

pi = jnp.ones(10) / 10
categorical_probs = jax.random.uniform(key, (10, 36, 19))
tr = cluster.simulate(key, (pi, categorical_probs,))
tr.get_choices()[0, "assignments"].unmask()