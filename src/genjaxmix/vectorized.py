import jax
import jax.numpy as jnp
from genjax import gen, repeat, normal, inverse_gamma, categorical, dirichlet, beta
from .utils import beta_to_logpi

N = 1000
K = 5
F_numerical = 2
F_categorical = 7

@gen
def hyperparameters(mu_0=0.0, v_0=1.0, shape=1.0, scale=1.0, alpha=1.0):
    sigma_sq = inverse_gamma(shape*jnp.ones((K, F_numerical)), scale*jnp.ones((K, F_numerical))) @ "sigma"
    sigma = jnp.sqrt(sigma_sq)
    mu = normal(mu_0*jnp.ones((K, F_numerical)), sigma*v_0) @ "mu"
    # logp = genjax.dirichlet(0.7*jnp.ones((K, L_num))) @ "logp"
    # logp = jnp.log(logp)
    return mu, sigma

@gen
def cluster(pi, mu, sigma):
    idx = categorical(pi) @ "c"
    y1 = normal(mu[idx], sigma[idx]) @ "y1"
    # y2 = genjax.categorical(logp[idx]) @ "y2"
    return idx, y1

@gen
def gem(alpha):
    betas = beta(alpha*jnp.ones(K), jnp.ones(K)) @ "pi"
    pi = beta_to_logpi(betas)
    return pi

def generate(N_max):
    cluster_repeat = repeat(n=N_max)(cluster)

    @gen
    def dpmm(alpha, mu_0, v_0, a, b):
        logpi = gem(alpha) @ "pi"
        mu, sigma = hyperparameters(mu_0, v_0, a, b, 1.0) @ "hyperparameters"
        y = cluster_repeat(logpi, mu, sigma) @ "assignments"
        return y

    return dpmm
