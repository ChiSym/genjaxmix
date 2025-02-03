"""Provides the generative function for a Dirichlet process mixture model.

This module allows the user to create and execute Dirichlet Process Mixture Models (DPMM) using GenJAX.
"""


from genjax import gen, repeat
from genjax import normal, inverse_gamma, dirichlet, categorical, beta
import jax.numpy as jnp
from .utils import beta_to_logpi
import jax

K = 5 
L_num = 7

@gen
def hyperparameters(mu_0=0.0, l=1.0, shape=1.0, scale=1.0, alpha=1.0):
    """
        µ, sigma^2 ~ N(µ|m_0, sigma^2*l)IG(sigma^2| shape, scale)
    """
    sigma_sq = inverse_gamma(shape*jnp.ones(K), scale*jnp.ones(K)) @ "sigma"
    sigma = jnp.sqrt(sigma_sq)
    mu = normal(mu_0 * jnp.ones(K), sigma * l) @ "mu"
    logp = dirichlet(alpha*jnp.ones((K, L_num))) @ "logp"
    logp = jnp.log(logp)
    return mu, sigma, logp

@gen
def cluster(pi:jax.Array, mu:jax.Array, sigma:jax.Array, logp:jax.Array):
    """Sample from a mixture model with proportions ``pi``, normal inverse gamma parameters ``mu`` and ``sigma``, and 
    categorical parameter ``logp``.

    Args:
        pi - a one dimensional array of proportions
        mu: - a K-dimensional array of means
        sigma - a K-dimensional array of standard deviations
        logp - a KxL_num array of log probabilities
    
    Returns:
        idx - an integer representing the cluster assignment
        y1 - an array representing the numerical feature
        y2 - an array representing the categorical feature

    """
    idx = categorical(pi) @ "c"
    y1 = normal(mu[idx], sigma[idx]) @ "y1"
    y2 = categorical(logp[idx]) @ "y2"
    return idx, y1, y2

@gen
def gem(alpha:float) -> jnp.ndarray:
    """Sample from a Griffiths, Engen, and McCloskey's (GEM) distribution with concentration ``alpha``.

    Args:
        alpha: a positive scalar
    
    Returns:
        A random array given by shape ``K``
        
    """
    betas = beta(jnp.ones(K), alpha*jnp.ones(K)) @ "pi"
    pi = beta_to_logpi(betas)
    return pi


# @gen
# def dpmm(concentration=1.0, mu_0=0.0, l=1.0, a=1.0, b=1.0):
#     """Sample from a Dirichlet process mixture model.

#     Args:
#         concentration: 
#         mu_0: ?
#         precision: ?
#         a: shape of the inverse gamma
#         b: scale of the inverse gamma
    
#     Returns:
#         A triplet ``(c, y1, y2)`` of three arrays. The first value, ``c``, is the assignments. The values ``y1` and ``y2``
#         represent the numerical and categorical features of each data point, respectively.
#     """

#     logpi = gem(concentration) @ "pi"
#     mu, sigma, logp = hyperparameters(mu_0, l, a, b) @ "hyperparameters"
#     y = cluster_repeat(logpi, mu, sigma, logp) @ "assignments"
#     return y

def generate(N_max: int):
    """ Construct a Dirichlet Procsess Mixture Model with a given number of data points.

    Args:
        N_max: maximum number of data points
    
    Returns:
        A generative function that generates a DPMM model with a given number of data points
    """
    cluster_repeat = repeat(n=N_max)(cluster)

    @gen
    def dpmm(concentration:float=1.0, mu_0:float=0.0, l:float=1.0, a:float=1.0, b:float=1.0):
        """Sample from a Dirichlet process mixture model.

        Args:
            concentration: Dirichlet Process concentration
            mu_0: mean prior
            precision: precision prior
            a: shape of the inverse gamma
            b: scale of the inverse gamma
        
        Returns:
            A triplet ``(c, y1, y2)`` of three arrays. The first value, ``c``, is the assignments. The values ``y1` and ``y2``
            represent the numerical and categorical features of each data point, respectively.
        """

        logpi = gem(concentration) @ "pi"
        mu, sigma, logp = hyperparameters(mu_0, l, a, b) @ "hyperparameters"
        y = cluster_repeat(logpi, mu, sigma, logp) @ "assignments"
        return y
    return dpmm