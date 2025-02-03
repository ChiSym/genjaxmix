from genjax import ChoiceMapBuilder as C
from genjax import beta, gen, inverse_gamma, normal
from genjax._src.core.interpreters.incremental import Diff
from genjax import Pytree
from genjax._src.generative_functions.distributions.distribution import Distribution
from genjax.typing import PRNGKey
from tensorflow_probability.substrates import jax as tfp
import jax.numpy as jnp
from .utils import beta_to_logpi
import jax
from .vectorized import K
from .conjugacy import posterior_normal_inverse_gamma
tfd = tfp.distributions

# def posterior_normal_inverse_gamma(assignments, x):
    # counts = jnp.bincount(assignments, length=N_max)
    # # sum_x = jnp.where(counts > 0, jax.ops.segment_sum(x, assignments, N_max)/counts, 0.0)
    # sum_x = jax.ops.segment_sum(x, assignments, N_max)
    # sum_x_sq = jax.ops.segment_sum(x**2, assignments, N_max)

    # l_0 = 0.01
    # m_0 = 1.0
    # a_0 = 1.0
    # b_0 = 1.0

    # l = l_0 + counts
    # m = (l_0 * m_0 + sum_x) / l
    # a = a_0 + counts / 2
    # b = b_0 + 0.5 * (sum_x_sq + l_0*m_0**2 - l * m ** 2)
    # return l,m,a,b

# def posterior_dirichlet(assignments, x):
#     one_hot_c = jax.nn.one_hot(assignments, N_max)
#     one_hot_y = jax.nn.one_hot(x, L_num)
#     frequency_matrix = one_hot_c.T @ one_hot_y
#     # print(frequency_matrix)
#     # row_sums = jnp.sum(frequency_matrix, axis=1, keepdims=True)
#     # row_sums = jnp.where(row_sums == 0, 1, row_sums)
#     # empirical = frequency_matrix / row_sums
#     # return jnp.log(empirical)
#     return frequency_matrix

@gen
def propose_parameters(obs):
    _propose_parameters(obs) @ "hyperparameters"

@gen
def _propose_parameters(obs):
    c = obs[:, "c"]
    y1 = obs[:, "y1"]

    mu_0, v_0, a, b = jax.vmap(posterior_normal_inverse_gamma, in_axes=(None, 1))(c, y1)
    mu_0 = mu_0.T
    v_0 = v_0.T
    a = a.T
    b = b.T

    # Propose sigma
    sigma_sq = inverse_gamma(a,b) @ "sigma"
    sigma = jnp.sqrt(sigma_sq)

    # Propose mu
    mu = normal(mu_0, sigma * v_0 ) @ "mu"

    return mu, sigma


def apply_decay(x, gamma):
    decay_factors = jnp.arange(x.shape[0]) * jnp.log(gamma)
    logpi = jnp.log(x) + decay_factors 
    log_max = jnp.max(logpi)
    log_shifted = logpi - log_max
    
    # Compute log-sum-exp for normalization
    # Normalize in log-space
    log_norm = jnp.log(jnp.sum(jnp.exp(log_shifted)))
    logpi = log_shifted - log_norm
    return logpi


@Pytree.dataclass
class DirichletBeta(Distribution):
    def random_weighted(self, key: PRNGKey, alpha):
        sampler = tfd.Dirichlet(concentration = alpha)
        pi = sampler.sample(seed=key)
        # logpi = jnp.log(pi)
        logpi = apply_decay(pi, gamma=0.80)

        def unfold(carry, pi):
            logbeta = pi - carry
            return carry + jnp.log(-jnp.expm1(logbeta)) , jnp.exp(logbeta)
        _, betas = jax.lax.scan(unfold, 0.0, logpi)

        inv_weight = -sampler.log_prob(pi)

        return inv_weight, betas

    def estimate_logpdf(self, key: PRNGKey, betas, alpha):
        sampler = tfd.Dirichlet(concentration = alpha)

        logpi = beta_to_logpi(betas)
        pi = jnp.exp(logpi)

        weight = jnp.sum(sampler.log_prob(pi))
        return weight

dirichlet_beta = DirichletBeta()

@gen 
def propose_pi(obs):
    pi = _propose_pi(obs) @ "pi"
    return pi

@gen
def _propose_pi(obs):
    c = obs[:, "c"]
    proportions = jnp.bincount(c, length = K) + 1e-6
    pi = dirichlet_beta(proportions) @ "pi"
    return pi