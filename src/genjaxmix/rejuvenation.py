import jax
import jax.numpy as jnp
from genjax import inverse_gamma, normal, dirichlet, Pytree,  gen
from genjax.typing import PRNGKey
from genjax._src.generative_functions.distributions.distribution import Distribution
from tensorflow_probability.substrates import jax as tfp
from genjax._src.core.interpreters.incremental import Diff
from .dpmm import K
from .utils import beta_to_logpi, logpi_to_beta
from .conjugacy import posterior_dirichlet, posterior_normal_inverse_gamma

tfd = tfp.distributions

def gibbs_move(model, proposal, model_args, tr, observations, key):
    proposal_args = (observations,)
    fwd_choices, fwd_weight, _ = proposal.propose(key, proposal_args)

    key, subkey = jax.random.split(key)
    argdiffs = Diff.no_change(model_args)
    tr_new, weight, _, discard = model.update(subkey, tr, fwd_choices, argdiffs)
    return tr_new


@gen
def propose_parameters(obs):
    _propose_parameters(obs) @ "hyperparameters"

@gen
def _propose_parameters(obs):
    c = obs[:, "c"]
    y1 = obs[:, "y1"]
    y2 = obs[:, "y2"]

    mu_n, v_n, a, b = posterior_normal_inverse_gamma(c, y1)
    frequency_matrix = posterior_dirichlet(c, y2) + 1e-6 # to prevent degeneracy

    # Propose sigma
    sigma_sq = inverse_gamma(a,b) @ "sigma"
    sigma = jnp.sqrt(sigma_sq)

    # Propose mu
    mu = normal(mu_n, sigma*v_n) @ "mu"

    # Propose logp
    p = dirichlet(frequency_matrix) @ "logp"
    
    return mu, sigma, p


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
        logpi = jnp.log(pi)
        # logpi = apply_decay(pi, gamma=0.80)

        betas = logpi_to_beta(logpi)

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