import genjaxmix.model.dsl as dsl
import jax
import jax.numpy as jnp
import genjaxmix.analytical.logpdf as logpdf
from genjaxmix.model.utils import count_unique
from dataclasses import dataclass
from typing import List, Dict
from genjaxmix.analytical.posterior import (
    get_segmented_posterior_sampler,
    get_posterior_sampler,
)


@dataclass
class MarkovBlanket:
    id: int
    parents: List[int]
    children: List[int]
    cousins: Dict
    types: Dict
    observed: Dict


def gibbs_pi(key, assignments, pi):
    alpha = 1.0
    K = count_unique(assignments)
    K_max = pi.shape[0]
    counts = jnp.bincount(assignments, length=K_max)
    alpha_new = jnp.where(jnp.arange(K_max) < K, counts, alpha)
    alpha_new = jnp.where(jnp.arange(K_max) < K + 1, alpha_new, 0.0)
    pi_new = jax.random.dirichlet(key, alpha_new)
    return pi_new


def build_parameter_proposal(blanket: MarkovBlanket):
    id = blanket.id
    if blanket.observed[id] or blanket.types[id] == dsl.Constant:
        return None

    if has_conjugate_rule(blanket):
        return build_gibbs_proposal(blanket)
    else:  # default to MH
        return build_mh_proposal(blanket)


def build_loglikelihood_at_node(blanket: MarkovBlanket):
    id = blanket.id
    observed = blanket.observed
    is_vectorized = observed[id] or any(
        [observed[parent] for parent in blanket.parents]
    )

    logpdf_lambda = logpdf.get_logpdf(blanket.types[id])
    if is_vectorized:
        inner_axes = (None,) + tuple(
            None if blanket.observed[ii] else 0 for ii in blanket.parents
        )
        outer_axes = (0,) + tuple(
            0 if blanket.observed[ii] else None for ii in blanket.parents
        )

        def loglikelihood(environment):
            return jax.vmap(
                jax.vmap(logpdf_lambda, in_axes=inner_axes), in_axes=outer_axes
            )(environment[id], *[environment[parent] for parent in blanket.parents])
    else:
        axes = (0,) + tuple(0 for ii in blanket.parents)

        def loglikelihood(environment):
            return jax.vmap(logpdf_lambda, in_axes=axes)(
                environment[id], *[environment[parent] for parent in blanket.parents]
            )

    return loglikelihood, is_vectorized


#############
# Conjugacy #
#############
def has_conjugate_rule(blanket: MarkovBlanket):
    if len(blanket.children) != 1:
        return False

    prior = blanket.types[blanket.id]
    likelihood = blanket.types[blanket.children[0]]
    return (prior, likelihood) in CONJUGACY_RULES


def get_conjugate_rule(blanket: MarkovBlanket):
    prior = blanket.types[blanket.id]
    likelihood = blanket.types[blanket.children[0]]
    return CONJUGACY_RULES[(prior, likelihood)]


def jax_normal_normal_sigma_known(blanket: MarkovBlanket):
    mu_0_id, sig_0_id = blanket.parents
    sig_id = blanket.cousins[blanket.children[0]][1]
    observations = blanket.children[0]
    return (mu_0_id, sig_0_id, sig_id, observations)


def jax_gamma_normal_mu_known(blanket: MarkovBlanket):
    alpha_id, beta_id = blanket["parents"]
    mu_id = blanket["cousins"][0]
    return (alpha_id, beta_id, mu_id)


CONJUGACY_RULES = {
    (dsl.Normal, dsl.Normal): jax_normal_normal_sigma_known,
    (dsl.Gamma, dsl.Normal): jax_gamma_normal_mu_known,
}


def build_gibbs_proposal(blanket: MarkovBlanket):
    arg_rule = get_conjugate_rule(blanket)
    if arg_rule is None:
        raise ValueError("No conjugate rule found???")

    id = blanket.id
    proposal_signature = (blanket.types[id], blanket.types[blanket.children[0]])
    posterior_args = arg_rule(blanket)

    # check if likelihood is an observable

    likelihood_observed = blanket.observed[blanket.children[0]]

    if not likelihood_observed:
        parameter_proposal = get_posterior_sampler(*proposal_signature)

        def gibbs_sweep(key, environment, assignments):
            environment = environment.copy()
            conditionals = [environment[ii] for ii in posterior_args]
            environment[id] = parameter_proposal(key, conditionals)
            return environment

        return gibbs_sweep
    else:
        parameter_proposal = get_segmented_posterior_sampler(*proposal_signature)

        def gibbs_sweep(key, environment, assignments):
            environment = environment.copy()
            conditionals = [environment[ii] for ii in posterior_args]
            environment[id] = parameter_proposal(key, conditionals, assignments)

            return environment

        return gibbs_sweep


###############
# MH Proposal #
###############


def _build_obs_likelihood_at_node(blanket: MarkovBlanket, substitute_id):
    # TODO: Both cases seem to be the same. See if combining works.
    id = blanket.id
    observed = blanket.observed
    is_vectorized = observed[id] or any(
        [observed[parent] for parent in blanket.parents]
    )

    logpdf_lambda = logpdf.get_logpdf(blanket.types[id])
    if is_vectorized:
        axes = (0,) + tuple(0 for ii in blanket.parents)

        def loglikelihood(substituted_value, assignments, environment):
            def promote_shape(ii, arr, assignments):
                if blanket.observed[ii]:
                    return arr
                else:
                    return arr[assignments]

            def swap(id):
                if id == substitute_id:
                    return substituted_value
                else:
                    return environment[id]

            x = swap(id)

            return jax.vmap(logpdf_lambda, in_axes=axes)(
                promote_shape(id, x, assignments),
                *[
                    promote_shape(parent, swap(parent), assignments)
                    for parent in blanket.parents
                ],
            )
    else:
        axes = (0,) + tuple(0 for ii in blanket.parents)

        def loglikelihood(substituted_value, assignments, environment):
            def swap(id):
                if id == substitute_id:
                    return substituted_value
                else:
                    return environment[id]

            return jax.vmap(logpdf_lambda, in_axes=axes)(
                swap(id), *[swap(parent) for parent in blanket.parents]
            )

    return loglikelihood, is_vectorized


def build_mh_proposal(blanket: MarkovBlanket):
    likelihood_fns = {"observed": dict(), "unobserved": dict()}
    id = blanket.id
    likelihood, is_vectorized = _build_obs_likelihood_at_node(blanket, id)
    if is_vectorized:
        likelihood_fns["observed"][id] = likelihood
    else:
        likelihood_fns["unobserved"][id] = likelihood

    for child in blanket.children:
        fake_blanket = MarkovBlanket(
            child, blanket.cousins[child], [], [], blanket.types, blanket.observed
        )
        likelihood, is_vectorized = _build_obs_likelihood_at_node(fake_blanket, id)
        if is_vectorized:
            likelihood_fns["observed"][child] = likelihood
        else:
            likelihood_fns["unobserved"][child] = likelihood

    def mh_move(key, environment, assignments):
        environment = environment.copy()

        # TODO: random walk must use the correct sample space
        x_old = environment[id]
        x_new = x_old + 0.1 * jax.random.normal(key, shape=x_old.shape)

        ratio = 0.0
        for ii, likelihood_fn in likelihood_fns["unobserved"].items():
            ratio += likelihood_fn(x_new, assignments, environment)
            ratio -= likelihood_fn(x_old, assignments, environment)

        for ii, likelihood_fn in likelihood_fns["observed"].items():
            increment = likelihood_fn(x_new, assignments, environment) - likelihood_fn(
                x_old, assignments, environment
            )
            increment = jax.ops.segment_sum(
                increment, assignments, num_segments=x_old.shape[0]
            )
            ratio += increment

        logprob = jnp.minimum(0.0, ratio)

        u = jax.random.uniform(key, shape=ratio.shape)
        accept = u < jnp.exp(logprob)
        x = jnp.where(accept[:, None], x_new, x_old)
        environment[id] = x
        return environment

    return mh_move
