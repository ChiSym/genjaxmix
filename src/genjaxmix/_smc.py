import jax.numpy as jnp
import jax
import numpy as np
from plum import dispatch
from jaxtyping import Array, Integer
from genjaxmix.distributions import (Dirichlet, MixtureModel,
    posterior, sample, logpdf, Normal, Categorical, Mixed, Cluster, Trace)
from functools import partial

@partial(jax.jit, static_argnames=['gibbs_iters', 'max_clusters', 'n_steps'])
def smc(key, trace, data_test, n_steps, data, gibbs_iters, max_clusters):
    smc_keys = jax.random.split(key, n_steps)

    def wrap_step(trace, n):
        key = smc_keys[n]
        keys = jax.random.split(key, 3)
        new_cluster = step(data=data, trace=trace, gibbs_iters=gibbs_iters, 
            max_clusters=max_clusters, key=keys[0], K=n+2)
        split_trace = Trace(
            gem=trace.gem,
            g=trace.g,
            cluster=new_cluster
        )

        rejuvenated_cluster = rejuvenate(keys[1], data, split_trace, gibbs_iters, max_clusters)
        rejuvenated_trace = Trace(
            gem=split_trace.gem,
            g=trace.g,
            cluster=rejuvenated_cluster
        )

        mixture_model = MixtureModel(
            pi=rejuvenated_trace.cluster.pi/jnp.sum(rejuvenated_trace.cluster.pi), 
            f=rejuvenated_trace.cluster.f[:max_clusters])
        logprobs = jax.vmap(logpdf, in_axes=(None, 0))(mixture_model, data_test)
        sum_logprobs = jnp.sum(logprobs)

        return rejuvenated_trace, (rejuvenated_trace, sum_logprobs)

    carry, (trace, sum_logprobs) = jax.lax.scan(wrap_step, trace, jnp.arange(n_steps))
    return trace, sum_logprobs


def rejuvenate(key, data, trace, gibbs_iters, max_clusters):
    extended_pi = jnp.concatenate((trace.cluster.pi, jnp.zeros(max_clusters)))
    log_likelihood_mask = jnp.where(extended_pi == 0, -jnp.inf, 0)

    partial_gibbs_step = partial(gibbs_step, 
        alpha=trace.gem.alpha, g=trace.g, data=data, 
        log_likelihood_mask=log_likelihood_mask, max_clusters=max_clusters,
        rejuvenation=True)

    keys = jax.random.split(key, gibbs_iters)
    _, q_split_trace = jax.lax.scan(partial_gibbs_step, trace.cluster.c, keys)

    cluster = q_split_trace[-1]
    cluster = Cluster(cluster.c, jnp.sum(trace.cluster.pi) * cluster.pi[:max_clusters], cluster.f)

    return cluster

@partial(jax.jit, static_argnames=['gibbs_iters', 'max_clusters'])
def step(data, gibbs_iters, key, K, trace, max_clusters):
    q_split_trace = q_split(data, gibbs_iters, max_clusters, key, trace.cluster.c, trace.gem.alpha, trace.g)

    cluster_weights = get_weights(trace, K, data, q_split_trace, max_clusters)

    logprob_pi0 = logpdf(trace.gem, jnp.sort(trace.cluster.pi, descending=True), K-1)

    weights = jnp.zeros(max_clusters + 1)
    weights = weights.at[1:].set(cluster_weights - logprob_pi0)
    weights = weights.at[0].set(-jnp.inf)  # temp, don't stop
    k = jax.random.categorical(key, weights)

    new_cluster = jax.lax.cond(k==0, 
        lambda cluster0, cluster1, k, K, max_clusters: trace.cluster, 
        lambda  cluster0, cluster1, k, K, max_clusters: split_cluster(cluster0, cluster1, k, K, max_clusters),
        trace.cluster, q_split_trace[-1], k-1, K, max_clusters)

    return new_cluster

def get_weights(trace, K, data, q_split_trace, max_clusters):
    # for each cluster, get the pi score
    pi_split = jax.vmap(make_pi, in_axes=(None, 0, None, None))(
        trace.cluster.pi, jnp.arange(max_clusters), q_split_trace.pi[-1], max_clusters)
    logpdf_pi = jax.vmap(logpdf, in_axes=(None, 0, None))(trace.gem, pi_split, K)

    logpdf_clusters = score_trace_cluster(data, trace.g, trace.cluster, max_clusters)
    logpdf_split_clusters = score_trace_cluster(data, trace.g, q_split_trace[-1], max_clusters, add_c=True)

    return logpdf_pi + logpdf_split_clusters - logpdf_clusters

def score_q_pi(q_pi, max_clusters, alpha):
    q_pi_dist = Dirichlet(alpha=jnp.ones((1, 2)) * alpha/2)
    q_pi_stack = Categorical(
        jnp.vstack((
        jnp.log(q_pi[:max_clusters]),
        jnp.log(q_pi[max_clusters:]),
     ))[None, :])

    return jax.vmap(logpdf, in_axes=(None, -1))(q_pi_dist, q_pi_stack)

def make_pi(pi, k, pi_split, max_clusters):
    pi_k0 = pi[k]
    pi = pi.at[k].set(pi_k0 * pi_split[k])
    idx = jnp.argwhere(pi == 0, size=1)
    pi = pi.at[idx].set(pi_k0 * pi_split[k + max_clusters])

    return jnp.sort(pi, descending=True)

def score_trace_cluster(data, g, cluster, max_clusters, add_c=False):
    c, pi, f = cluster.c, cluster.pi, cluster.f

    x_scores = jax.vmap(logpdf, in_axes=(None, 0, 0))(f, data, c)

    c = jnp.mod(c, max_clusters)

    pi_dist = Categorical(logprobs=jnp.log(pi.reshape(1, -1)))
    theta_scores = jax.vmap(logpdf, in_axes=(None, 0))(g, f)[:max_clusters]

    if add_c:
        c_scores = jax.vmap(logpdf, in_axes=(None, 0))(pi_dist, c.reshape(-1, 1))
        x_scores = x_scores + c_scores

    xc_scores_cluster = jax.ops.segment_sum(
        x_scores, c, 
        num_segments=max_clusters)

    return xc_scores_cluster + theta_scores

def split_cluster(cluster, split_clusters, k, K, max_clusters):
    # update pi
    pi = cluster.pi
    pi0 = pi[k]
    pi = pi.at[k].set(pi0 * split_clusters.pi[k])    
    pi = pi.at[K - 1].set(pi0 * split_clusters.pi[k + max_clusters])    

    # update c
    c = cluster.c
    c = jnp.where(c == k, split_clusters.c, c)
    c = jnp.where(c == k + max_clusters, K-1, c)

    # update f
    f = update_f(cluster.f, split_clusters.f, k, K-1, max_clusters)

    return Cluster(c, pi, f)

@dispatch
def update_f(f0: Normal, f: Normal, k: Integer[Array, ""], K: Integer[Array, ""], max_clusters: Integer[Array, ""]):
    mu = update_vector(f0.mu, f.mu, k, K, max_clusters)
    std = update_vector(f0.std, f.std, k, K, max_clusters)

    return Normal(mu, std)

@dispatch
def update_f(f0: Categorical, f: Categorical, k: Integer[Array, ""], K: Integer[Array, ""], max_clusters: Integer[Array, ""]):
    logprobs = update_vector(f0.logprobs, f.logprobs, k, K, max_clusters)

    return Categorical(logprobs)

@dispatch
def update_f(f0: Mixed, f: Mixed, k: Integer[Array, ""], K: Integer[Array, ""], max_clusters: Integer[Array, ""]):
    return Mixed(
        update_f(f0.normal, f.normal, k, K, max_clusters),
        update_f(f0.categorical, f.categorical, k, K, max_clusters)
    )

def update_vector(v0, split_v, k, K, max_clusters):
    v = v0
    v = v.at[k].set(split_v[k])
    v = v.at[K].set(split_v[k + max_clusters])
    return v

@partial(jax.jit, static_argnames=['gibbs_iters', 'max_clusters'])
def q_split(data, gibbs_iters, max_clusters, key, c0, alpha, g) -> Cluster:
    keys = jax.random.split(key, 3)
    c = (c0 + max_clusters * jax.random.bernoulli(keys[0], shape=c0.shape)).astype(int)
    # c = jnp.concatenate((jnp.zeros(100), jnp.ones(100) * 2))

    log_likelihood_mask = make_log_likelihood_mask(c0, max_clusters)

    partial_gibbs_step = partial(gibbs_step, 
        alpha=alpha, g=g, data=data, 
        log_likelihood_mask=log_likelihood_mask, max_clusters=max_clusters)

    keys = jax.random.split(keys[2], gibbs_iters)
    _, q_split_trace = jax.lax.scan(partial_gibbs_step, c, keys)


    return q_split_trace

def make_log_likelihood_mask(c, max_clusters):
    log_likelihood_mask = -jnp.inf * jnp.ones((c.shape[0], 2 * np.array(max_clusters, dtype=int)))

    n = jnp.arange(c.shape[0], dtype=int)
    clusters_x = jnp.concatenate((n, n))
    clusters_y = jnp.concatenate((c, max_clusters + c), dtype=int)
    return log_likelihood_mask.at[clusters_x, clusters_y].set(0)

def gibbs_step(assignments, key, alpha, g, data, log_likelihood_mask, max_clusters, rejuvenation=False):
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    f = gibbs_f(max_clusters, data, subkey1, g, assignments)
    pi = gibbs_pi(max_clusters, subkey2, alpha, assignments, rejuvenation=rejuvenation)

    assignments = gibbs_c(subkey3, pi, log_likelihood_mask, f, data)

    # now compute the logpdf for the assignments given the new distribution
    return assignments, Cluster(assignments, pi, f)

def gibbs_f(max_clusters, data, key, g, assignments):
    g_prime = posterior(g, data, assignments, 2*max_clusters)
    f = sample(key, g_prime)
    return f

def gibbs_c(key, pi, log_likelihood_mask, f, data):
    log_likelihoods = jax.vmap(jax.vmap(logpdf, in_axes=(0, None)), in_axes=(None, 0))(f, data)
    log_likelihoods = log_likelihoods + log_likelihood_mask
    log_score = log_likelihoods + jnp.log(pi)

    assignments = jax.random.categorical(key, log_score, axis=-1).astype(int)
    return assignments

def gibbs_pi(max_clusters, key, alpha, c, rejuvenation=False):
    cluster_counts = jnp.sum(jax.nn.one_hot(c, num_classes=2 * max_clusters, dtype=jnp.int32), axis=0)
    if rejuvenation:
        pi = jax.random.dirichlet(key, cluster_counts)
        return pi
    else:
        pi = jax.random.dirichlet(key, alpha / 2 + cluster_counts)
        pi_pairs = pi.reshape((2, -1))
        pi_pairs = pi_pairs / jnp.sum(pi_pairs, axis=0)
        return pi_pairs.reshape(-1)