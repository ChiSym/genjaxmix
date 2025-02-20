def build_mh_proposal(self, id: int):
    blanket = self.markov_blanket(id)

    prior_fn = logpdf.logpdf(self.nodes[id])
    # TODO: is an observation node
    is_observation_node = []
    for child in blanket["children"]:
        is_observation_node.append(child in self.observed)

    likelihood_fns = [logpdf.logpdf(self.nodes[child]) for child in blanket["children"]]

    def mh_move(key, environment, observations, assignments):
        x = environment[id]
        x_new = x + 0.1 * jax.random.normal(key, shape=x.shape)

        prior_conditionals = [environment[ii] for ii in blanket["parents"]]
        ratio = 0.0

        for child in blanket["children"]:
            input_nodes = self.edges[child]
            assert id in input_nodes
            idx = input_nodes.index(id)
            # TODO: perform shape promotion to ensure parameters have same shape as observations
            is_observation_node = True
            if is_observation_node:
                likelihood_conditionals = [
                    environment[jj][assignments] for jj in input_nodes
                ]
                log_p_old = likelihood_fns[child](observations, likelihood_conditionals)
                likelihood_conditionals[idx] = x_new[assignments]
                log_p_new = likelihood_fns[child](observations, likelihood_conditionals)
                increment = log_p_new - log_p_old
                increment = jax.ops.segment_sum(
                    increment, assignments, num_segments=x.shape[0]
                )
                ratio += increment
            else:
                likelihood_conditionals = [
                    environment[jj][assignments] for jj in input_nodes
                ]
                log_p_old = likelihood_fns[child](likelihood_conditionals)
                likelihood_conditionals[idx] = x_new[assignments]
                log_p_new = likelihood_fns[child](likelihood_conditionals)
                ratio += log_p_new - log_p_old

        ratio += prior_fn(x_new, prior_conditionals) - prior_fn(x, prior_conditionals)
        logprob = jnp.minimum(0.0, ratio)

        u = jax.random.uniform(key, shape=ratio.shape)
        accept = u < jnp.exp(logprob)
        x = jnp.where(accept[:, None], x_new, x)
        return x

    return mh_move
