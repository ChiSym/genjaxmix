# alpha = 1.0
# K_max = 6
# key = jax.random.key(1)
# key, *subkeys = jax.random.split(key, 5)
# assignments = jax.random.categorical(subkeys[0], jnp.ones(K) / K, shape=N)
# pi = dpmm.gibbs_pi(subkeys[1], jnp.array([alpha]), assignments, jnp.ones(K_max), K)

# prior = core.Normal()
# likelihood = core.Normal()
# parameter_proposal = dpmm.gibbs_parameters_proposal(prior, likelihood)
# hyperparameters = (
#     jnp.zeros((K_max, 2)),
#     jnp.ones((K_max, 2)),
#     0.1 * jnp.ones((K_max, 2)),
# )
# mu = parameter_proposal(subkeys[2], hyperparameters, observations, assignments)

# z_proposal = dpmm.gibbs_z_proposal(prior, likelihood)
# z_proposal(subkeys[3], hyperparameters, (mu,), observations, pi, K)
