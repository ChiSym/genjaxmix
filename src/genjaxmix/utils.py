import jax.numpy as jnp

def beta_to_logpi(betas):
    logb = jnp.log(betas)
    logb_not = jnp.log(1-betas)
    C = betas.shape[0]
    logpi = jnp.zeros(C)
    for i in range(1,C):
        logpi = logpi.at[i].set(jnp.sum(logb_not[:i]))
    for i in range(C):
        logpi = logpi.at[i].set(logpi[i] + logb[i])
    return logpi

def logpi_to_beta(logpi):
    C = logpi.shape[0]
    betas = logpi[0]*jnp.ones(C)
    for i in range(1,C):
        betas = betas.at[i].set(logpi[i]-jnp.sum(jnp.log(-jnp.expm1(betas[:i]))))
    return betas