import jax
import jax.numpy as jnp
from jax import jit
import numpy as np


def make_data(cov, key = None):
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 10000))
        
    d = len(cov)    
    
    if len(cov.shape) == 1:
        # covariance is diagonal
        return jax.random.normal(key, (d,)) * jnp.sqrt(cov)
        
    return jax.random.multivariate_normal(key, mean=jnp.zeros(d), cov=cov)

@jax.jit
def make_B(params, optimal_params, cov):
    W =  jnp.concatenate([params, optimal_params], axis = 1)
    if len(cov.shape) == 1:        
        B = W. T @ (W * cov[:,None])
    else:
        B = W. T @ cov @ W   
    return B