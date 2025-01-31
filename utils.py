import jax
import jax.numpy as jnp
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
    
    # if len(cov.shape) == 1:        
    #     B11 = jnp.dot(params, cov * params)
    #     B12 = jnp.dot(optimal_params, cov * params)
    #     B22 = jnp.dot(optimal_params, cov * optimal_params)
    #     B = jnp.array([[B11,B12],[B12, B22]])
    # else:
    #     B11 = jnp.dot(params, cov @ params)
    #     B12 = jnp.dot(optimal_params, cov @ params)
    #     B22 = jnp.dot(optimal_params, cov @ optimal_params)
    #     B = jnp.array([[B11,B12],[B12, B22]])
    
    B11 = jnp.dot(params, cov * params)
    B12 = jnp.dot(optimal_params, cov * params)
    B22 = jnp.dot(optimal_params, cov * optimal_params)
    B = jnp.array([[B11,B12],[B12, B22]])
    
    return B