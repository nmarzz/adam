import jax
import jax.numpy as jnp
from jax import jit
import numpy as np

def compute_ci(data, alpha=0.2):
    lower = jnp.percentile(data, 100 * (alpha / 2), axis=0)
    upper = jnp.percentile(data, 100 * (1 - alpha / 2), axis=0)
    mean = jnp.mean(data, axis=0)
    return mean, lower, upper

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

def sym_sqrt(B):
    """Symmetric square root of a PSD matrix (eigenvalues clipped at zero).

    Unlike a Cholesky factor it is defined for singular ``B`` (e.g. a zero
    student at initialization), so Monte-Carlo laws ``O + sym_sqrt(B) @ g`` stay
    valid on the whole PSD cone.
    """
    vals, vecs = jnp.linalg.eigh((B + B.T) / 2)
    return (vecs * jnp.sqrt(jnp.clip(vals, 0.0))) @ vecs.T


def haar_orthogonal(key, d):
    """Haar-distributed ``d x d`` orthogonal matrix (QR with sign correction)."""
    Q, R = jnp.linalg.qr(jax.random.normal(key, (d, d)))
    return Q * jnp.sign(jnp.diag(R))[None, :]
