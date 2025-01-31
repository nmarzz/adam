import jax
import numpy as np
from jax import jit
import jax.numpy as jnp
from jax.nn import sigmoid as sig
from utils import make_data, make_B
from tqdm import tqdm
import pickle


@jit
def logreg(data, theta):
    logits = jnp.dot(data, theta)  
    probabilities = sig(logits)
    return probabilities

@jit
def grad_logreg(params, data, target):
    p_student = logreg(data, params)
    return (p_student - target) * data

@jit
def grad_linreg(params, data, target):    
    return (jnp.inner(data, params) - target) * data

@jit
def linreg_target(optimal_params, data):
    return jnp.inner(optimal_params, data)

@jit
def logreg_target(optimal_params, data):
    return sig(jnp.inner(optimal_params, data))

@jit
def risk_from_B_linreg(B):
    return (B[0,0] + B[1,1] - 2 * B[0,1]) / 2



from functools import partial
from jax.scipy.stats import norm

@partial(jax.jit, static_argnames=['func'])
def simpsons_rule(func, a, b, num_points=101):
    """
    Numerical integration using Simpson's rule.
    Assumes `num_points` is odd for simplicity.
    """
    # if num_points % 2 == 0:
    #     raise ValueError("num_points must be odd for Simpson's rule.")
    
    x = jnp.linspace(a, b, num_points)  # Uniformly spaced points
    dx = (b - a) / (num_points - 1)    # Step size
    y = func(x)                        # Function values
    
    # Simpson's rule weights: 1, 4, 2, ..., 4, 1
    weights = jnp.ones(num_points)
    weights = weights.at[1:-1:2].set(4)
    weights = weights.at[2:-1:2].set(2)
    
    return jnp.sum(weights * y) * dx / 3

def integrand_first_term(x, B11):
    """Integrand for the first term: E[e^X / (1 + e^X)^2]."""
    return jnp.exp(x) / (1 + jnp.exp(x))**2 * norm.pdf(x, scale=jnp.sqrt(B11))

def integrand_second_term(x, B00):
    """Integrand for the second term: E[log(1 + e^Y)]."""
    return jnp.log(1 + jnp.exp(x)) * norm.pdf(x, scale=jnp.sqrt(B00))

@jit
def risk_from_B_logreg(B):
    B00 = B[0, 0]
    B11 = B[1, 1]
    B01 = B[0, 1]
    
    # Define bounds for the Gaussian (truncate effectively at ±5 standard deviations)
    # bounds = 5.0 * jnp.sqrt(jnp.max(B00, B11))
    bounds = 10
    
    # Compute the first term integral using Simpson's rule
    first_term = simpsons_rule(
        lambda x: integrand_first_term(x, B11), -bounds, bounds
    )
    
    # Compute the second term integral using Simpson's rule
    second_term = simpsons_rule(
        lambda x: integrand_second_term(x, B00), -bounds, bounds
    )
    
    # Combine terms
    risk = -B01 * first_term + second_term
    return risk


import jax
import jax.numpy as jnp
import numpy as np
import pickle
from tqdm import tqdm

@jax.tree_util.register_pytree_node_class
class Adam:
    def __init__(self, problem, key=None):
        self.problem = problem        
        self.key = key if key is not None else jax.random.PRNGKey(np.random.randint(0, 10000))        

        if problem == 'logreg':
            self.grad = grad_logreg
            self.get_target = linreg_target
        elif problem == 'linreg':
            self.grad = grad_linreg
            self.get_target = logreg_target
                 
    @jax.jit
    def update(self, params, m, v, lr, beta1, beta2, cov, optimal_params, eps, key):
        """JAX-compatible functional update that returns new parameters and state."""
        key, subkey = jax.random.split(key)
        data = make_data(cov, subkey)
        target = self.get_target(data, optimal_params)

        gradient = self.grad(params, data, target)
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2

        m_hat = m  # Bias-corrected first moment
        v_hat = v  # Bias-corrected second moment

        params = params - lr * m_hat / (jnp.sqrt(v_hat + eps))
        
        return params, m, v, key

    def run(self, params, cov, T, lr, beta1, beta2, optimal_params, eps=0):
        """Runs the optimizer for T steps, updating params explicitly."""
        risks = []
        risk_fun = risk_from_B_linreg if self.problem == 'linreg' else risk_from_B_logreg

        d = len(cov)
        m, v = jnp.zeros(d), jnp.zeros(d)
        key = self.key

        for _ in tqdm(range(T * d)):
            params, m, v, key = self.update(params, m, v, lr, beta1, beta2, cov, optimal_params, eps, key)
            B = make_B(params, optimal_params, cov)
            risks.append(risk_fun(B))

        return params, risks

    def tree_flatten(self):
        """Flatten the PyTree into a tuple of arrays and auxiliary data."""
        leaves = (self.key)
        aux_data = (self.problem)
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        """Reconstruct the PyTree from its flattened representation."""
        key = leaves
        problem = aux_data
        return cls(problem, key=key)
