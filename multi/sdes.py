import jax
import numpy as np
from jax import jit
import jax.numpy as jnp
from utils import make_B
from tqdm import tqdm


from risks_and_discounts import *


class OptimizerSDE:
    def __init__(self, problem, key=None):
        self.problem = problem        
        self.key = key if key is not None else jax.random.PRNGKey(np.random.randint(0, 10000))        

        if problem == 'linreg':
            self.risk_fun = risk_from_B_linreg
            self.f = f_linreg        
    
    def run(self, params, optimal_params, cov, T, lr_fun, dt = 0.005, **kwargs):
        risks = []                
        key = self.key
                                   
        B = make_B(params, optimal_params, cov)                
        iters = int(T / dt)
        for i in tqdm(range(iters)):
            t = i * dt
            key, subkey = jax.random.split(key)
            if callable(lr_fun):
                lr = lr_fun(t)
            else:
                lr = lr_fun
            params = self.update(params, optimal_params, B, lr, cov, dt, subkey, **kwargs)
            risks.append(self.risk_fun(B))
            B = make_B(params, optimal_params, cov)

        return params, jnp.array(risks), jnp.linspace(0, T, iters)

    def tree_flatten(self):        
        leaves = (self.key)
        aux_data = (self.problem)
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):        
        key = leaves
        problem = aux_data
        return cls(problem, key=key)


@jax.tree_util.register_pytree_node_class
class AdamSDE(OptimizerSDE):
    
    def run(self, params, optimal_params, cov, T, lr, dt=0.005, **kwargs):
        if len(cov.shape) == 1:
            covbar = cov / jnp.sqrt(cov)
        else:
            covbar = cov / jnp.sqrt(jnp.diag(cov))[:, None]

        kwargs['covbar'] = covbar

        return super().run(params, optimal_params, cov, T, lr, dt, **kwargs)
    
    @jit
    def update(self, params, optimal_params, B, lr, cov, dt, subkey, **kwargs):
        beta2 = kwargs['beta2']
        covbar = kwargs['covbar']
        
        d, m = params.shape
        subkey_phi, subkey_cov, subkey_brownian = jax.random.split(subkey, 3)
        W = jax.random.normal(subkey_brownian, (d,m)) * jnp.sqrt(dt)
        
        phi = phi_from_B(B, self.f, beta2, subkey_phi)
                
        if len(covbar.shape) == 1:
            mean =  covbar[:,None] * params * phi[0:m] + covbar[:,None] * optimal_params * phi[m:]
        else:
            mean =  covbar @ params * phi[0:m] + covbar @ optimal_params * phi[m:]
        
        cov = cov_from_B(B, self.f, beta2,  subkey_cov)
        params = params - lr * mean * dt + lr * W @ jnp.sqrt(cov) / jnp.sqrt(d)
        # cov = 1
        # params = params - lr * mean * dt + lr * W * jnp.sqrt(cov) / jnp.sqrt(d)
                
        return params


@jax.tree_util.register_pytree_node_class
class SgdSDE(OptimizerSDE):
    
    @jit
    def update(self,  params, optimal_params, B, lr, cov, dt, subkey, **kwargs):
        d, num_classes = params.shape
        subkey_mean, subkey_cov, subkey_brownian = jax.random.split(subkey, 3)
        W = jax.random.normal(subkey_brownian, optimal_params.shape) * jnp.sqrt(dt)
                
        H = compute_H(B, self.f, num_classes, subkey_mean)
        I = compute_I(B, self.f, num_classes, subkey_cov)
        
        if len(cov.shape) == 1:            
            mean =  cov[:,None] * params * H[0] + cov[:,None] * optimal_params * H[1]
        else:
            mean =  cov @ params * H[0] + cov @ optimal_params * H[1]
                        
        params = params - lr * mean * dt + lr * jnp.sqrt(I) * W * jnp.sqrt(cov)[:,None] / jnp.sqrt(d)
                
        return params
    