from risks_and_discounts import *

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

class ODE:
    def __init__(self, problem, key=None):
        self.problem = problem        
        self.key = key if key is not None else jax.random.PRNGKey(np.random.randint(0, 10000))        

        if problem == 'logreg':
            self.risk_fun = risk_from_B_logreg
            self.f = f_logreg
        elif problem == 'linreg':
            self.risk_fun = risk_from_B_linreg
            self.f = f_linreg
        elif problem == 'real_phase_ret':
            self.risk_fun = risk_from_B_real_phase_ret
            self.f = f_real_phase_ret
        else:
            raise NotImplementedError(f"{problem} not implement for ODEs")
    
    def run(self, params, optimal_params, cov, T, lr_fun, dt = 0.01, **kwargs):
        risks = []
                
        key = self.key
                                                                                
        risks = []
        ode_time = []
        iters = int(T / dt)
        Bs = []
        
        y, eigs, extra = self.init_odes(cov, params, optimal_params)
        for i in tqdm(range(iters)):
            t = i * dt
            ode_time.append(t)
            
            B = self.make_B(y, eigs)
            R = self.risk_fun(B)
                            
            key, subkey = jax.random.split(key)
            if callable(lr_fun):
                lr = lr_fun(t)
            else:
                lr = lr_fun
            y_update = self.update_odes(y, eigs, B, lr, subkey, extra, **kwargs)
            y += dt * y_update
            
            risks.append(R)
            Bs.append(B)
                        
        return jnp.array(risks), jnp.array(ode_time), jnp.array(Bs)
    
    def make_B(self):
        pass
    
    def init_odes(self):
        pass

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
class AdamODE(ODE):
    
    @staticmethod
    @jit    
    def make_B(y, eigs):
        d = len(eigs)
        p,u,q = y[:d,:,:], y[d:2*d,:,:], y[2*d:,:,:]
        B11 = jnp.sum(p,axis = 0)
        B12 = jnp.sum(u,axis = 0)
        B22 = jnp.sum(q,axis = 0)

        return jnp.block([[B11,B12],[B12.T,B22]])
        
    def init_odes(self, cov, params, optimal_params):
        d = len(cov)
        if len(cov.shape) == 1: # diagonal covariance            
            eigs = jnp.sqrt(cov)
            var_force = cov
            p = jnp.array([jnp.outer(params[j,:], params[j,:]) * cov[j] for j in range(d)])
            u = jnp.array([jnp.outer(params[j,:], optimal_params[j,:]) * cov[j] for j in range(d)])
            q = jnp.array([jnp.outer(optimal_params[j,:], optimal_params[j,:]) * cov[j] for j in range(d)])

        else:            
            covbar = cov / jnp.sqrt(jnp.diag(cov))
            eigs, L = jnp.linalg.eig(covbar)
            R = jnp.linalg.inv(L).T
            eigs, L, R = jnp.real(eigs), jnp.real(L), jnp.real(R)
        
            var_force = jnp.array([jnp.inner(R[:, j], cov @ L[:, j]) for j in range(d)])
            p = jnp.array([params.T @ jnp.outer(cov @ L[:, j], R[:, j]) @ params for j in range(d)])
            u = jnp.array([params.T @ jnp.outer(cov @ L[:, j], R[:, j]) @ optimal_params for j in range(d)])            
            q = jnp.array([optimal_params.T @ jnp.outer(cov @ L[:, j], R[:, j]) @ optimal_params for j in range(d)])
                        
        return jnp.concatenate([p, u, q]), eigs, var_force
    
    @jit
    def update_odes(self, y, eigs, B, lr, subkey, extra, **kwargs):
        var_force = extra
        d = len(eigs)
        p, u, q = y[:d,:,:], y[d:2*d,:,:], y[2*d:,:,:]
        subkey_mean, subkey_cov = jax.random.split(subkey)
        beta2 = kwargs['beta2']
        beta1 = kwargs['beta1']
        
        m = len(B) // 2
        phi = phi_from_B(B, self.f, beta1, beta2, subkey_mean)
        sigma = cov_from_B(B, self.f, beta1, beta2, subkey_cov)
        phi1, phi2 = phi[0:m], phi[m:]
        
        p_update = -2 * lr * eigs[:,None,None]  * (p * phi1 + u * phi2) + lr**2 * var_force[:,None,None] * sigma / d
        u_update = -lr * eigs[:,None,None] * (phi1 * u + phi2 * q)
                
        return jnp.concatenate([p_update, u_update, jnp.zeros(u_update.shape)])
    

@jax.tree_util.register_pytree_node_class
class SgdODE(ODE):
    
    @staticmethod
    @jit
    def make_B(y, eigs):
        d = len(eigs)
        p,u,q = y[:d,:,:], y[d:2*d,:,:], y[2*d:,:,:]
                
        return jnp.einsum('abc,a->bc',jnp.block([[p,u],[jnp.swapaxes(u, 1, 2),q]]), eigs)
    
    def init_odes(self, cov, params, optimal_params):                
        d = len(cov)
        if len(cov.shape) == 1: # diagonal covariance            
            eigs = cov
                                            
            p = jnp.array([jnp.outer(params[j,:], params[j,:]) for j in range(d)])
            u = jnp.array([jnp.outer(params[j,:], optimal_params[j,:]) for j in range(d)])
            q = jnp.array([jnp.outer(optimal_params[j,:], optimal_params[j,:]) for j in range(d)])
        else:                        
            eigs, L = jnp.linalg.eig(cov)
            R = jnp.linalg.inv(L).T
            eigs, L, R = jnp.real(eigs), jnp.real(L), jnp.real(R)
                    
            p = jnp.array([params.T @ jnp.outer(L[:, j], R[:, j]) @ params for j in range(d)])
            u = jnp.array([params.T @ jnp.outer(L[:, j], R[:, j]) @ optimal_params for j in range(d)])            
            q = jnp.array([optimal_params.T @ jnp.outer(L[:, j], R[:, j]) @ optimal_params for j in range(d)])
                        
            
        return jnp.concatenate([p,u,q]), eigs, None
    
    @jit
    def update_odes(self, y, eigs, B, lr, subkey, extra):
        d = len(eigs)
        subkey_mean, subkey_cov = jax.random.split(subkey)
        p, u, q = y[:d,:,:], y[d:2*d,:,:], y[2*d:,:,:]
        
        I = compute_I(B, self.f, subkey_cov)
        H = compute_H(B, self.f, subkey_mean)
        m = len(B) // 2
        
        p_update = -2*lr * eigs[:,None,None] * (p * H[0:m] + u * H[m:]) + eigs[:,None,None] * lr**2 * I / d
        u_update = -lr * eigs[:,None,None] * (H[0:m] * u + H[m:] * q)                
                
        return jnp.concatenate([p_update, u_update, jnp.zeros(u_update.shape)])
    