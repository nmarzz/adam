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
        elif problem == 'lip_phaseret':
            self.risk_fun = risk_from_B_lip_phase_retrieval
            self.f = f_lip_phase_ret
        elif problem == 'real_phaseret':
            self.risk_fun = risk_from_B_real_phase_retrieval
            self.f = f_real_phase_ret
    
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
        p,u,v,q = y[:d], y[d:2*d], y[2*d:3*d], y[3*d:]
        B11 = jnp.sum(p)
        B12 = jnp.sum(u)
        B21 = jnp.sum(v)
        B22 = jnp.sum(q)
        return jnp.array([[B11,B12],[B21,B22]])
    
    def init_odes(self, cov, params, optimal_params):                
        d = len(cov)
        if len(cov.shape) == 1: # diagonal covariance            
            eigs = cov / jnp.sqrt(cov)
            L = jnp.eye(d)
            R = jnp.eye(d)
                    
            var_force = jnp.array([jnp.inner(R[:, j], cov * L[:, j]) for j in range(d)])
            p = jnp.array([jnp.inner(params, cov * L[:, j]) * jnp.inner(R[:, j], params) for j in range(d)])
            u = jnp.array([jnp.inner(params, cov * L[:, j]) * jnp.inner(R[:, j], optimal_params) for j in range(d)])
            v = jnp.array([jnp.inner(optimal_params, cov * L[:, j]) * jnp.inner(R[:, j], params) for j in range(d)])
            q = jnp.array([jnp.inner(optimal_params, cov * L[:, j]) * jnp.inner(R[:, j], optimal_params) for j in range(d)])

        else:
            covbar = cov / jnp.sqrt(jnp.diag(cov))
            eigs, L = jnp.linalg.eig(covbar)
            R = jnp.linalg.inv(L).T
            eigs, L, R = jnp.real(eigs), jnp.real(L), jnp.real(R)
        
        
            var_force = jnp.array([jnp.inner(R[:, j], cov @ L[:, j]) for j in range(d)])
            p = jnp.array([jnp.inner(params, cov @ L[:,j]) * jnp.inner(R[:,j], params) for j in range(d)])
            u = jnp.array([jnp.inner(params, cov @ L[:,j]) * jnp.inner(R[:,j], optimal_params) for j in range(d)])
            v = jnp.array([jnp.inner(optimal_params, cov @ L[:,j]) * jnp.inner(R[:,j], params) for j in range(d)])
            q = jnp.array([jnp.inner(optimal_params, cov @ L[:,j]) * jnp.inner(R[:,j], optimal_params) for j in range(d)])
            

        return jnp.concatenate([p, u, v, q]), eigs, var_force
    
    @jit    
    def update_odes(self, y, eigs, B, lr, subkey, extra, **kwargs):
        var_force = extra
        d = len(eigs)
        p, u, v, q = y[:d], y[d:2*d], y[2*d:3*d], y[3*d:]
        subkey_mean, subkey_cov = jax.random.split(subkey)
        beta2 = kwargs['beta2']
        
        phi1_B, phi2_B = phi_from_B(B, self.f, beta2, subkey_mean)
        
        p_update = -lr * eigs * (2 * p * phi1_B + phi2_B * (u + v))
        p_update += lr**2 * cov_from_B(B, self.f, beta2, subkey_cov) * var_force / d
        
        u_update = -lr * eigs * (phi1_B * u + phi2_B * q)
        v_update = -lr * eigs * (phi1_B * v + phi2_B * q)
                
        return jnp.concatenate([p_update, u_update, v_update, jnp.zeros(d)])
    

@jax.tree_util.register_pytree_node_class
class SgdODE(ODE):
    
    @staticmethod
    @jit
    def make_B(y, eigs):
        d = len(eigs)
        p,u,q = y[:d], y[d:2*d], y[2*d:]
        B11 = jnp.inner(p, eigs)
        B12 = jnp.inner(u, eigs)
        B22 = jnp.inner(q, eigs)
        return jnp.array([[B11,B12],[B12,B22]])
    
    def init_odes(self, cov, params, optimal_params):                
        d = len(cov)
        if len(cov.shape) == 1: # diagonal covariance            
            eigs = cov
            L = jnp.eye(d)
            R = jnp.eye(d)
                                
            p = jnp.array([jnp.inner(params, L[:, j]) * jnp.inner(R[:, j], params) for j in range(d)])
            u = jnp.array([jnp.inner(params, L[:, j]) * jnp.inner(R[:, j], optimal_params) for j in range(d)])            
            q = jnp.array([jnp.inner(optimal_params, L[:, j]) * jnp.inner(R[:, j], optimal_params) for j in range(d)])
            
        else:
            raise NotImplementedError('SGD ODEs not implemented for non-diagonal covariance')
            

        return jnp.concatenate([p,u,q]), eigs, None
    
    @jit
    def update_odes(self, y, eigs, B, lr, subkey, extra):
        d = len(eigs)
        subkey_mean, subkey_cov = jax.random.split(subkey)
        p, u, q, = y[:d], y[d:2*d], y[2*d:]
        
        I = compute_I(B, self.f, subkey_cov)
        H = compute_H(B, self.f, subkey_mean)
                
        p_update = -2*lr * eigs * (p * H[0] + u * H[1]) + eigs * lr**2 * I / d                                
        u_update = -lr * eigs * (H[0] * u + H[1] * q)
                
        return jnp.concatenate([p_update,u_update,jnp.zeros(d)])
    