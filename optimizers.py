from risks_and_discounts import *
    
class Optimizer:
    def __init__(self, problem, key=None):
        self.problem = problem
        self.key = key if key is not None else jax.random.PRNGKey(np.random.randint(0, 10000))        

        if problem == 'logreg':
            self.grad = grad_logreg
            self.get_target = logreg_target
            self.get_target = logreg_target
            self.risk_fun = risk_from_B_logreg
        elif problem == 'linreg':
            self.grad = grad_linreg
            self.get_target = linreg_target
            self.risk_fun = risk_from_B_linreg

    def init_state(self, d):
        return None
                                        
    def update(self):
        pass

    def run(self, params, cov, T, lr_fun, optimal_params, **kwargs):
        risks = []        
        d = len(cov)        
        key = self.key
        
        state = self.init_state(d)
        B = make_B(params, optimal_params, cov)
        for k in tqdm(range(T * d)):
            if callable(lr_fun):
                lr = lr_fun(k)
            params, key, state = self.update(params, lr, cov, optimal_params, key, state, **kwargs)
            risks.append(self.risk_fun(B))
            B = make_B(params, optimal_params, cov)

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
    
    
@jax.tree_util.register_pytree_node_class
class Adam(Optimizer):   
    def init_state(self, d):
        return (jnp.zeros(d), jnp.zeros(d))
        
    @jit
    def update(self, params, lr, cov, optimal_params, key, state, beta1, beta2, eps = 0):
        m, v = state
        key, subkey = jax.random.split(key)
        data = make_data(cov, subkey)
        target = self.get_target(data, optimal_params)

        gradient = self.grad(params, data, target)
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2

        m_hat = m  # Bias-corrected first moment
        v_hat = v  # Bias-corrected second moment

        params = params - lr * m_hat / (jnp.sqrt(v_hat + eps))
        
        return params, key, (m, v)


@jax.tree_util.register_pytree_node_class
class SGD(Optimizer):   
                 
    @jit
    def update(self, params, lr, cov, optimal_params, key, state):
        key, subkey = jax.random.split(key)
        data = make_data(cov, subkey)
        target = self.get_target(data, optimal_params)

        gradient = self.grad(params, data, target)       
        params = params - lr * gradient
        
        return params, key, None





# @jax.tree_util.register_pytree_node_class
# class Adam:
#     def __init__(self, problem, key=None):
#         self.problem = problem        
#         self.key = key if key is not None else jax.random.PRNGKey(np.random.randint(0, 10000))        

#         if problem == 'logreg':
#             self.grad = grad_logreg
#             self.get_target = logreg_target
#             self.get_target = logreg_target
#             self.risk_fun = risk_from_B_logreg
#         elif problem == 'linreg':
#             self.grad = grad_linreg
#             self.get_target = linreg_target
#             self.risk_fun = risk_from_B_linreg
                 
#     @jit
#     def update(self, params, m, v, lr, beta1, beta2, cov, optimal_params, eps, key):        
#         key, subkey = jax.random.split(key)
#         data = make_data(cov, subkey)
#         target = self.get_target(data, optimal_params)

#         gradient = self.grad(params, data, target)
#         m = beta1 * m + (1 - beta1) * gradient
#         v = beta2 * v + (1 - beta2) * gradient**2

#         m_hat = m  # Bias-corrected first moment
#         v_hat = v  # Bias-corrected second moment

#         params = params - lr * m_hat / (jnp.sqrt(v_hat + eps))
        
#         return params, m, v, key

#     def run(self, params, cov, T, lr, beta1, beta2, optimal_params, eps=0):                
#         risks = []        
#         d = len(cov)        
#         key = self.key
        
#         m, v = jnp.zeros(d), jnp.zeros(d)
#         B = make_B(params, optimal_params, cov)
#         for _ in tqdm(range(T * d)):
#             params, m, v, key = self.update(params, m, v, lr, beta1, beta2, cov, optimal_params, eps, key)
#             risks.append(self.risk_fun(B))
#             B = make_B(params, optimal_params, cov)

#         return params, risks

#     def tree_flatten(self):
#         """Flatten the PyTree into a tuple of arrays and auxiliary data."""
#         leaves = (self.key)
#         aux_data = (self.problem)
#         return leaves, aux_data

#     @classmethod
#     def tree_unflatten(cls, aux_data, leaves):
#         """Reconstruct the PyTree from its flattened representation."""
#         key = leaves
#         problem = aux_data
#         return cls(problem, key=key)
