from tqdm import tqdm
from utils import make_data, make_B
from risks_and_discounts import *
from functools import partial
    
class Optimizer:
    def __init__(self, problem, key=None):
        self.problem = problem        
        self.key = key if key is not None else jax.random.PRNGKey(np.random.randint(0, 10000))        
        
        if problem == 'linreg':
            self.grad = grad_linreg
            self.get_target = linreg_target
            self.risk_fun = risk_from_B_linreg   
        elif problem == 'logreg':
            self.grad = grad_logreg
            self.get_target = logreg_target
            self.risk_fun = risk_from_B_logreg
        elif problem == 'real_phase_ret':
            self.grad = grad_real_phase_ret
            self.get_target = real_phase_ret_target
            self.risk_fun = risk_from_B_real_phase_ret
        else:
            raise NotImplementedError(f'Problem {problem} not implemented')

    def init_state(self, d, num_classes):
        return None
                                        
    def update(self):
        pass
        
    def run(self, params, cov, T, lr_fun, optimal_params, **kwargs):
        risks = []        
        d, num_classes = params.shape
        key = self.key
        
        state = self.init_state(d,num_classes)
        B = make_B(params, optimal_params, cov)
        for k in tqdm(range(T * d)):
            if callable(lr_fun):
                lr = lr_fun(k)
            else:
                lr = lr_fun
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
    def init_state(self, d, num_classes):
        return (jnp.zeros((d,num_classes)), jnp.zeros((d,num_classes)))
        
    @jit
    def update(self, params, lr, cov, optimal_params, key, state, beta1, beta2, eps):
        # print(f'I am beta1 in adam: {beta1}')
        m, v = state
        key, subkey = jax.random.split(key)
        data = make_data(cov, subkey)
        target = self.get_target(data, optimal_params)

        gradient = self.grad(params, data, target)
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2
        
        m_hat = m  # Bias-corrected first moment
        v_hat = v  # Bias-corrected second moment

        params = params - lr * m_hat / jnp.sqrt(v_hat + eps)
        
        return params, key, (m, v)
    
@jax.tree_util.register_pytree_node_class
class BlockAdam(Optimizer):        
    def init_state(self, d, num_classes):        
        # returns m0, v0, last_block_params, step_count = 0 
        return (jnp.zeros((d,num_classes)), jnp.zeros((d,num_classes)), jnp.zeros((d,num_classes)), 0)
            
    # @jit
    # def update(self, params, lr, cov, optimal_params, key, state, beta1, beta2, eps):
    #     M = 100
    #     m, v, last_block_params, step = state        
        
    #     if step == 1:
    #         last_block_params = params
            
    #     key, subkey = jax.random.split(key)
    #     data = make_data(cov, subkey)
    #     target = self.get_target(data, optimal_params)

                
    #     gradient = self.grad(last_block_params, data, target)
    #     not_equal_block = (step % M != 0)

    #     m = beta1 * m * not_equal_block + (1 - beta1) * gradient
    #     v = beta2 * v * not_equal_block + (1 - beta2) * gradient**2

    #     m_hat = m  # Bias-corrected first moment
    #     v_hat = v  # Bias-corrected second moment

    #     params = params - lr * m_hat / jnp.sqrt(v_hat + eps)
                
    #     if step % M == 0:
    #         last_block_params = params

    #     return params, key, (m, v, last_block_params)


    @partial(jax.jit, static_argnums=0)
    def update(self, params, lr, cov, optimal_params, key, state, beta1, beta2, eps):
        M = 40
        m, v, last_block_params, step = state

        key, subkey = jax.random.split(key)
        data = make_data(cov, subkey)
        target = self.get_target(data, optimal_params)

        # Pre-grad reset at very first step
        pre_reset = (step == 1)
        last_block_params = jax.tree.map(
            lambda p, lbp: jnp.where(pre_reset, p, lbp), params, last_block_params
        )

        g = self.grad(last_block_params, data, target)

        in_block = (step % M != 0)
        m = jax.tree.map(
            lambda m_, g_: jnp.where(in_block, beta1*m_ + (1-beta1)*g_, (1-beta1)*g_), m, g
        )
        v = jax.tree.map(
            lambda v_, g_: jnp.where(in_block, beta2*v_ + (1-beta2)*(g_**2), (1-beta2)*(g_**2)), v, g
        )

        params = jax.tree.map(
            lambda p, mh, vh: p - lr * mh / jnp.sqrt(vh + eps), params, m, v
        )

        # End-of-block reset
        post_reset = (step % M == 0)
        last_block_params = jax.tree.map(
            lambda p, lbp: jnp.where(post_reset, p, lbp), params, last_block_params
        )

        step = step + 1
        return params, key, (m, v, last_block_params, step)


@jax.tree_util.register_pytree_node_class
class ResampledAdam(Optimizer):
          
    @jit
    def update(self, params, lr, cov, optimal_params, key, state, beta1, beta2, eps = 0):
         
        history_length = 15
        d_vec1 = jnp.array([beta1**i for i in range(0, history_length)]) * (1-beta1)
        d_vec2 = jnp.array([beta2**i for i in range(0, history_length)]) * (1-beta2)

        key, subkey = jax.random.split(key)
        data = make_data(cov, subkey)
        target = self.get_target(data, optimal_params)
        current_grad = self.grad(params, data, target)


        # TODO: make this more efficient, 'tis very bad
        second_mnts = []
        for l in range(history_length):
            gradients = []
            for _ in range(history_length):
                key, subkey = jax.random.split(key)
                data = make_data(cov, subkey)
                target = self.get_target(data, optimal_params)

                gradient = self.grad(params, data, target)
                gradients.append(gradient)
            gradients = jnp.array(gradients)
            gradients = gradients.at[l,:,:].set(current_grad)
            gradients2 = gradients**2
            
            second_mnt = jnp.sqrt(jnp.einsum('i,ijk->jk', d_vec2, gradients2))
            second_mnts.append(second_mnt)
        second_mnts = jnp.array(second_mnts)    

        update = jnp.einsum('i,ijk->jk', d_vec1, (current_grad / second_mnts))

        params = params - lr * update
        
        return params, key, state
    


@jax.tree_util.register_pytree_node_class
class SGD(Optimizer):   
                 
    @jit
    def update(self, params, lr, cov, optimal_params, key, state):
        key, subkey = jax.random.split(key)
        data = make_data(cov, subkey)        
        target = self.get_target(data, optimal_params)

        gradient = self.grad(params, data, target)       
        params = params - lr * gradient
        
        return params, key, state
