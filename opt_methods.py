import numpy as np
import jax
import jax.numpy as jnp

from jax import jit
from functools import partial

global dt
dt = 0.01

####################################### UTILS #######################################

@jit
def risk(K, x, xstar):
    return (x-xstar).transpose() @ K @ (x-xstar) / 2

####################################### Adam #######################################

## Plain ol' Adam

def one_pass_adam(quadratic, grad_function, K, data, targets, params0, optimal_params, lrk, beta1, beta2):
        
    def update(carry, idx):
        params, m, v, quad_vals, step, Vs = carry
        data_point, target = data[idx], targets[idx]

        # Compute gradient
        grad = grad_function(params, data_point, target)

        # Update first moment estimate
        m = beta1 * m + (1 - beta1) * grad

        # Update second moment estimate
        v = beta2 * v + (1 - beta2) * grad**2

        # Bias correction
        # mhat = m / (1 - beta1 ** (step + 1))
        # vhat = v / (1 - beta2 ** (step + 1))
        
        ## This is the truth
        mhat = m
        # vhat = m
        
        ## This is to test things
        vhat = beta2 * jnp.diag(K)  + (1 - beta2) * grad**2
        # vhat = 2 * beta2 * Pbeta * jnp.diag(K)  + (1 - beta2) * grad**2 

        # Compute update step
        eps = 0
        step_update = mhat / (jnp.sqrt(vhat) + eps)

        # Update params
        params = params - lrk * step_update

        # Update Pbeta
        Q = quadratic(K,params,optimal_params)
        
        
        # Pbeta = beta2 * Pbeta + (1 - beta2) * Q
        # Pbetas = Pbetas.at[step].set(Pbeta)   
        
            
        Vs = Vs.at[step].set(vhat)                
        quad_vals = quad_vals.at[step].set(Q)
        
        
        return (params, m, v, quad_vals, step + 1, Vs), None

    # Preallocate arrays for risk_vals, times, Pbetas, and Vs
    max_steps = len(data)
    quad_vals = jnp.zeros(max_steps)
    times = jnp.arange(max_steps)    
    Vs = jnp.zeros((len(data), K.shape[0]))

    # Initialize variables
    d = K.shape[0]
    m = jnp.zeros(d)
    v = jnp.zeros(d)
    
    carry = (params0, m, v, quad_vals, 0, Vs)

    # Use JAX lax.scan for iteration
    carry, _ = jax.lax.scan(update, carry, times)

    params, m, v, quad_vals, step, Vs = carry    

    return quad_vals, times


## SDE(s) for Adam

@partial(jax.jit, static_argnames=['f'])
def adam_mean_from_params(params, optimal_params, f, K, Kbar, beta, fk, key, n_samples = 10000):
    """
    Compute the mean of the adam update in terms of parameters 
    TODO: Document this better    
    """
        
    B = jnp.array([[params.T @ K @ params, optimal_params.T @ K @ params],[optimal_params.T @ K @ params, optimal_params.T @ K @ optimal_params]])
    Binv = jnp.linalg.inv(B)
    key_Q, key_z = jax.random.split(key)
    
    
    Q = jax.random.multivariate_normal(key_Q, mean = np.zeros(2), cov = B, shape=n_samples)
    z = jax.random.normal(key_z, (n_samples,))
    
    phi = jnp.mean((z**2 * f(Q) / jnp.sqrt(beta*fk + (1-beta) * f(Q)**2 * z**2))[:,None]  *  (Q @ Binv), axis = 0)
    
    
    return Kbar @ params * phi[0] + Kbar @ optimal_params * phi[1]
    
@partial(jax.jit, static_argnames=['f'])
def adam_cov_from_params_diag(params, optimal_params, f, K, beta, fk, key, n_samples = 10000):
    """
    Compute the mean of the adam update in terms of parameters 
    TODO: Document this better    
    """
    B = jnp.array([[params.T @ K @ params, optimal_params.T @ K @ params],[optimal_params.T @ K @ params, optimal_params.T @ K @ optimal_params]])        
    
    key_Q, key_z = jax.random.split(key)    
    Q = jax.random.multivariate_normal(key_Q, mean = np.zeros(2), cov = B, shape=n_samples)
    z = jax.random.normal(key_z, (n_samples,))
    
    fq = f(Q)
    
    return jnp.mean(z**2 * fq**2 / (beta*fk + (1-beta) * fq**2 * z**2))
        
        
def adam_sde_diag(quadratic, f,  T, lr, K, Kbar, beta, params0, optimal_params, d, key = None):
    dt = 0.001
    sqrtdt = np.sqrt(dt)
    
    fk = 1
    steps = int(T / dt)
    params = params0
    Q = quadratic(K, params, optimal_params)
    quad_vals = [Q]
    if key is None:
        key = jax.random.PRNGKey(0)
            
    key_mean, key_cov, key_brownian = jax.random.split(key, 3)
            
    for _ in range(steps):
        # W = np.random.randn(d) * sqrtdt       
        key_brownian, subkey_brownian = jax.random.split(key_brownian)
        W = jax.random.normal(subkey_brownian, (d,)) * sqrtdt

        key_mean, subkey_mean = jax.random.split(key_mean)
        key_cov, subkey_cov = jax.random.split(key_cov)
        mean = adam_mean_from_params(params, optimal_params, f, K, Kbar, beta, fk, subkey_mean)
        cov = adam_cov_from_params_diag(params, optimal_params, f, K, beta, fk, subkey_cov)
                                        
        params = params - lr * mean * dt + lr * jnp.sqrt(cov) * W / jnp.sqrt(d)
                
        Q = quadratic(K, params, optimal_params)        
        quad_vals.append(Q)
    
    times = jnp.array(range(steps + 1)) * dt
    return jnp.array(quad_vals), times


####################################### Vanilla SGD (Needs updating) #######################################

def vanilla_linreg_ode(K,T, noise_std, x0, xstar, lr, diag_preconditioner = None):
    d = len(x0)

    vals, vecs = np.linalg.eigh(K)
    v = ((x0-xstar) @ vecs)**2 / 2
    R = np.dot(vals,v)
    risks = []

    ode_time = []
    iters = int(T / dt)
    for i in range(iters + 1):
        t = i * dt
        R = np.dot(vals,v)
        
        if diag_preconditioner is not None:
            update = -lr * 2 * v * vals * diag_preconditioner + lr**2 * (vals * (2*R+ noise_std**2)) / (2 * d)        
        else:
            update = -lr * 2 * v * vals + lr**2 * (vals * (2*R+ noise_std**2)) / (2 * d)        
        
        v = v + dt * update

        ode_time.append(t)
        risks.append(R)
    return np.array(risks), np.array(ode_time)

def one_pass_sgd(K, data, targets, x0, xstar, lrk, diag_preconditioner = None):
    x = x0
    risk_vals = []
    times = []

    d = K.shape[0]    
    sched = (np.size(lrk) > 1) or callable(lrk)
    
    if not sched:
        lr = lrk

    for i,(a,b) in enumerate(zip(data,targets)):
        if i % 20 == 0:
            times.append(i)
            risk_vals.append(risk(K, x,xstar))

        grad = (np.dot(x,a) - b) * a        
        
        if diag_preconditioner is not None:
            grad = diag_preconditioner * grad
                
        if sched:
            if callable(lrk):
                R = risk(K,x,xstar)
                lr = lrk(R)
            else:
                lr = lrk[int(np.round(i / d / dt))]         
        
        x = x - lr * grad

    times.append(i+1)
    risk_vals.append(risk(K, x,xstar))
    
    return np.array(risk_vals), np.array(times)




