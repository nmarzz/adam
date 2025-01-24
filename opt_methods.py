import numpy as np
import jax
import jax.numpy as jnp

from jax import jit
from functools import partial

# global dt
# dt = 0.01

####################################### UTILS #######################################

def make_B(K, params, optimal_params):
    B11 = jnp.dot(params, K @ params)
    B12 = jnp.dot(optimal_params, K @ params)
    B22 = jnp.dot(optimal_params, K @ optimal_params)
    return jnp.array([[B11,B12],[B12, B22]])

# @jit
# def risk(K, x, xstar):
#     return (x-xstar).transpose() @ K @ (x-xstar) / 2

####################################### Adam #######################################

def one_pass_adam(risk, grad_function, K, data, targets, params0, optimal_params, lrk, beta1, beta2):
        
    def update(carry, idx):
        params, m, v, key_risk, risk_vals, step, Bs = carry
        data_point, target = data[idx], targets[idx]
        
        key_risk, subkey_risk = jax.random.split(key_risk)
        B = make_B(K, params, optimal_params)
        Q = risk(B, subkey_risk)
        Bs = Bs.at[step].set(B)
        risk_vals = risk_vals.at[step].set(Q)
        
        # Compute gradient
        grad = grad_function(params, data_point, target)

        # Update moment estimates
        m = beta1 * m + (1 - beta1) * grad        
        v = beta2 * v + (1 - beta2) * grad**2

        # Bias correction
        # mhat = m / (1 - beta1 ** (step + 1))
        # vhat = v / (1 - beta2 ** (step + 1))
        
        ## This is the truth
        mhat = m
        vhat = v ## This is the truth
        
        # Compute update step
        eps = 0
        step_update = mhat / (jnp.sqrt(vhat) + eps)

        # Update params
        params = params - lrk * step_update
                
        return (params, m, v, key_risk, risk_vals, step + 1, Bs), None

    # Preallocate arrays for risk_vals, times, Pbetas, and Vs
    max_steps = len(data)
    risk_vals = jnp.zeros(max_steps)
    times = jnp.arange(max_steps)    
    Bs = jnp.zeros((len(data), 2,2))

    # Initialize variables
    d = K.shape[0]
    m = jnp.zeros(d)
    v = jnp.zeros(d)
    key_risk = jax.random.PRNGKey(0)
    
    carry = (params0, m, v, key_risk, risk_vals, 0, Bs)

    # Use JAX lax.scan for iteration
    carry, _ = jax.lax.scan(update, carry, times)

    _, m, v, key_risk, risk_vals, _, Bs = carry

    return risk_vals, times, Bs





## SDE(s) for Adam

@partial(jax.jit, static_argnames=['f'])
def adam_mean_from_params(params, optimal_params, f, K, Kbar, beta,  key, n_samples = 10000):
    """
    Compute the mean of the adam update in terms of parameters 
    TODO: Document this better    
    """
        
    B = jnp.array([[params.T @ K @ params, optimal_params.T @ K @ params],[optimal_params.T @ K @ params, optimal_params.T @ K @ optimal_params]])
    Binv = jnp.linalg.inv(B)
    key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)


    history_length = 50
    # Q = jax.random.multivariate_normal(key_Q, mean = np.zeros(2), cov = B, shape=(n_samples, history_length))
    Q = jax.random.multivariate_normal(key_Q, mean = np.zeros(2), cov = B, shape=(n_samples, 1))
    z = jax.random.normal(key_z, (n_samples))

    
    Q_history = jax.random.multivariate_normal(key_Q_hist, mean = np.zeros(2), cov = B, shape=(n_samples, history_length))
    z_history = jax.random.normal(key_z_hist, (n_samples, history_length))

    decay_vec = jnp.array([beta ** i for i in range(1, history_length + 1)])
    history_average = (1-beta)*(f(Q_history)**2 * z_history**2) @ decay_vec

    fq = f(Q)
    fq = fq.squeeze()
    Q = Q.squeeze()
        
    phi = jnp.mean((z**2 * fq / jnp.sqrt( history_average + (1-beta) * fq**2 * z**2))[:,None]  *  (Q @ Binv), axis = 0)
        
    return Kbar @ params * phi[0] + Kbar @ optimal_params * phi[1]
    

@partial(jax.jit, static_argnames=['f'])
def adam_cov_from_params_diag(params, optimal_params, f, K, beta, key, n_samples = 10000):
    """
    Compute the mean of the adam update in terms of parameters 
    TODO: Document this better
    """
    B = jnp.array([[params.T @ K @ params, optimal_params.T @ K @ params],[optimal_params.T @ K @ params, optimal_params.T @ K @ optimal_params]])    
    key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)


    history_length = 50
    # Q = jax.random.multivariate_normal(key_Q, mean = np.zeros(2), cov = B, shape=(n_samples, history_length))
    Q = jax.random.multivariate_normal(key_Q, mean = np.zeros(2), cov = B, shape=(n_samples, 1))
    z = jax.random.normal(key_z, (n_samples))
    
    Q_history = jax.random.multivariate_normal(key_Q_hist, mean = np.zeros(2), cov = B, shape=(n_samples, history_length))
    z_history = jax.random.normal(key_z_hist, (n_samples, history_length))

    decay_vec = jnp.array([beta ** i for i in range(1, history_length + 1)])
    history_average = (1-beta)*(f(Q_history)**2 * z_history**2) @ decay_vec    

    fq = f(Q)
    fq = fq.squeeze()        
    
    return jnp.mean(z**2 * fq**2 / (history_average + (1-beta) * fq**2 * z**2))
        
        
def adam_sde_diag(risk, f,  T, lr, K, Kbar, beta, params0, optimal_params, d, key = None):
    # dt = 0.001
    dt = 1/d
    sqrtdt = np.sqrt(dt)
            
    steps = int(T / dt)
    params = params0
    if key is None:
        key = jax.random.PRNGKey(0)
            
    key_mean, key_cov, key_brownian, key_risk = jax.random.split(key, 4)
    B = make_B(K,params, optimal_params)
    Q = risk(B, key_risk)
    risk_vals = [Q]
            
    for _ in range(steps):
        # W = np.random.randn(d) * sqrtdt       
        key_brownian, subkey_brownian = jax.random.split(key_brownian)
        W = jax.random.normal(subkey_brownian, (d,)) * sqrtdt

        key_mean, subkey_mean = jax.random.split(key_mean)
        key_cov, subkey_cov = jax.random.split(key_cov)        
        key_risk, subkey_risk = jax.random.split(key_risk)        
                        
        mean = adam_mean_from_params(params, optimal_params, f, K, Kbar, beta, subkey_mean)
        cov = adam_cov_from_params_diag(params, optimal_params, f, K, beta, subkey_cov)
                                                        
        params = params - lr * mean * dt + lr * jnp.sqrt(cov) * W / jnp.sqrt(d)
                
        B = make_B(K,params, optimal_params)
        Q = risk(B, subkey_risk)
        risk_vals.append(Q)
    
    times = jnp.array(range(steps + 1)) * dt
    return jnp.array(risk_vals), times

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


def phi_from_B(B, f, beta,  key, n_samples = 10000):
    """
    Compute the mean of the adam update in terms of B
    TODO: Document this better    
    """
    
    Binv = jnp.linalg.inv(B)
    key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)

    history_length = 50    
    Q = jax.random.multivariate_normal(key_Q, mean = np.zeros(2), cov = B, shape=(n_samples, 1))
    z = jax.random.normal(key_z, (n_samples))

    if beta > 0:
        Q_history = jax.random.multivariate_normal(key_Q_hist, mean = np.zeros(2), cov = B, shape=(n_samples, history_length))
        z_history = jax.random.normal(key_z_hist, (n_samples, history_length))

        decay_vec = jnp.array([beta ** i for i in range(1, history_length + 1)])
        history_average = (1-beta)*(f(Q_history)**2 * z_history**2) @ decay_vec
    else:
        history_average = 0

    fq = f(Q)
    fq = fq.squeeze()
    Q = Q.squeeze()
        
    phi = jnp.mean((z**2 * fq / jnp.sqrt( history_average + (1-beta) * fq**2 * z**2))[:,None]  *  (Q @ Binv), axis = 0)
        
    return phi


def cov_from_B(B, f, beta,  key, n_samples = 10000):
    """
    Compute the covariance update the adam update in terms of B
    TODO: Document this better    
    """
        
    key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)

    history_length = 50    
    Q = jax.random.multivariate_normal(key_Q, mean = np.zeros(2), cov = B, shape=(n_samples, 1))
    z = jax.random.normal(key_z, (n_samples))

    if beta > 0:        
        Q_history = jax.random.multivariate_normal(key_Q_hist, mean = np.zeros(2), cov = B, shape=(n_samples, history_length))
        z_history = jax.random.normal(key_z_hist, (n_samples, history_length))

        decay_vec = jnp.array([beta ** i for i in range(1, history_length + 1)])
        history_average = (1-beta)*(f(Q_history)**2 * z_history**2) @ decay_vec
    else:
        history_average = 0

    fq = f(Q)
    fq = fq.squeeze()        
    
    return jnp.mean(z**2 * fq**2 / (history_average + (1-beta) * fq**2 * z**2))

def adam_ode(K, Kbar, T, params0, optimal_params, lr, beta, f, risk_from_B):
    print('Precomputations...')

    # Perform the diagonalization
    d = K.shape[0]

    eigs, L = jnp.linalg.eig(Kbar)
    R = jnp.linalg.inv(L).T
    eigs, L, R = jnp.real(eigs), jnp.real(L), jnp.real(R)

    # Setup the ODE systems 
    var_force = jnp.array([jnp.inner(R[:, j], K @ L[:, j]) for j in range(len(K))])
    p = jnp.array([jnp.inner(params0, K @ L[:,j]) * jnp.inner(R[:,j], params0) for j in range(d)])
    u = jnp.array([jnp.inner(params0, K @ L[:,j]) * jnp.inner(R[:,j], optimal_params) for j in range(d)])
    v = jnp.array([jnp.inner(optimal_params, K @ L[:,j]) * jnp.inner(R[:,j], params0) for j in range(d)])
    q = jnp.array([jnp.inner(optimal_params, K @ L[:,j]) * jnp.inner(R[:,j], optimal_params) for j in range(d)])

        
    B11 = jnp.sum(p)
    B12 = jnp.sum(u)
    B21 = jnp.sum(v)
    B22 = jnp.sum(q)
    
    
    # Setup jax rng
    key = jax.random.PRNGKey(0)
    key_mean, key_cov, key_risk = jax.random.split(key, 3)

    risks = []
    dt = 1/d

    ode_time = []

    iters = int(T / dt)
    Bs = []
    print('Iterating...')
    for i in range(iters):
        key_mean, subkey_mean = jax.random.split(key_mean)
        key_cov, subkey_cov = jax.random.split(key_cov)
        key_risk, subkey_risk = jax.random.split(key_risk)
        
        t = i * dt
        ode_time.append(t)
        
        
        
        B11 = jnp.sum(p)
        B12 = jnp.sum(u)
        B21 = jnp.sum(v)
        B22 = jnp.sum(q)
        B = jnp.array([[B11,B12],[B21,B22]])
        
        R = risk_from_B(B, subkey_risk)
                
        risks.append(R)
        Bs.append(B)
                        
        
        phi1_B, phi2_B = phi_from_B(B, f, beta, subkey_mean)
        
        
        p_update = -lr * eigs * (2* p* phi1_B + phi2_B * (u + v))
        p_update += lr**2 * cov_from_B(B, f, beta, subkey_cov) * var_force / d
        # p_update += lr**2 * 1 * var_force / d        
                
        u_update = -lr * eigs * (phi1_B * u + phi2_B * q)
        v_update = -lr * eigs * (phi1_B * v + phi2_B * q)
                        
        p = p + dt * p_update
        u = u + dt * u_update
        v = v + dt * v_update


    return jnp.array(risks), jnp.array(ode_time), jnp.array(Bs)

