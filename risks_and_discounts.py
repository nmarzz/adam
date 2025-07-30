import jax
import numpy as np
from numpy.polynomial.hermite import hermgauss
from jax import jit
import jax.numpy as jnp
from functools import partial
from jax.nn import sigmoid


@jit
def logreg(data, params):
    logits = params.T @ data
    probabilities = sigmoid(logits)
    return probabilities

@jit
def grad_logreg(params, data, target):
    p_student = logreg(data, params)    
    return jnp.outer(data, p_student - target)

@jit
def grad_linreg(params, data, target):    
    return jnp.outer(data, data @ params - target)

@jit
def grad_real_phase_ret(params, data, target):
    return jnp.outer(data, 4 * (params.T @ data) * ((params.T @ data)**2 - target))

@jit
def linreg_target(optimal_params, data):    
    return optimal_params.T @ data

@jit
def logreg_target(optimal_params, data):
    return sigmoid(optimal_params.T @ data)

@jit
def real_phase_ret_target(optimal_params, data):
    return (optimal_params.T @ data)**2


@jit
def risk_from_B_linreg(B):
    return (jnp.trace(B) - 2 * jnp.trace(B[0:B.shape[0]//2, B.shape[0]//2:]))/2

@jit
def risk_from_B_logreg(B):
    if len(B) > 2:
        raise NotImplementedError("Only binary logistic regression is implemented")    
    B00 = B[0, 0]
    B11 = B[1, 1]
    B01 = B[0, 1]
    
    num_points = 20
    points, weights = hermgauss(num_points)  # Nodes and weights for Hermite quadrature
    points *= jnp.sqrt(2)
    
    t1 = -B01 * jnp.exp(jnp.sqrt(B11) * points) / (1 + jnp.exp(jnp.sqrt(B11) * points))**2
    t2 = jnp.log(1 + jnp.exp(jnp.sqrt(B00) * points))
        
    return jnp.sum(weights * (t1 + t2)) / jnp.sqrt(jnp.pi)

@jit
def risk_from_B_real_phase_ret(B):
    m = len(B)//2
    B11 = B[0:m,0:m]
    B12 = B[0:m,m:]
    B22 = B[m:,m:]
    return 3*jnp.trace(B11@B11) -2 * jnp.trace(B11 @ B22) -4*jnp.trace(B12 @ B12.T) + 3 * jnp.trace(B22@B22)


# @partial(jax.jit, static_argnames=['m'])
@jit
def f_linreg(q):
    m = q.shape[2] // 2
    q1 = q[:,:,0:m]
    q2 = q[:,:,m:]
    return q1 - q2

@jit
def f_logreg(q):
    m = q.shape[2] // 2
    q1 = q[:,:,0:m]
    q2 = q[:,:,m:]
    return sigmoid(q1) - sigmoid(q2)

@jit
def f_real_phase_ret(q):
    m = q.shape[2] // 2
    q1 = q[:,:,0:m]
    q2 = q[:,:,m:]
    return 4 * q1 * ((q1)**2 - (q2)**2)



# @partial(jax.jit, static_argnames=['f'])
# def phi_from_B(B, f, beta1, beta2,  key, n_samples = 10000):
#     key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)
#     Binv = jnp.linalg.inv(B)

#     history_length = 250
#     Q = jax.random.multivariate_normal(key_Q, mean = jnp.zeros(len(B)), cov = B, shape=(n_samples, 1))
#     z = jax.random.normal(key_z, (n_samples,1))

#     Q_history = jax.random.multivariate_normal(key_Q_hist, mean = jnp.zeros(len(B)), cov = B, shape=(n_samples, history_length))
#     z_history = jax.random.normal(key_z_hist, (n_samples, history_length))

#     decay_vec2 = jnp.array([beta2**i for i in range(1, history_length + 1)])
#     decay_vec1 = jnp.array([beta1**i for i in range(1, history_length + 1)])
    
#     history_average2 = (1 - beta2) * jnp.einsum('abc,b->ac', f(Q_history)**2 * z_history[:,:,None]**2, decay_vec2)
#     history_average1 = (1 - beta1) * jnp.einsum('abc,b->ac', f(Q_history) * z_history[:,:,None]**2, decay_vec1)


#     fq = f(Q).squeeze(axis=1)
#     Q = Q.squeeze()
    
#     phi_coef = (1-beta1)*z**2 * fq / jnp.sqrt( history_average2 + (1-beta2) * z**2 * fq**2)
        
#     phi_coef = jnp.concatenate([phi_coef,phi_coef], axis = 1)
#     phi = jnp.mean(phi_coef * Q @ Binv, axis = 0)

#     return phi

@partial(jax.jit, static_argnames=['f','num_samples'])
def phi_from_B(B, f, beta1, beta2, key, num_samples = 100000):
    key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)
    Binv = jnp.linalg.pinv(B, hermitian=True)
    
    history_length = 500    
    
    Q = jax.random.multivariate_normal(key_Q, mean = jnp.zeros(len(B)), cov = B, shape=(num_samples, 1))
    z = jax.random.normal(key_z, (num_samples,1))

    Q_history = jax.random.multivariate_normal(key_Q_hist, mean = jnp.zeros(len(B)), cov = B, shape=(num_samples, history_length))
    z_history = jax.random.normal(key_z_hist, (num_samples, history_length))

    decay_vec1 = jnp.power(beta1, jnp.arange(1, history_length + 1))
    decay_vec2 = jnp.power(beta2, jnp.arange(1, history_length + 1))

    fq = f(Q).squeeze(axis=1)
    Q = Q.squeeze()
    
    second_moment_history = jnp.einsum('abc,b->ac', f(Q_history)**2 * z_history[:,:,None]**2, decay_vec2)
    second_moment_average = jnp.sqrt((1-beta2) * (second_moment_history + z**2 * fq**2))

    first_moment_history = f(Q_history) * z_history[:,:,None]**2
    first_moment_history = first_moment_history / second_moment_average[:,None,:]
    first_moment_history = jnp.concatenate([first_moment_history,first_moment_history], axis = -1)
    first_moment_history_w_Q = first_moment_history * Q_history @ Binv


    current_avg = z**2 * fq / second_moment_average
    current_avg = (jnp.concatenate([current_avg,current_avg], axis = -1) * Q @ Binv).mean(axis=0)
    
    history_avg = jnp.einsum('abc,b->ac',first_moment_history_w_Q, decay_vec1).mean(axis=0)

    phi = (1-beta1) * (current_avg + history_avg)
    return phi


@partial(jax.jit, static_argnames=['f','num_samples'])
def cov_from_B(B, f, beta1, beta2, key, num_samples = 100000):
    key, subkey = jax.random.split(key)
    key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(subkey, 4)
            
    history_length = 100
    
    Q = jax.random.multivariate_normal(key_Q, mean = jnp.zeros(len(B)), cov = B, shape=(num_samples, 1))
    z = jax.random.normal(key_z, (num_samples,1))

    Q_history = jax.random.multivariate_normal(key_Q_hist, mean = jnp.zeros(len(B)), cov = B, shape=(num_samples, history_length))
    z_history = jax.random.normal(key_z_hist, (num_samples, history_length))

    decay_vec1 = jnp.power(beta1, jnp.arange(1, history_length + 1))
    decay_vec2 = jnp.power(beta2, jnp.arange(1, history_length + 1))

    fq = f(Q).squeeze(axis=1)
    Q = Q.squeeze()

    current_grad = fq*z
    current_grad2 = current_grad**2

    second_moments = []
    for l in range(history_length):
        sample_grads = f(Q_history)**2 * z_history[:,:,None]**2
        sample_grads = sample_grads.at[:,l,:].set(current_grad2)
        second_moment_average = jnp.sqrt(jnp.einsum('abc,b->ac', sample_grads, decay_vec2))        
        second_moments.append(second_moment_average)
        
    second_moments = jnp.array(second_moments)

    contributions = current_grad / jnp.array(second_moments)
    update = jnp.einsum('abc,a->bc', contributions, decay_vec1) 

    op = jnp.einsum('ab,ac->abc', update, update)
    return op.mean(axis=0)


# @partial(jax.jit, static_argnames=['f','num_samples'])
# def cov_from_B(B, f, beta1, beta2,  key, num_samples = 100000):
#     key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)
    
#     history_length = 500
#     Q = jax.random.multivariate_normal(key_Q, mean = jnp.zeros(len(B)), cov = B, shape=(num_samples, 1))
#     z = jax.random.normal(key_z, (num_samples,1))

#     Q_history = jax.random.multivariate_normal(key_Q_hist, mean = jnp.zeros(len(B)), cov = B, shape=(num_samples, history_length))
#     z_history = jax.random.normal(key_z_hist, (num_samples, history_length))

#     decay_vec1 = jnp.array([beta1**i for i in range(1, history_length + 1)])
#     decay_vec2 = jnp.array([beta2**i for i in range(1, history_length + 1)])

#     fq = f(Q).squeeze(axis=1)
#     Q = Q.squeeze()

#     second_moment_history = jnp.einsum('abc,b->ac', f(Q_history)**2 * z_history[:,:,None]**2, decay_vec2)
#     second_moment_average = jnp.sqrt((1-beta2) * (second_moment_history + z**2 * fq**2))

#     first_moment_history = f(Q_history) * z_history[:,:,None]
#     first_moment_history = first_moment_history / second_moment_average[:,None,:]

#     current_avg = z * fq / second_moment_average

#     history_avg = jnp.einsum('abc,b->ac',first_moment_history, decay_vec1)

#     vvv = (1-beta1) * (current_avg + history_avg)
#     op = jnp.einsum('ab,ac->abc', vvv, vvv)
#     return op.mean(axis=0)


@partial(jax.jit, static_argnames=['f'])
def compute_I(B, f, key, n_samples = 10000):
    Q = jax.random.multivariate_normal(key, mean = np.zeros(len(B)), cov = B, shape=(n_samples, 1))
    fq = f(Q).squeeze(axis=1)
    op = jnp.einsum('ab,ac->abc', fq, fq)
    
    return jnp.mean(op, axis = 0)
    

@partial(jax.jit, static_argnames=['f'])
def compute_H(B, f, key, n_samples = 10000):    
    Binv = jnp.linalg.inv(B)        
    Q = jax.random.multivariate_normal(key, mean = jnp.zeros(len(B)), cov = B, shape=(n_samples, 1))
    
    
    fq = f(Q).squeeze(axis=1)
    Q = Q.squeeze()
    
    fq = jnp.concatenate([fq,fq], axis = 1)
    H = jnp.mean(fq  * (Q @ Binv), axis = 0)
    return H

