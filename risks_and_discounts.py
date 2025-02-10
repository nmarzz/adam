
# import jax
# import pickle
# import numpy as np
# from numpy.polynomial.hermite import hermgauss
# from jax import jit
# import jax.numpy as jnp
# from jax.nn import sigmoid as sig
# from utils import make_data, make_B
# from tqdm import tqdm


# @jit
# def logreg(data, theta):
#     logits = jnp.dot(data, theta)  
#     probabilities = sig(logits)
#     return probabilities

# @jit
# def grad_logreg(params, data, target):
#     p_student = logreg(data, params)
#     return (p_student - target) * data

# @jit
# def grad_linreg(params, data, target):    
#     return (jnp.inner(data, params) - target) * data

# @jit
# def grad_phase_ret(params, data, target):
#     # TODO
#     pass    

# @jit
# def linreg_target(optimal_params, data):
#     return jnp.inner(optimal_params, data)

# @jit
# def logreg_target(optimal_params, data):
#     return sig(jnp.inner(optimal_params, data))

# @jit
# def phase_ret_target(optimal_params, data):
#     # TODO
#     pass    

# @jit
# def risk_from_B_linreg(B):
#     return (B[0,0] + B[1,1] - 2 * B[0,1]) / 2

# @jit
# def risk_from_B_phase_retrieval(B):
#     return (3*B[0,0]**2 -2 * B[0,0] * B[1,1] - 4 * B[0,1]**2 + 3 * B[1,1]**2)

# @jit
# def risk_from_B_logreg(B):
#     B00 = B[0, 0]
#     B11 = B[1, 1]
#     B01 = B[0, 1]
    
#     num_points = 20
#     points, weights = hermgauss(num_points)  # Nodes and weights for Hermite quadrature
#     points *= jnp.sqrt(2)
    
#     t1 = -B01 * jnp.exp(jnp.sqrt(B11) * points) / (1 + jnp.exp(jnp.sqrt(B11) * points))**2
#     t2 = jnp.log(1 + jnp.exp(jnp.sqrt(B00) * points))
        
#     return jnp.sum(weights * (t1 + t2)) / jnp.sqrt(jnp.pi)
    
    
    
    
    
    
# import jax
# import numpy as np
# import jax.numpy as jnp
# from jax import jit
# from tqdm import tqdm
# from utils import make_B
# from jax.nn import sigmoid
# from functools import partial
# from numpy.polynomial.hermite import hermgauss

# @jit
# def logreg(data, theta):
#     logits = jnp.dot(data, theta)  
#     probabilities = sigmoid(logits)
#     return probabilities

# @jit
# def risk_from_B_linreg(B):
#     return (B[0,0] + B[1,1] - 2 * B[0,1]) / 2

# @jit
# def f_linreg(r):
#     r1 = r[:,:,0]
#     r2 = r[:,:,1]
#     return r1 - r2

# @jit
# def f_logreg(r):
#     r1 = r[:,:,0]
#     r2 = r[:,:,1]
#     return sigmoid(r1) - sigmoid(r2)


# @jit
# def risk_from_B_logreg(B):
#     B00 = B[0, 0]
#     B11 = B[1, 1]
#     B01 = B[0, 1]
    
#     num_points = 20
#     points, weights = hermgauss(num_points)  # Nodes and weights for Hermite quadrature
#     points *= jnp.sqrt(2)
    
#     t1 = -B01 * jnp.exp(jnp.sqrt(B11) * points) / (1 + jnp.exp(jnp.sqrt(B11) * points))**2
#     t2 = jnp.log(1 + jnp.exp(jnp.sqrt(B00) * points))
        
#     return jnp.sum(weights * (t1 + t2)) / jnp.sqrt(jnp.pi)
    

# @partial(jax.jit, static_argnames=['f'])
# def phi_from_B(B, f, beta,  key, n_samples = 10000):
#     key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)
#     Binv = jnp.linalg.inv(B)

#     history_length = 50
#     Q = jax.random.multivariate_normal(key_Q, mean = np.zeros(2), cov = B, shape=(n_samples, 1))
#     z = jax.random.normal(key_z, (n_samples))
    
#     Q_history = jax.random.multivariate_normal(key_Q_hist, mean = np.zeros(2), cov = B, shape=(n_samples, history_length))
#     z_history = jax.random.normal(key_z_hist, (n_samples, history_length))

#     decay_vec = jnp.array([beta ** i for i in range(1, history_length + 1)])
#     history_average = (1-beta)*(f(Q_history)**2 * z_history**2) @ decay_vec

#     fq = f(Q)
#     fq = fq.squeeze()
#     Q = Q.squeeze()
        
#     phi = jnp.mean((z**2 * fq / jnp.sqrt( history_average + (1-beta) * z**2 * fq**2))[:,None]  *  (Q @ Binv), axis = 0)
#     return phi



# @partial(jax.jit, static_argnames=['f'])
# def cov_from_B(B, f, beta,  key, n_samples = 10000):        
#     key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)

#     history_length = 50
#     Q = jax.random.multivariate_normal(key_Q, mean = np.zeros(2), cov = B, shape=(n_samples, 1))
#     z = jax.random.normal(key_z, (n_samples))

#     Q_history = jax.random.multivariate_normal(key_Q_hist, mean = np.zeros(2), cov = B, shape=(n_samples, history_length))
#     z_history = jax.random.normal(key_z_hist, (n_samples, history_length))

#     decay_vec = jnp.array([beta ** i for i in range(1, history_length + 1)])
#     history_average = (1-beta)*(f(Q_history)**2 * z_history**2) @ decay_vec    

#     fq = f(Q)
#     fq = fq.squeeze()        
    
#     return jnp.mean(z**2 * fq**2 / (history_average + (1-beta) * z**2 * fq**2))



# @partial(jax.jit, static_argnames=['f'])
# def compute_I(B, f, key, n_samples = 10000):    
#     Q = jax.random.multivariate_normal(key, mean = np.zeros(2), cov = B, shape=(n_samples, 1))    

#     fq = f(Q)
#     fq = fq.squeeze()    
        
#     I = jnp.mean(fq**2)
    
#     return I


# @partial(jax.jit, static_argnames=['f'])
# def compute_H(B, f,  key, n_samples = 10000):    
#     Binv = jnp.linalg.inv(B)
    
#     Q = jax.random.multivariate_normal(key, mean = np.zeros(2), cov = B, shape=(n_samples, 1))    
    
#     fq = f(Q)
#     fq = fq.squeeze()
#     Q = Q.squeeze()
        
#     H = jnp.mean(fq[:, None]  * (Q @ Binv), axis = 0)
#     return H



# import jax
# import numpy as np
# from numpy.polynomial.hermite import hermgauss
# from jax import jit
# import jax.numpy as jnp
# from jax.nn import sigmoid
# from utils import make_B
# from tqdm import tqdm
# from functools import partial


# @partial(jax.jit, static_argnames=['f'])
# def phi_from_B(B, f, beta,  key, n_samples = 10000):
#     key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)
#     Binv = jnp.linalg.inv(B)

#     history_length = 50
#     Q = jax.random.multivariate_normal(key_Q, mean = np.zeros(2), cov = B, shape=(n_samples, 1))
#     z = jax.random.normal(key_z, (n_samples))
    
#     Q_history = jax.random.multivariate_normal(key_Q_hist, mean = jnp.zeros(2), cov = B, shape=(n_samples, history_length))
#     z_history = jax.random.normal(key_z_hist, (n_samples, history_length))

#     decay_vec = jnp.array([beta ** i for i in range(1, history_length + 1)])
#     history_average = (1-beta)*(f(Q_history)**2 * z_history**2) @ decay_vec

#     fq = f(Q)
#     fq = fq.squeeze()
#     Q = Q.squeeze()
        
#     phi = jnp.mean((z**2 * fq / jnp.sqrt( history_average + (1-beta) * z**2 * fq**2))[:,None]  *  (Q @ Binv), axis = 0)
#     return phi

# @partial(jax.jit, static_argnames=['f'])
# def cov_from_B(B, f, beta,  key, n_samples = 10000):
#     key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)

#     history_length = 50
#     Q = jax.random.multivariate_normal(key_Q, mean = np.zeros(2), cov = B, shape=(n_samples, 1))
#     z = jax.random.normal(key_z, (n_samples))

#     Q_history = jax.random.multivariate_normal(key_Q_hist, mean = np.zeros(2), cov = B, shape=(n_samples, history_length))
#     z_history = jax.random.normal(key_z_hist, (n_samples, history_length))

#     decay_vec = jnp.array([beta ** i for i in range(1, history_length + 1)])
#     history_average = (1-beta)*(f(Q_history)**2 * z_history**2) @ decay_vec    

#     fq = f(Q)
#     fq = fq.squeeze()        
    
#     return jnp.mean(z**2 * fq**2 / (history_average + (1-beta) * z**2 * fq**2))

# @jit
# def logreg(data, theta):
#     logits = jnp.dot(data, theta)  
#     probabilities = sigmoid(logits)
#     return probabilities

# @jit
# def grad_logreg(params, data, target):
#     p_student = logreg(data, params)
#     return (p_student - target) * data

# @jit
# def grad_linreg(params, data, target):    
#     return (jnp.inner(data, params) - target) * data

# @jit
# def linreg_target(optimal_params, data):
#     return jnp.inner(optimal_params, data)

# @jit
# def logreg_target(optimal_params, data):
#     return sigmoid(jnp.inner(optimal_params, data))

# @jit
# def risk_from_B_linreg(B):
#     return (B[0,0] + B[1,1] - 2 * B[0,1]) / 2

# @jit
# def f_linreg(r):
#     r1 = r[:,:,0]
#     r2 = r[:,:,1]
#     return r1 - r2

# @jit
# def f_logreg(r):
#     r1 = r[:,:,0]
#     r2 = r[:,:,1]
#     return sigmoid(r1) - sigmoid(r2)

# @jit
# def risk_from_B_logreg(B):
#     B00 = B[0, 0]
#     B11 = B[1, 1]
#     B01 = B[0, 1]
    
#     num_points = 20
#     points, weights = hermgauss(num_points)  # Nodes and weights for Hermite quadrature
#     points *= jnp.sqrt(2)
    
#     t1 = -B01 * jnp.exp(jnp.sqrt(B11) * points) / (1 + jnp.exp(jnp.sqrt(B11) * points))**2
#     t2 = jnp.log(1 + jnp.exp(jnp.sqrt(B00) * points))
        
#     return jnp.sum(weights * (t1 + t2)) / jnp.sqrt(jnp.pi)
    
import jax
import numpy as np
from numpy.polynomial.hermite import hermgauss
from jax import jit
import jax.numpy as jnp
from jax.nn import sigmoid as sig
from functools import partial
from utils import make_B, make_data
from jax.nn import sigmoid
from tqdm import tqdm

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
def grad_phase_ret(params, data, target):
    # TODO
    pass    

@jit
def linreg_target(optimal_params, data):
    return jnp.inner(optimal_params, data)

@jit
def logreg_target(optimal_params, data):
    return sig(jnp.inner(optimal_params, data))

@jit
def phase_ret_target(optimal_params, data):
    # TODO
    pass    

@jit
def risk_from_B_linreg(B):
    return (B[0,0] + B[1,1] - 2 * B[0,1]) / 2

@jit
def risk_from_B_phase_retrieval(B):
    return (3*B[0,0]**2 -2 * B[0,0] * B[1,1] - 4 * B[0,1]**2 + 3 * B[1,1]**2)

@jit
def risk_from_B_logreg(B):
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
def f_linreg(r):
    r1 = r[:,:,0]
    r2 = r[:,:,1]
    return r1 - r2

@jit
def f_logreg(r):
    r1 = r[:,:,0]
    r2 = r[:,:,1]
    return sigmoid(r1) - sigmoid(r2)

@partial(jax.jit, static_argnames=['f'])
def phi_from_B(B, f, beta,  key, n_samples = 10000):
    key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)
    Binv = jnp.linalg.inv(B)

    history_length = 50
    Q = jax.random.multivariate_normal(key_Q, mean = np.zeros(2), cov = B, shape=(n_samples, 1))
    z = jax.random.normal(key_z, (n_samples))
    
    Q_history = jax.random.multivariate_normal(key_Q_hist, mean = np.zeros(2), cov = B, shape=(n_samples, history_length))
    z_history = jax.random.normal(key_z_hist, (n_samples, history_length))

    decay_vec = jnp.array([beta ** i for i in range(1, history_length + 1)])
    history_average = (1-beta)*(f(Q_history)**2 * z_history**2) @ decay_vec

    fq = f(Q)
    fq = fq.squeeze()
    Q = Q.squeeze()
        
    phi = jnp.mean((z**2 * fq / jnp.sqrt( history_average + (1-beta) * z**2 * fq**2))[:,None]  *  (Q @ Binv), axis = 0)
    return phi

@partial(jax.jit, static_argnames=['f'])
def cov_from_B(B, f, beta,  key, n_samples = 10000):        
    key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)

    history_length = 50
    Q = jax.random.multivariate_normal(key_Q, mean = np.zeros(2), cov = B, shape=(n_samples, 1))
    z = jax.random.normal(key_z, (n_samples))

    Q_history = jax.random.multivariate_normal(key_Q_hist, mean = np.zeros(2), cov = B, shape=(n_samples, history_length))
    z_history = jax.random.normal(key_z_hist, (n_samples, history_length))

    decay_vec = jnp.array([beta ** i for i in range(1, history_length + 1)])
    history_average = (1-beta)*(f(Q_history)**2 * z_history**2) @ decay_vec    

    fq = f(Q)
    fq = fq.squeeze()        
    
    return jnp.mean(z**2 * fq**2 / (history_average + (1-beta) * z**2 * fq**2))

@partial(jax.jit, static_argnames=['f'])
def compute_I(B, f, key, n_samples = 10000):    
    Q = jax.random.multivariate_normal(key, mean = np.zeros(2), cov = B, shape=(n_samples, 1))    

    fq = f(Q)
    fq = fq.squeeze()    
        
    I = jnp.mean(fq**2)
    
    return I

@partial(jax.jit, static_argnames=['f'])
def compute_H(B, f,  key, n_samples = 10000):    
    Binv = jnp.linalg.inv(B)
    
    Q = jax.random.multivariate_normal(key, mean = np.zeros(2), cov = B, shape=(n_samples, 1))    
    
    fq = f(Q)
    fq = fq.squeeze()
    Q = Q.squeeze()
        
    H = jnp.mean(fq[:, None]  * (Q @ Binv), axis = 0)
    return H
