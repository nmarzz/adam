import jax
from jax.numpy.linalg import norm
import os
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from optimizers import Adam, SGD, ResampledAdam
from sdes import AdamSDE, SgdSDE
from odes import AdamODE, SgdODE
print('imported')


# Fixed config
T = 20
m = 1
dt = 0.01
# problem_type = 'linreg'


# Parameter sweep
# beta1_list = [0.99] # be careful + increase "history_length" above beta* = 0.9 or so
# beta2_list = [0.999]
beta1_list = [0.1, 0.5, 0.9, 0.99, 0.999] # be careful + increase "history_length" above beta* = 0.9 or so
beta2_list = [0.1, 0.5, 0.9, 0.99, 0.999]
d_list = [250, 1000]
# d_list = [250]
num_samples_list = [10000]
scale_list = [5]
lr_list = [1.5]
problem_list = ['linreg', 'logreg']

# Testing
# beta1_list = [0.9] # be careful + increase "history_length" above beta* = 0.9 or so
# beta2_list = [0.3]
# d_list = [100,200]
# num_samples_list = [10000]
# scale_list = [1]
# lr_list = [0.2]
# problem_list = ['linreg', 'logreg']

print('Setup')

folder = "results_isotropic_betasweep"

os.makedirs(folder, exist_ok=True)

def generate_cov(d, beta=0.5):
    # return jnp.array([j**(-beta) for j in range(1, d+1)])
    return jnp.ones(d)

def run_single_experiment(beta1, beta2, d, num_samples, scale, lr, problem_type):
    cov = generate_cov(d)
    lrk = lr / d

    num_runs = 30
    all_sgd_risks = []
    all_adam_risks = []
    key = jax.random.PRNGKey(1331)
    # key = jax.random.PRNGKey(np.random.randint(0, 10000))                
    key_init, key_opt, _ = jax.random.split(key, 3)

    params0 = jax.random.normal(key_init, (d, m))
    optimal_params = jax.random.normal(key_opt, (d, m))

    params0 /= norm(params0, axis=0)
    optimal_params /= norm(optimal_params, axis=0)
    optimal_params *= scale
    for _ in range(num_runs):
        # # Run optimizers
        # print('Running optimizers')
        sgd = SGD(problem_type)
        adam = Adam(problem_type)        
            
        _, sgd_risks = sgd.run(params0, cov, T, lrk, optimal_params)    
        _, adam_risks = adam.run(params0, cov, T, lrk, optimal_params, beta1=beta1, beta2=beta2, eps=0.00)
        all_sgd_risks.append(sgd_risks)
        all_adam_risks.append(adam_risks)
    
    all_sgd_risks = jnp.array(all_sgd_risks)
    all_adam_risks = jnp.array(all_adam_risks)
    
    lower_sgd = jnp.quantile(all_sgd_risks, 0.1, axis=0)
    upper_sgd = jnp.quantile(all_sgd_risks, 0.9, axis=0)
    mean_sgd = jnp.mean(all_sgd_risks, axis=0)

    lower_adam = jnp.quantile(all_adam_risks, 0.1, axis=0)
    upper_adam = jnp.quantile(all_adam_risks, 0.9, axis=0)
    mean_adam = jnp.mean(all_adam_risks, axis=0)

    
    # sgd_ode = SgdODE(problem_type)
    # adam_ode = AdamODE(problem_type)
    # sgd_risk, sgd_time, _ = sgd_ode.run(params0, optimal_params, cov, T, lr, dt=2*dt)
    # adam_risk, adam_time, _ = adam_ode.run(params0, optimal_params, cov, T, lr, dt=dt, beta1=beta1, beta2=beta2, eps=0.00, num_samples=num_samples)

    # Save data
    tag = f"b1_{beta1}_b2_{beta2}_d_{d}_n_{num_samples}_scale_{scale}_lr_{lr}_problem_{problem_type}"
    np.savez(f"{folder}/{problem_type}_{tag}.npz",
             adam_risks=all_adam_risks,
            #  adam_risks_beta0=adam_risks_beta0,
            #  resampled_adam_risks=resampled_adam_risks,
             sgd_risks=all_sgd_risks,
            #  adam_risk=adam_risk,
            #  adam_time=adam_time,
            #  sgd_risk=sgd_risk,
            #  sgd_time=sgd_time
             )

    plt.figure()
    plt.yscale('log')

    t_range = jnp.arange(all_adam_risks.shape[1])

    plt.fill_between(t_range, lower_adam, upper_adam, alpha=0.3, label='Adam 80% CI')
    plt.plot(t_range, mean_adam, label='Adam')

    plt.fill_between(t_range, lower_sgd, upper_sgd, alpha=0.3, label='SGD 80% CI')
    plt.plot(t_range, mean_sgd, label='SGD')

    # plt.plot(adam_time * d, adam_risk, label='Adam ODE')
    # plt.plot(sgd_time * d, sgd_risk, label='SGD ODE')

    plt.legend()
    plt.title(tag.replace("_", " "))
    plt.savefig(f"{folder}/{problem_type}_{tag}.pdf", format = 'pdf')
    plt.close()

    print(f'Finished: {tag}')

import time

# Run all combinations
for d in d_list:
    for problem_type in problem_list:
        for beta2 in beta2_list:
            for beta1 in beta1_list:
                for num_samples in num_samples_list:
                    for scale in scale_list:
                        for lr in lr_list:                                
                            print(f'Running: b1={beta1}, b2={beta2}, d={d}, n={num_samples}, scale={scale}, lr={lr}, problem={problem_type}')                
                            start = time.time()
                            run_single_experiment(beta1, beta2, d, num_samples, scale, lr, problem_type)
                            end = time.time()
                            print(f"Elapsed time: {end - start:.2f} seconds\n")
