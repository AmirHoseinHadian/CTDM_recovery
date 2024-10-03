import numpy as np
from numba import njit

@njit
def sample_ddm_no_constraint_trial(theta, type, beta=0.5, dt=0.001, s=1.0, max_iter=1e5):
    v, a0, lamda, tau = theta
    n_iter = 0
    x = a0 * beta
    c = np.sqrt(dt) * s
    t = np.arange(0, max_iter * dt, dt)
    if type == "hyperbolic":
        upper_bound = a0 / (1 + lamda * t)
    else:
        upper_bound = a0 * np.exp(-lamda * t)
    lower_bound = -1 * upper_bound

    while x > lower_bound[n_iter] and x < upper_bound[n_iter] and n_iter < max_iter:
        x += v*dt + c * np.random.randn()
        n_iter += 1
    rt = n_iter * dt
    return np.array([rt+tau if x >= 0 else -(rt+tau)])


@njit
def sample_ddm_ndt_constraint_trial(theta, type, beta=0.5, dt=0.001, s=1.0, max_iter=1e5):
    v, a0, lamda, tau, sigma_z = theta
    n_iter = 0
    x = a0 * beta
    c = np.sqrt(dt) * s
    t = np.arange(0, max_iter * dt, dt)
    if type == "hyperbolic":
        upper_bound = a0 / (1 + lamda * t)
    else:
        upper_bound = a0 * np.exp(-lamda * t)
    lower_bound = -1 * upper_bound

    while x > lower_bound[n_iter] and x < upper_bound[n_iter] and n_iter < max_iter:
        x += v*dt + c * np.random.randn()
        n_iter += 1
    rt = n_iter * dt
    return np.array([rt+tau if x >= 0 else -(rt+tau), np.random.normal(tau, sigma_z)])