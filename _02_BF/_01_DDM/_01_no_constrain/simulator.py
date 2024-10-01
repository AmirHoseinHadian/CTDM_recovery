import numpy as np
from numba import njit
from helpers import truncnorm_better

def sample_cddm_prior():
    drift = truncnorm_better(0, 3, low=0.0, high=np.inf, size=1)
    a0 = truncnorm_better(2.5, 2.5, low=0.25, high=np.inf, size=1)
    lamda = truncnorm_better(0, 0.05, low=0.0, high=np.inf, size=1)
    tau = truncnorm_better(0, 0.75, low=0.0, high=np.inf, size=1)
    return np.concatenate([drift, a0, lamda, tau], dtype=np.float32)

@njit
def sample_cddm_trial(v, a0, lamda, tau, beta=0.5, dt=0.001, s=1.0, max_iter=1e5):
    n_iter = 0
    x = a0 * beta
    c = np.sqrt(dt) * s
    upper_bound = a0 / (1 + lamda * np.arange(max_iter))
    lower_bound = -1 * upper_bound
    while x > lower_bound[n_iter] and x < upper_bound[n_iter] and n_iter < max_iter:
        x += v*dt + c * np.random.randn()
        n_iter += 1
    rt = n_iter * dt
    return rt+tau if x >= 0 else -(rt+tau)

@njit
def sample_cddm_experiment(theta, num_obs):
    v, a0, lamda, tau = theta
    rt = np.zeros(num_obs)
    for i in range(num_obs):
        rt[i] = sample_cddm_trial(v, a0, lamda, tau)
    return rt