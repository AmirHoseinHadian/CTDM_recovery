import numpy as np
from numba import njit

#----------------------------------------------------------------------------------------#
# DDM
#----------------------------------------------------------------------------------------#
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
    v, a0, lamda, tau, scale_z = theta
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
    loc_z = np.log(tau) - 0.5 * scale_z**2
    return np.array([rt+tau if x >= 0 else -(rt+tau), np.random.lognormal(loc_z, scale_z)])

#----------------------------------------------------------------------------------------#
# CDDM
#----------------------------------------------------------------------------------------#
# @njit
# def sample_cdm_no_constraint_trial(theta, type, dt=0.001, s=1.0, max_iter=1e5):
#     mu1, mu2, a0, lamda, tau = theta
#     mu = np.array([mu1, mu2])
#     n_iter = 0
#     x = np.zeros(2)
#     c = np.sqrt(dt) * s
#     t = np.arange(0, max_iter * dt, dt)
#     if type == "hyperbolic":
#         threshold = a0 / (1 + lamda * t)
#     else:
#         threshold = a0 * np.exp(-lamda * t)

#     while np.linalg.norm(x, 2) < threshold[n_iter] and n_iter < max_iter:
#         x += mu*dt + c * np.random.randn(2)
#         n_iter += 1
#     rt = n_iter * dt + tau
#     resp = np.arctan2(x[1], x[0])
#     return np.array([rt, resp])

# @njit
# def sample_cdm_ndt_constraint_trial(theta, type, dt=0.001, s=1.0, max_iter=1e5):
#     mu1, mu2, a0, lamda, tau, sigma_z = theta
#     mu = np.array([mu1, mu2])
#     n_iter = 0
#     x = np.zeros(2)
#     c = np.sqrt(dt) * s
#     t = np.arange(0, max_iter * dt, dt)
#     if type == "hyperbolic":
#         threshold = a0 / (1 + lamda * t)
#     else:
#         threshold = a0 * np.exp(-lamda * t)

#     while np.linalg.norm(x, 2) < threshold[n_iter] and n_iter < max_iter:
#         x += mu*dt + c * np.random.randn(2)
#         n_iter += 1
#     rt = n_iter * dt + tau
#     resp = np.arctan2(x[1], x[0])
#     return np.array([rt, resp, np.random.normal(tau, sigma_z)])


