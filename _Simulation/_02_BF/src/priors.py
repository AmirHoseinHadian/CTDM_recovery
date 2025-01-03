import numpy as np
from helpers import truncnorm_better

#----------------------------------------------------------------------------------------#
# DDM
#----------------------------------------------------------------------------------------#

# def sample_ddm_no_constraint_prior(batch_size):
#     drift = truncnorm_better(1, 2, low=0.0, high=np.inf, size=batch_size)
#     a0 = truncnorm_better(2.5, 1.5, low=0.25, high=np.inf, size=batch_size)
#     lamda = truncnorm_better(1, 1, low=0.0, high=np.inf, size=batch_size)
#     tau = truncnorm_better(0.3, 0.3, low=0.0, high=np.inf, size=batch_size)
#     return np.c_[drift, a0, lamda, tau]

def sample_ddm_no_constraint_prior(batch_size):
    drift = np.random.uniform(0, 3, size=batch_size)
    a0 = np.random.uniform(1.5, 4, size=batch_size)
    lamda = np.random.uniform(0.1, 2, size=batch_size)
    tau = np.random.uniform(0.05, 1, size=batch_size)
    return np.c_[drift, a0, lamda, tau]

# def sample_ddm_ndt_constraint_prior(batch_size):
#     drift = truncnorm_better(1, 2, low=0.0, high=np.inf, size=batch_size)
#     a0 = truncnorm_better(2.5, 1.5, low=0.25, high=np.inf, size=batch_size)
#     lamda = truncnorm_better(1, 1, low=0.0, high=np.inf, size=batch_size)
#     tau = truncnorm_better(0.3, 0.3, low=0.0, high=np.inf, size=batch_size)
#     sigma_z = truncnorm_better(0.5, 0.5, low=0.001, high=np.inf, size=batch_size)
#     return np.c_[drift, a0, lamda, tau, sigma_z]

def sample_ddm_ndt_constraint_prior(batch_size):
    drift = np.random.uniform(0, 3, size=batch_size)
    a0 = np.random.uniform(1.5, 4, size=batch_size)
    lamda = np.random.uniform(0.1, 2, size=batch_size)
    tau = np.random.uniform(0.05, 1, size=batch_size)
    sigma_z = np.random.uniform(0.1, 1.0, size=batch_size)
    return np.c_[drift, a0, lamda, tau, sigma_z]

#----------------------------------------------------------------------------------------#
# CDM
#----------------------------------------------------------------------------------------#
# def sample_cdm_no_constraint_prior(batch_size):
#     mu = np.random.uniform(-3, 3, size=(batch_size, 2))
#     a0 = np.random.uniform(1.5, 4, size=(batch_size, 1))
#     lamda = np.random.uniform(0.1, 2, size=(batch_size, 1))
#     tau = np.random.uniform(0.05, 1, size=(batch_size, 1))
#     return np.c_[mu, a0, lamda, tau]

# def sample_cdm_ndt_constraint_prior(batch_size):
#     mu = np.random.uniform(-3, 3, size=(batch_size, 2))
#     a0 = np.random.uniform(1.5, 4, size=(batch_size, 1))
#     lamda = np.random.uniform(0.1, 2, size=(batch_size, 1))
#     tau = np.random.uniform(0.05, 1, size=(batch_size, 1))
#     sigma_z = np.random.uniform(0.05, 1.0, size=(batch_size, 1))
#     return np.c_[mu, a0, lamda, tau, sigma_z]