import numpy as np
from priors import *
from likelihood import *

model_configs = {
    "hyperbolic_no_contraint": {
        "name": "hyperbolic_no_contraint",
        "prior": sample_ddm_no_constraint_prior,
        "likelihood": sample_ddm_no_constraint_trial,
        "type": "hyperbolic",
        "param_names": [r"$v$", r"$a_0$", r"$\lambda$", r"$\tau$"],
        "prior_means": np.array([2.4, 3.0 , 2.0 , 0.6]),
        "prior_stds": np.array([1.82, 1.83, 1.39, 0.45])
    },
    "hyperbolic_ndt_contraint": {
        "name": "hyperbolic_ndt_contraint",
        "prior": sample_ddm_ndt_constraint_prior,
        "likelihood": sample_ddm_ndt_constraint_trial,
        "type": "hyperbolic",
        "param_names": [r"$v$", r"$a_0$", r"$\lambda$", r"$\tau$", r"$\sigma_z$"],
        "prior_means": np.array([2.4, 3.0 , 2.0 , 0.6, 0.2]),
        "prior_stds": np.array([1.82, 1.83, 1.39, 0.45, 0.15])
    },
    "exponential_no_contraint": {
        "name": "exponential_no_contraint",
        "prior": sample_ddm_no_constraint_prior,
        "likelihood": sample_ddm_no_constraint_trial,
        "type": "exponential",
        "param_names": [r"$v$", r"$a_0$", r"$\lambda$", r"$\tau$"],
        "prior_means": np.array([2.4, 3.0 , 2.0 , 0.6]),
        "prior_stds": np.array([1.82, 1.83, 1.39, 0.45])
    },
    "exponential_ndt_contraint": {
        "name": "exponential_ndt_contraint",
        "prior": sample_ddm_ndt_constraint_prior,
        "likelihood": sample_ddm_ndt_constraint_trial,
        "type": "exponential",
        "param_names": [r"$v$", r"$a_0$", r"$\lambda$", r"$\tau$", r"$\sigma_z$"],
        "prior_means": np.array([2.4, 3.0 , 2.0 , 0.6, 0.2]),
        "prior_stds": np.array([1.82, 1.83, 1.39, 0.45, 0.15])
    },
}