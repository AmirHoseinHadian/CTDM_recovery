import numpy as np
from .priors import *
from .likelihood import *

model_configs = {
    "hyperbolic_ddm_no_contraint": {
        "name": "hyperbolic_ddm_no_contraint",
        "prior": sample_ddm_no_constraint_prior,
        "likelihood": sample_ddm_no_constraint_trial,
        "model_type": "ddm",
        "type": "hyperbolic",
        "output_dims": 1,
        "param_names": [r"$v$", r"$a_0$", r"$\lambda$", r"$\tau$"],
        "prior_means": np.array([1.5 , 2.75, 1.05, 0.53]),
        "prior_stds": np.array([0.86, 0.72, 0.55, 0.27])
    },
    "hyperbolic_ddm_ndt_contraint": {
        "name": "hyperbolic_ddm_ndt_contraint",
        "prior": sample_ddm_ndt_constraint_prior,
        "likelihood": sample_ddm_ndt_constraint_trial,
        "model_type": "ddm",
        "type": "hyperbolic",
        "output_dims": 2,
        "param_names": [r"$v$", r"$a_0$", r"$\lambda$", r"$\tau$", r"$\sigma_z$"],
        "prior_means": np.array([1.5 , 2.75, 1.05, 0.53, 0.5]),
        "prior_stds": np.array([0.86, 0.72, 0.55, 0.27, 0.29])
    },
    "exponential_ddm_no_contraint": {
        "name": "exponential_ddm_no_contraint",
        "prior": sample_ddm_no_constraint_prior,
        "likelihood": sample_ddm_no_constraint_trial,
        "model_type": "ddm",
        "type": "exponential",
        "output_dims": 1,
        "param_names": [r"$v$", r"$a_0$", r"$\lambda$", r"$\tau$"],
        "prior_means": np.array([1.5 , 2.75, 1.05, 0.53]),
        "prior_stds": np.array([0.86, 0.72, 0.55, 0.27])
    },
    "exponential_ddm_ndt_contraint": {
        "name": "exponential_ddm_ndt_contraint",
        "prior": sample_ddm_ndt_constraint_prior,
        "likelihood": sample_ddm_ndt_constraint_trial,
        "model_type": "ddm",
        "type": "exponential",
        "output_dims": 2,
        "param_names": [r"$v$", r"$a_0$", r"$\lambda$", r"$\tau$", r"$\sigma_z$"],
        "prior_means": np.array([1.5 , 2.75, 1.05, 0.53, 0.5]),
        "prior_stds": np.array([0.86, 0.72, 0.55, 0.27, 0.29])
    },
}