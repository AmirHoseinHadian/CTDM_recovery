import numpy as np
from priors import *
from likelihood import *


model_configs = {
    "hyperbolic_ddm_no_contraint": {
        "name": "hyperbolic_ddm_no_contraint",
        "prior": sample_ddm_no_constraint_prior,
        "likelihood": sample_ddm_no_constraint_trial,
        "model_type": "ddm",
        "type": "hyperbolic",
        "output_dims": 1,
        "param_names": [r"$v$", r"$a_0$", r"$\lambda$", r"$\tau$"],
        "prior_means": np.array([2.4, 3.0 , 2.0 , 0.6]),
        "prior_stds": np.array([1.82, 1.83, 1.39, 0.45])
    },
    "hyperbolic_ddm_ndt_contraint": {
        "name": "hyperbolic_ddm_ndt_contraint",
        "prior": sample_ddm_ndt_constraint_prior,
        "likelihood": sample_ddm_ndt_constraint_trial,
        "model_type": "ddm",
        "type": "hyperbolic",
        "output_dims": 2,
        "param_names": [r"$v$", r"$a_0$", r"$\lambda$", r"$\tau$", r"$\sigma_z$"],
        "prior_means": np.array([1.5, 2.8, 1.0 , 0.5, 0.3]),
        "prior_stds": np.array([0.9, 0.7, 0.6, 0.3, 0.1])
    },
    "exponential_ddm_no_contraint": {
        "name": "exponential_ddm_no_contraint",
        "prior": sample_ddm_no_constraint_prior,
        "likelihood": sample_ddm_no_constraint_trial,
        "model_type": "ddm",
        "type": "exponential",
        "output_dims": 1,
        "param_names": [r"$v$", r"$a_0$", r"$\lambda$", r"$\tau$"],
        "prior_means": np.array([2.4, 3.0 , 2.0 , 0.6]),
        "prior_stds": np.array([1.82, 1.83, 1.39, 0.45])
    },
    "exponential_ddm_ndt_contraint": {
        "name": "exponential_ddm_ndt_contraint",
        "prior": sample_ddm_ndt_constraint_prior,
        "likelihood": sample_ddm_ndt_constraint_trial,
        "model_type": "ddm",
        "type": "exponential",
        "output_dims": 2,
        "param_names": [r"$v$", r"$a_0$", r"$\lambda$", r"$\tau$", r"$\sigma_z$"],
        "prior_means": np.array([2.4, 3.0 , 2.0 , 0.6, 0.2]),
        "prior_stds": np.array([1.82, 1.83, 1.39, 0.45, 0.15])
    },
    "hyperbolic_cdm_no_contraint": {
        "name": "hyperbolic_cdm_no_contraint",
        "prior": sample_cdm_no_constraint_prior,
        "likelihood": sample_cdm_no_constraint_trial,
        "model_type": "cdm",
        "type": "hyperbolic",
        "output_dims": 2,
        "param_names": [r"$mu_1$", r"$mu_2$", r"$a_0$", r"$\lambda$", r"$\tau$"],
        # "prior_means": np.array([2.4, 3.0 , 2.0 , 0.6]),
        # "prior_stds": np.array([1.82, 1.83, 1.39, 0.45])
    },
    "hyperbolic_cdm_ndt_contraint": {
        "name": "hyperbolic_cdm_no_contraint",
        "prior": sample_cdm_ndt_constraint_prior,
        "likelihood": sample_cdm_ndt_constraint_trial,
        "model_type": "cdm",
        "type": "hyperbolic",
        "output_dims": 3,
        "param_names": [r"$mu_1$", r"$mu_2$", r"$a_0$", r"$\lambda$", r"$\tau$", r"$\sigma_z$"],
        # "prior_means": np.array([2.4, 3.0 , 2.0 , 0.6]),
        # "prior_stds": np.array([1.82, 1.83, 1.39, 0.45])
    },
    "exponential_cdm_no_contraint": {
        "name": "exponential_cdm_no_contraint",
        "prior": sample_cdm_no_constraint_prior,
        "likelihood": sample_cdm_no_constraint_trial,
        "model_type": "cdm",
        "type": "exponential",
        "output_dims": 2,
        "param_names": [r"$mu_1$", r"$mu_2$", r"$a_0$", r"$\lambda$", r"$\tau$"],
        # "prior_means": np.array([2.4, 3.0 , 2.0 , 0.6]),
        # "prior_stds": np.array([1.82, 1.83, 1.39, 0.45])
    },
    "exponential_cdm_ndt_contraint": {
        "name": "exponential_cdm_no_contraint",
        "prior": sample_cdm_ndt_constraint_prior,
        "likelihood": sample_cdm_ndt_constraint_trial,
        "model_type": "cdm",
        "type": "exponential",
        "output_dims": 3,
        "param_names": [r"$mu_1$", r"$mu_2$", r"$a_0$", r"$\lambda$", r"$\tau$", r"$\sigma_z$"],
        # "prior_means": np.array([2.4, 3.0 , 2.0 , 0.6]),
        # "prior_stds": np.array([1.82, 1.83, 1.39, 0.45])
    },
}