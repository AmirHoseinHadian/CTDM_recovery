import numpy as np
import bayesflow as bf
import pickle
from sklearn.metrics import r2_score

from src.helpers import sample_ddm, CollapsingDDM, NeuralApproximator
from src.configurations import model_configs
from src.likelihood import sample_ddm_only_noise_trial

NUM_VALIDATION_SIMS = 1000
NUM_POST_SAMPLES = 2000
NUM_OBS = 500
NUM_OBS_VECTOR = NUM_OBS * np.ones((NUM_VALIDATION_SIMS, 1))

# PARAMETER RECOVERY
#----------------------------------------------------------------------------------------#
r2_scores_dict = {}

model_keys = model_configs.keys()
for key in model_keys:
    # get model
    current_config = model_configs[key]
    model = CollapsingDDM(current_config)
    approximator = NeuralApproximator(model)
    # simulate validation data
    theta_true = current_config['prior'](NUM_VALIDATION_SIMS)
    validation_sim = sample_ddm(
        theta_true, NUM_OBS,
        current_config['likelihood'],
        type=current_config['type']
    )
    # configure data for inference
    validation_data = dict(
        direct_conditions=np.sqrt(NUM_OBS_VECTOR).astype(np.float32),
        summary_conditions=validation_sim.astype(np.float32),
    )
    # fit model to validation data
    post_samples_z = approximator.amortizer.sample(validation_data, n_samples=NUM_POST_SAMPLES)
    post_samples = post_samples_z * model.prior_stds + model.prior_means
    theta_estimated = np.median(post_samples, axis=1)
    # calculate r2 scores between true and estimated parameters
    r2_scores = r2_score(theta_true, theta_estimated, multioutput='raw_values')
    r2_scores_dict[key] = r2_scores
with open('data/BF_recovery_r2_scores.pkl', 'wb') as f:
    pickle.dump(r2_scores_dict, f)

# SENSITIVITY TO NOISE
#----------------------------------------------------------------------------------------#
r2_scores_dict = {
    key: value
    for key, value in r2_scores_dict.items()
    if 'no_contraint' in key
}
config_keys = list(model_configs.keys())
selected_keys = [config_keys[1], config_keys[-1]]
NOISE_LEVELS = [0.3, 0.6, 0.9]

for key in selected_keys:
    # get model
    current_config = model_configs[key]
    model = CollapsingDDM(model_configs[key])
    approximator = NeuralApproximator(model)
    for sigma_z in NOISE_LEVELS:
        # simulate validation data
        theta_true = current_config['prior'](NUM_VALIDATION_SIMS)
        theta_true[:, -1] = np.ones(NUM_VALIDATION_SIMS) * sigma_z
        validation_sim = sample_ddm(
            theta_true, NUM_OBS,
            current_config['likelihood'],
            type=current_config['type']
        )
        # configure data for inference
        validation_data = dict(
            direct_conditions=np.sqrt(NUM_OBS_VECTOR).astype(np.float32),
            summary_conditions=validation_sim.astype(np.float32),
        )
        # fit model to validation data
        post_samples_z = approximator.amortizer.sample(validation_data, n_samples=NUM_POST_SAMPLES)
        post_samples = post_samples_z * model.prior_stds + model.prior_means
        theta_estimated = np.median(post_samples, axis=1)
        # calculate r2 scores between true and estimated parameters
        r2_scores = r2_score(theta_true, theta_estimated, multioutput='raw_values')
        r2_scores_dict[current_config['type'] + f'_sigma_z_{sigma_z}'] = r2_scores
    
    # simulate pure noise
    theta_true = np.zeros((NUM_VALIDATION_SIMS, 6))
    theta_true[:, :-1] = current_config['prior'](NUM_VALIDATION_SIMS)
    for i in range(NUM_VALIDATION_SIMS):
        loc_z = np.random.uniform(0.05, 1.0)
        while np.abs(np.exp(loc_z + theta_true[i, -2]/2) - theta_true[i, -3]) < 0.5:
            loc_z = np.random.uniform(0.05, 1.0)
        theta_true[i, -1] = loc_z
    # simulate validation data
    validation_sim = sample_ddm(
        theta_true, NUM_OBS,
        sample_ddm_only_noise_trial,
        type=current_config['type']
    )
    # configure data for inference
    validation_data = dict(
        direct_conditions=np.sqrt(NUM_OBS_VECTOR).astype(np.float32),
        summary_conditions=validation_sim.astype(np.float32),
    )
    # fit model to validation data
    post_samples_z = approximator.amortizer.sample(validation_data, n_samples=NUM_POST_SAMPLES)
    post_samples = post_samples_z * model.prior_stds + model.prior_means
    theta_estimated = np.median(post_samples, axis=1)
    # calculate r2 scores between true and estimated parameters
    r2_scores = r2_score(theta_true[:, :4], theta_estimated[:, :4], multioutput='raw_values')
    r2_scores_dict[current_config['type'] + f'_only_noise'] = r2_scores

with open('data/BF_noise_sensitivity_r2_scores.pkl', 'wb') as f:
    pickle.dump(r2_scores_dict, f)

# SENSITIVITY TO NUM_OBS
#----------------------------------------------------------------------------------------#
r2_scores_dict = {}
NUM_OBS_LEVELS = [100, 250, 500, 1000]

for key in selected_keys:
    # get model
    current_config = model_configs[key]
    model = CollapsingDDM(model_configs[key])
    approximator = NeuralApproximator(model)
    for num_obs in NUM_OBS_LEVELS:
            # simulate validation data
            theta_true = current_config['prior'](NUM_VALIDATION_SIMS)
            validation_sim = sample_ddm(
                theta_true, num_obs,
                current_config['likelihood'],
                type=current_config['type']
            )
            num_obs_vector = num_obs * np.ones((NUM_VALIDATION_SIMS, 1))
            # configure data for inference
            validation_data = dict(
                direct_conditions=np.sqrt(num_obs_vector).astype(np.float32),
                summary_conditions=validation_sim.astype(np.float32),
            )
            # fit model to validation data
            post_samples_z = approximator.amortizer.sample(validation_data, n_samples=NUM_POST_SAMPLES)
            post_samples = post_samples_z * model.prior_stds + model.prior_means
            theta_estimated = np.median(post_samples, axis=1)
            # calculate r2 scores between true and estimated parameters
            r2_scores = r2_score(theta_true, theta_estimated, multioutput='raw_values')
            r2_scores_dict[current_config['type'] + f'_num_obs_{num_obs}'] = r2_scores

with open('data/BF_num_obs_sensitivity_r2_scores.pkl', 'wb') as f:
    pickle.dump(r2_scores_dict, f)