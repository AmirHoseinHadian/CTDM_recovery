import numpy as np
from scipy.stats import truncnorm

MIN_OBS = 100
MAX_OBS = 500


def truncnorm_better(loc=0, scale=1, low=-np.inf, high=np.inf, size=1):
    return truncnorm.rvs(
        (low - loc) / scale, (high - loc) / scale, loc=loc, scale=scale, size=size
    )

def random_num_obs(min_obs=MIN_OBS, max_obs=MAX_OBS):
    return np.random.randint(low=min_obs, high=max_obs + 1)