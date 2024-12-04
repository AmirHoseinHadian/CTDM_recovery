import numpy as np
from numba import njit, prange
from functools import partial
import bayesflow as bf
from scipy.stats import truncnorm

MIN_OBS = 100
MAX_OBS = 500

def truncnorm_better(loc=0, scale=1, low=-np.inf, high=np.inf, size=1):
    return truncnorm.rvs(
        (low - loc) / scale, (high - loc) / scale, loc=loc, scale=scale, size=size
    )

def random_num_obs(min_obs=MIN_OBS, max_obs=MAX_OBS):
    return np.random.randint(low=min_obs, high=max_obs + 1)

@njit(parallel=True)
def sample_ddm(theta, likelihood, type):
    num_obs = 500
    batch_size = theta.shape[0]
    if theta.shape[1] == 4:
        x = np.zeros((batch_size, num_obs, 1))
    else:
        x = np.zeros((batch_size, num_obs, 2))

    for i in prange(batch_size):
        for t in range(num_obs):
            x[i, t] = likelihood(theta[i], type)
    return x


class CollapsingDDM():
    def __init__(self, config):
        self.prior_means = config["prior_means"]
        self.prior_stds = config["prior_stds"]
        self.prior = bf.simulation.Prior(
            batch_prior_fun=config["prior"],
            param_names=config["param_names"]
        )
        self.output_dims = config["output_dims"]
        # self.context = bf.simulation.ContextGenerator(
        #     non_batchable_context_fun=random_num_obs
        # )
        self.likelihood = bf.simulation.Simulator(
            batch_simulator_fun=partial(sample_ddm, likelihood=config['likelihood'], type=config['type']),
            # context_generator=self.context
        )
        self.generator = bf.simulation.GenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name=config["name"],
            prior_is_batched=True,
            simulator_is_batched=True
        )

    def generate(self, batch_size):
        return self.generator(batch_size)
    
    def configure(self, forward_dict):
        data = forward_dict["sim_data"]
        # vec_num_obs = forward_dict["sim_non_batchable_context"] * np.ones((data.shape[0], 1))
        params = forward_dict["prior_draws"].astype(np.float32)
        out_dict = dict(
            parameters=(params - self.prior_means) / self.prior_stds,
            # direct_conditions=np.sqrt(vec_num_obs).astype(np.float32),
            summary_conditions=data.astype(np.float32),
        )
        return out_dict


@njit(parallel=True)
def sample_cdm(theta, likelihood, type):
    num_obs = 500
    batch_size = theta.shape[0]
    if theta.shape[1] == 5:
        x = np.zeros((batch_size, num_obs, 2))
    else:
        x = np.zeros((batch_size, num_obs, 3))

    for i in prange(batch_size):
        for t in range(num_obs):
            x[i, t] = likelihood(theta[i], type)
    return x


class CollapsingCDM():
    def __init__(self, config):
        self.prior_means = config["prior_means"]
        self.prior_stds = config["prior_stds"]
        self.prior = bf.simulation.Prior(
            batch_prior_fun=config["prior"],
            param_names=config["param_names"]
        )
        self.output_dims = config["output_dims"]
        # self.context = bf.simulation.ContextGenerator(
        #     non_batchable_context_fun=random_num_obs
        # )
        self.likelihood = bf.simulation.Simulator(
            batch_simulator_fun=partial(sample_cdm, likelihood=config['likelihood'], type=config['type']),
            # context_generator=self.context
        )
        self.generator = bf.simulation.GenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name=config["name"],
            prior_is_batched=True,
            simulator_is_batched=True
        )

    def generate(self, batch_size):
        return self.generator(batch_size)
    
    def configure(self, forward_dict):
        data = forward_dict["sim_data"]
        # vec_num_obs = forward_dict["sim_non_batchable_context"] * np.ones((data.shape[0], 1))
        params = forward_dict["prior_draws"].astype(np.float32)
        out_dict = dict(
            parameters=(params - self.prior_means) / self.prior_stds,
            # direct_conditions=np.sqrt(vec_num_obs).astype(np.float32),
            summary_conditions=data.astype(np.float32),
        )
        return out_dict


class NeuralApproximator():
    def __init__(self, model):
        self.model = model
        self.summary_net = bf.networks.SetTransformer(input_dim=model.output_dims, summary_dim=32)
        self.inference_net = bf.networks.InvertibleNetwork(
            num_params=len(model.prior.param_names),
            coupling_design="spline",
            coupling_settings={"dense_args": dict(kernel_regularizer=None), "dropout": False},
        )
        self.amortizer = bf.amortizers.AmortizedPosterior(self.inference_net, self.summary_net)
        self.trainer = bf.trainers.Trainer(
            generative_model=self.model.generate,
            configurator=self.model.configure,
            amortizer=self.amortizer,
            checkpoint_path=f"../../checkpoints/{self.model.generator.name}"
        )

    def run(self, epochs=75, iterations_per_epoch=1000, batch_size=32):
        history = self.trainer.train_online(epochs, iterations_per_epoch, batch_size)
        return history