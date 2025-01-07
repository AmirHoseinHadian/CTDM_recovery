import numpy as np
import bayesflow as np

from src.helpers import CollapsingDDM, NeuralApproximator
from src.configurations import model_configs

EPOCHS = 75

for key in model_configs.keys():
    # get model
    model = CollapsingDDM(model_configs[key])
    approximator = NeuralApproximator(model)
    # train neural approximator
    history = approximator.run(EPOCHS)