# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
import numpy as np
from typing import Any, Dict, List, Optional
import models

#pylint: disable=no-member

class Decoder(nn.Module):
    '''
    Decoder network
    '''
    def __init__(self, config, condition_on_device):
        super(Decoder, self).__init__()
        print("Initialising decoder")
        # This is already a model object because of the use of "!!python/object:... in the yaml file.
        #self.ode_models = {k: model.init_with_params(config) for k, model in config.models.items()}
        ode_model_class = models.LOOKUP[config.model]
        # Set various attributes of the model
        self.ode_model = ode_model_class(config)
        self.state_names = self.ode_model.species
        self.condition_on_device = condition_on_device
        self.config = config
        
    def forward(self, theta, data, writer, epoch):
        if self.condition_on_device:
            theta_conditioned = self.ode_model.condition_theta(theta, data.dev_1hot, writer, epoch)
        else:
            theta_conditioned = theta
        solution = self.ode_model.simulate(self.config, data.times, theta_conditioned, data.inputs, data.dev_1hot, condition_on_device=self.condition_on_device)
        x_states, precisions = self.ode_model.expand_precisions(theta_conditioned, data.times, solution)
        x_predict = self.ode_model.observe(x_states, theta_conditioned)
        if writer is not None:
            self.ode_model.summaries(writer, epoch)
        return (x_states, x_predict, precisions), theta_conditioned
