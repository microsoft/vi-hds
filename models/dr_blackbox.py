# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import torch
import torch.nn as nn

# pylint: disable=no-member

from vihds.ode import OdeModel, OdeFunc, NeuralStates
from vihds.precisions import NeuralPrecisions
from vihds.utils import default_get_value


class DR_Blackbox_RHS(OdeFunc):
    def __init__(
        self, states: NeuralStates, offset_layer, config, theta, treatments, dev_1hot, precisions=None,
    ):
        super(DR_Blackbox_RHS, self).__init__(config, theta, treatments, dev_1hot)

        # Pass in a class instance for dynamic (neural) states.
        self.states = states

        # Pass in a class instance for dynamic (neural) precisions. If None, then it is expected that you have
        # latent variables for the precisions, as these will be assigned as part of
        # BaseModel.expand_precisions_by_time()
        self.precisions = precisions

        self.n_batch = theta.get_n_batch()
        self.n_iwae = theta.get_n_samples()

        devices = dev_1hot.unsqueeze(1).repeat([1, self.n_iwae, 1])
        treatments_rep = treatments.unsqueeze(1).repeat([1, self.n_iwae, 1])

        latent_list = []
        # locals
        n_z = config.params.n_z
        for i in range(n_z):
            latent_list.append(getattr(theta, "z%d" % (i + 1)))
        # globals
        n_x = config.params.n_x
        for i in range(n_x):
            latent_list.append(getattr(theta, "x%d" % (i + 1)))
        latents = torch.stack(latent_list, dim=-1)

        # Read global conditionals, then store constants concatenated together, ready for neural components
        n_y = config.params.n_y
        if n_y > 0:
            # Y = torch.stack([getattr(theta, "y%d"%(i+1)) for i in range(n_y)], dim=-1) + offset_layer(devices)
            Y = torch.stack([getattr(theta, "y%d" % (i + 1)) for i in range(n_y)], dim=-1)
            self.constants = torch.cat([latents, Y, treatments_rep, devices], dim=2)
        else:
            self.constants = torch.cat([latents, treatments_rep, devices], dim=2)

    def forward(self, t, state):
        dx = self.states(state[:, :, :-4], self.constants)
        dvrs = self.precisions(t, state, self.constants, self.n_batch, self.n_iwae)
        return torch.cat([dx, dvrs], dim=2)


class DR_Blackbox(OdeModel):
    def __init__(self, config):
        super(DR_Blackbox, self).__init__(config)
        self.n_x = config.params.n_x
        self.n_y = config.params.n_y
        self.n_z = config.params.n_z
        n_latents = self.n_x + self.n_y + self.n_z
        self.n_species = 4
        self.n_latent_species = config.params.n_latent_species
        self.n_hidden_precisions = config.params.n_hidden_decoder_precisions
        self.n_states = self.n_species + self.n_latent_species
        n_inputs = self.n_states + n_latents + self.n_treatments + self.device_depth
        self.precisions = NeuralPrecisions(n_inputs, self.n_hidden_precisions, 4, hidden_activation=nn.ReLU)
        self.species = ["OD", "RFP", "YFP", "CFP"]
        # do the other inits now
        self.n_hidden = config.params.n_hidden_decoder
        self.init_latent_species = default_get_value(config.params, "init_latent_species", 0.001)
        self.init_prec = default_get_value(config.params, "init_prec", 0.00001)
        self.offset_layer = nn.Linear(
            self.device_depth, self.n_y
        )  # Default initializer better than glorot_uniform here
        # nn.init.xavier_uniform_(self.offset_layer.weight)
        n = self.n_species + self.n_latent_species + n_latents + self.n_treatments + self.device_depth
        self.neural_states = NeuralStates(n, config.params.n_hidden_decoder, self.n_states, n_latents)

    def condition_theta(self, theta, dev_1hot, writer, epoch):
        """Condition on device information by mapping param_cond = f(param, d; phi) where d is one-hot rep of device"""
        # n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        devices = dev_1hot.unsqueeze(1).repeat([1, n_iwae, 1])
        offset = self.offset_layer(devices)
        for i in range(self.n_y):
            pname = "y%d" % (i + 1)
            v = getattr(theta, pname) + offset[:, :, i]
            setattr(theta, pname, v)
        return theta

    def initialize_state(self, theta, _treatments):
        n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        x0 = torch.stack([theta.init_x, theta.init_rfp, theta.init_yfp, theta.init_cfp], axis=2)
        h0 = torch.full([n_batch, n_iwae, self.n_latent_species], self.init_latent_species)
        prec0 = torch.full([n_batch, n_iwae, 4], self.init_prec)
        return torch.cat([x0, h0, prec0], dim=2)

    def gen_reaction_equations(self, config, theta, treatments, dev_1hot):
        func = DR_Blackbox_RHS(
            self.neural_states, self.offset_layer, config, theta, treatments, dev_1hot, precisions=self.precisions,
        )
        return func

    def observe(self, x_sample, _theta):
        # x0 = [theta.x0, theta.rfp0, theta.yfp0, theta.cfp0]
        x_predict = [
            x_sample[:, :, 0, :],
            x_sample[:, :, 0, :] * x_sample[:, :, 1, :],
            x_sample[:, :, 0, :] * x_sample[:, :, 2, :],
            x_sample[:, :, 0, :] * x_sample[:, :, 3, :],
        ]
        x_predict = torch.stack(x_predict, axis=-1).permute(0, 1, 3, 2)
        return x_predict

    def summaries(self, writer, epoch):
        self.neural_states.summaries(writer, epoch)
        self.precisions.summaries(writer, epoch)
