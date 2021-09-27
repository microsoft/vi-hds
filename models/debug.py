# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import torch
from vihds.ode import OdeModel

# pylint: disable=no-member


class Debug_Constant(OdeModel):
    def __init__(self, config):
        super(Debug_Constant, self).__init__(config)
        self.species = ["OD", "RFP", "YFP", "CFP"]
        self.n_species = len(self.species)

    def initialize_state(self, theta, _treatments):
        n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        x0 = torch.cat(
            [theta.init_x.reshape([n_batch, n_iwae, 1]), torch.zeros([n_batch, n_iwae, self.n_species - 1])], 2,
        )
        return x0

    def observe(self, x_sample, _theta):
        x_predict = [
            x_sample[:, :, :, 0],
            x_sample[:, :, :, 0] * x_sample[:, :, :, 1],
            x_sample[:, :, :, 0] * x_sample[:, :, :, 2],
            x_sample[:, :, :, 0] * x_sample[:, :, :, 3],
        ]
        x_predict = torch.stack(x_predict, axis=-1)
        return x_predict

    def gen_reaction_equations(self, theta, treatments, dev_1hot, condition_on_device=True):

        r = theta.r

        def reaction_equations(t, state):
            x, rfp, yfp, cfp = torch.unbind(state, axis=2)

            gamma = r * (1.0 - x)
            # Right-hand sides
            d_x = x * gamma
            # d_x = tf.verify_tensor_all_finite(d_x, "d_x NOT finite")
            d_rfp = 1.0 - (gamma + 1.0) * rfp
            d_yfp = 1.0 - (gamma + 1.0) * yfp
            d_cfp = 1.0 - (gamma + 1.0) * cfp

            X = torch.stack([d_x, d_rfp, d_yfp, d_cfp], axis=2)
            return X

        return reaction_equations
