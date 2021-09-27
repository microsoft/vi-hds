# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from vihds.ode import OdeModel, OdeFunc, power
from vihds.precisions import ConstantPrecisions, NeuralPrecisions
from vihds.utils import default_get_value, variable_summaries
import torch
import numpy as np
import pdb

# pylint: disable = no-member, not-callable


class Inducer_Constant_RHS(OdeFunc):
    def __init__(
        self, config, theta, treatments, dev_1hot, condition_on_device, precisions=None, version=1,
    ):
        super(Inducer_Constant_RHS, self).__init__(config, theta, treatments, dev_1hot, condition_on_device)

        # Pass in a class instance for dynamic (neural) precisions. If None, then it is expected that you have
        # latent variables for the precisions, as these will be assigned as part of BaseModel.expand_precisions_by_time()
        self.precisions = precisions

        self.n_batch = theta.get_n_batch()
        self.n_iwae = theta.get_n_samples()
        self.n_species = 5

        # tile treatments, one per iwae sample
        Ara = torch.clamp(torch.exp(treatments) - 1.0, 1e-12, 1e6)
        ## c6a, c12a = torch.unbind(treatments_transformed, axis=1)
        ## c6 = torch.transpose(c6a.repeat([self.n_iwae, 1]),0,1)
        ## c12 = torch.transpose(c12a.repeat([self.n_iwae, 1]),0,1)

        # need to clip these to avoid overflow
        self.r = torch.clamp(theta.r, 0.0, 4.0)
        self.K = torch.clamp(theta.K, 0.0, 4.0)
        self.tlag = theta.tlag
        self.rc = theta.rc
        self.a530 = theta.a530
        self.a480 = theta.a480

        self.drfp = torch.clamp(theta.drfp, 1e-12, 2.0)
        self.dyfp = torch.clamp(theta.dyfp, 1e-12, 2.0)
        self.dcfp = torch.clamp(theta.dcfp, 1e-12, 2.0)

        self.aCFP = theta.aCFP_Inducer
        self.aYFP = theta.aYFP_Inducer

        self.nA = torch.clamp(theta.nA, 0.5, 3.0)
        self.eA = theta.eA
        self.KAra = theta.KAra

        # Promoter activity
        self.PBAD = (power(Ara, self.nA) + (self.eA * power(self.KAra, self.nA))) / (
            power(Ara, self.nA) + power(self.KAra, self.nA)
        )

    def forward(self, t, state):
        x, rfp, yfp, f530, f480 = torch.unbind(state[:, :, : self.n_species], axis=2)

        # Cells growing or not (not before lag-time)
        gr = self.r * torch.sigmoid(4.0 * (t - self.tlag))

        # Specific growth and dilution
        g = 1.0 - x / self.K
        gamma = gr * g

        # Right-hand sides
        d_x = gamma * x
        d_rfp = self.rc - (gamma + self.drfp) * rfp
        d_f530 = self.rc * self.a530 - gamma * f530
        d_f480 = self.rc * self.a480 - gamma * f480

        d_yfp = self.rc * self.aYFP * self.PBAD - (gamma + self.dyfp) * yfp

        dX = torch.stack([d_x, d_rfp, d_yfp, d_f530, d_f480], axis=2)
        if self.precisions is not None:
            dV = self.precisions(t, state, None, self.n_batch, self.n_iwae)
            return torch.cat([dX, dV], dim=2)
        else:
            return dX


class Inducer_Constant(OdeModel):
    def __init__(self, config):
        super(Inducer_Constant, self).init_with_params(config)
        self.precisions = ConstantPrecisions(["prec_x", "prec_rfp", "prec_yfp", "prec_cfp"])
        self.species = ["OD", "RFP", "YFP", "F530", "F480"]
        self.n_species = 5
        self.device = config.device
        self.version = 1

    def initialize_state(self, theta, _treatments):
        n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        zero = torch.zeros([n_batch, n_iwae], device=self.device)
        x0 = torch.stack([theta.init_x, theta.init_rfp, theta.init_yfp, zero, zero], axis=2)
        return x0

    def gen_reaction_equations(self, config, theta, treatments, dev_1hot, condition_on_device=True):
        func = Inducer_Constant_RHS(config, theta, treatments, dev_1hot, condition_on_device, version=self.version,)
        return func

    def summaries(self, writer, epoch):
        pass

    def observe(self, x_sample, _theta):
        x_predict = [
            x_sample[:, :, 0, :],
            x_sample[:, :, 0, :] * x_sample[:, :, 1, :],
            x_sample[:, :, 0, :] * (x_sample[:, :, 2, :] + x_sample[:, :, 3, :]),
            x_sample[:, :, 0, :] * x_sample[:, :, 4, :],
        ]
        x_predict = torch.stack(x_predict, axis=-1).permute(0, 1, 3, 2)
        return x_predict


class Inducer_Constant_Precisions(Inducer_Constant):
    def __init__(self, config):
        super(Inducer_Constant_Precisions, self).init_with_params(config)
        self.species = ["OD", "RFP", "YFP", "F530", "F480"]
        self.n_species = 5
        self.precisions = NeuralPrecisions(self.n_species, config.params.n_hidden_decoder_precisions, 4)
        self.version = 1

    def initialize_state(self, theta, _treatments):
        n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        zero = torch.zeros([n_batch, n_iwae])
        x0 = torch.stack(
            [
                theta.init_x,
                theta.init_rfp,
                theta.init_yfp,
                zero,
                zero,
                theta.init_prec_x,
                theta.init_prec_rfp,
                theta.init_prec_yfp,
                theta.init_prec_cfp,
            ],
            axis=2,
        )
        return x0

    def gen_reaction_equations(self, config, theta, treatments, dev_1hot, condition_on_device=True):
        func = Inducer_Constant_RHS(
            config, theta, treatments, dev_1hot, condition_on_device, precisions=self.precisions, version=self.version,
        )
        return func

    def summaries(self, writer, epoch):
        self.precisions.summaries(writer, epoch)
