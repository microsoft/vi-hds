# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
from vihds.ode import OdeModel, OdeFunc, power
from vihds.precisions import ConstantPrecisions, NeuralPrecisions
from vihds.utils import default_get_value, variable_summaries
import torch
import numpy as np
import pdb

# pylint: disable = no-member, not-callable


class Degrader_Constant_RHS(OdeFunc):
    def __init__(
        self, config, theta, treatments, dev_1hot, condition_on_device, precisions=None, version=1,
    ):
        super(Degrader_Constant_RHS, self).__init__(config, theta, treatments, dev_1hot, condition_on_device)

        # Pass in a class instance for dynamic (neural) precisions. If None, then it is expected that you have
        # latent variables for the precisions, as these will be assigned as part of BaseModel.expand_precisions_by_time()
        self.precisions = precisions

        self.n_batch = theta.get_n_batch()
        self.n_iwae = theta.get_n_samples()
        self.n_species = 9

        # tile treatments, one per iwae sample
        treatments_transformed = torch.clamp(torch.exp(treatments) - 1.0, 1e-12, 1e6)
        c6a, c12a, ara_a = torch.unbind(treatments_transformed, axis=1)
        c6 = torch.transpose(c6a.repeat([self.n_iwae, 1]), 0, 1)
        c12 = torch.transpose(c12a.repeat([self.n_iwae, 1]), 0, 1)
        ara = torch.transpose(ara_a.repeat([self.n_iwae, 1]), 0, 1)

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
        self.dR = torch.clamp(theta.dR, 1e-12, 5.0)
        self.dS = torch.clamp(theta.dS, 1e-12, 5.0)

        self.e76 = theta.e76
        self.e81 = theta.e81
        self.aCFP = theta.aCFP
        self.aYFP = theta.aYFP
        self.KGR_76 = theta.KGR_76
        self.KGS_76 = theta.KGS_76
        self.KGR_81 = theta.KGR_81
        self.KGS_81 = theta.KGS_81

        # Condition on device information by mapping param_cond = f(param, d; \phi) where d is one-hot rep of device
        ## if condition_on_device:
        ##     print('Reached here?')
        ##
        ##     ones = torch.tensor([1.0]).repeat([self.n_batch, self.n_iwae])
        ##     self.aR = self.device_conditioner(ones, 'aR', dev_1hot)
        ##     self.aS = self.device_conditioner(ones, 'aS', dev_1hot)
        ## else:
        ##     print('Using Condition on device information')
        ##     self.aR = theta.aR
        ##     self.aS = theta.aS

        self.aR = theta.aR
        self.aS = theta.aS
        self.aI = theta.aI

        self.dA6 = theta.dA6
        self.dA12 = theta.dA12
        self.daiiA = theta.daiiA

        self.nA = torch.clamp(theta.nA, 0.5, 3.0)
        self.eA = theta.eA
        self.KAra = theta.KAra

        # Promoter activity
        self.PBAD = (power(ara, self.nA) + (self.eA * power(self.KAra, self.nA))) / (
            power(ara, self.nA) + power(self.KAra, self.nA)
        )
        self.rC6 = self.dA6 * c6
        self.rC12 = self.dA12 * c12

        # Activation constants for convenience
        nR = torch.clamp(theta.nR, 0.5, 3.0)
        nS = torch.clamp(theta.nS, 0.5, 3.0)
        lb = 1e-12
        ub = 1e0
        if version == 1:
            KR6 = torch.clamp(theta.KR6, lb, ub)
            KR12 = torch.clamp(theta.KR12, lb, ub)
            KS6 = torch.clamp(theta.KS6, lb, ub)
            KS12 = torch.clamp(theta.KS12, lb, ub)
            # self.fracLuxR = torch.clamp((power(KR6*c6, nR) + power(KR12*c12, nR)) / power(1.0 + KR6*c6 + KR12*c12, nR), 1e-6, 1.0)
            # self.fracLasR = torch.clamp((power(KS6*c6, nS) + power(KS12*c12, nS)) / power(1.0 + KS6*c6 + KS12*c12, nS), 1e-6, 1.0)
            self.fracLuxR = (power(KR6 * c6, nR) + power(KR12 * c12, nR)) / power(1.0 + KR6 * c6 + KR12 * c12, nR)
            self.fracLasR = (power(KS6 * c6, nS) + power(KS12 * c12, nS)) / power(1.0 + KS6 * c6 + KS12 * c12, nS)
        else:
            raise Exception("Unknown version of Degrader_Constant: %d" % version)

    def forward(self, t, state):
        x, rfp, yfp, cfp, f530, f480, luxR, lasR, aiiA = torch.unbind(state[:, :, : self.n_species], axis=2)

        # Cells growing or not (not before lag-time)
        gr = self.r * torch.sigmoid(4.0 * (t - self.tlag))

        # Specific growth and dilution
        g = 1.0 - x / self.K
        gamma = gr * g

        # Promoter activity
        boundLuxR = luxR * luxR * self.fracLuxR
        boundLasR = lasR * lasR * self.fracLasR
        P76 = (self.e76 + self.KGR_76 * boundLuxR + self.KGS_76 * boundLasR) / (
            1.0 + self.KGR_76 * boundLuxR + self.KGS_76 * boundLasR
        )
        P81 = (self.e81 + self.KGR_81 * boundLuxR + self.KGS_81 * boundLasR) / (
            1.0 + self.KGR_81 * boundLuxR + self.KGS_81 * boundLasR
        )

        # Right-hand sides
        d_x = gamma * x
        d_rfp = self.rc - (gamma + self.drfp) * rfp
        d_yfp = self.rc * self.aYFP * P81 - (gamma + self.dyfp) * yfp
        d_cfp = self.rc * self.aCFP * P76 - (gamma + self.dcfp) * cfp
        d_f530 = self.rc * self.a530 - gamma * f530
        d_f480 = self.rc * self.a480 - gamma * f480
        d_luxR = self.rc * self.aR - (gamma + self.dR) * luxR
        d_lasR = self.rc * self.aS - (gamma + self.dS) * lasR

        d_aiiA = self.rc * self.aI * self.PBAD - (self.daiiA + (gamma * aiiA))

        d_c6 = x * self.rC6 * aiiA
        d_c12 = x * self.rC12 * aiiA

        dX = torch.stack([d_x, d_rfp, d_yfp, d_cfp, d_f530, d_f480, d_luxR, d_lasR, d_aiiA, d_c6, d_c12,], axis=2,)
        if self.precisions is not None:
            dV = self.precisions(t, state, None, self.n_batch, self.n_iwae)
            return torch.cat([dX, dV], dim=2)
        else:
            return dX


class Degrader_Constant(OdeModel):
    def __init__(self, config):
        super(Degrader_Constant, self).__init__(config)
        self.precisions = ConstantPrecisions(["prec_x", "prec_rfp", "prec_yfp", "prec_cfp"])
        self.species = [
            "OD",
            "RFP",
            "YFP",
            "CFP",
            "F530",
            "F480",
            "LuxR",
            "LasR",
            "AiiA",
            "C6",
            "C12",
        ]
        self.n_species = 11
        self.device = config.device
        self.version = 1

    def initialize_state(self, theta, _treatments):
        n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        zero = torch.zeros([n_batch, n_iwae], device=self.device)

        treatments_transformed = torch.clamp(torch.exp(_treatments) - 1.0, 1e-12, 1e6)
        c6a, c12a, ara_a = torch.unbind(treatments_transformed, axis=1)
        c6 = torch.transpose(c6a.repeat([n_iwae, 1]), 0, 1)
        c12 = torch.transpose(c12a.repeat([n_iwae, 1]), 0, 1)

        x0 = torch.stack(
            [
                theta.init_x,
                theta.init_rfp,
                theta.init_yfp,
                theta.init_cfp,
                zero,
                zero,
                theta.init_luxR,
                theta.init_lasR,
                theta.init_aiiA,
                c6,
                c12,
            ],
            axis=2,
        )
        return x0

    def gen_reaction_equations(self, config, theta, treatments, dev_1hot, condition_on_device=True):
        func = Degrader_Constant_RHS(config, theta, treatments, dev_1hot, condition_on_device, version=self.version,)
        self.aR = func.aR
        self.aS = func.aS
        return func

    def summaries(self, writer, epoch):
        variable_summaries(writer, epoch, self.aR, "aR.conditioned")
        variable_summaries(writer, epoch, self.aS, "aS.conditioned")


class Degrader_Constant_Precisions(OdeModel):
    def __init__(self, config):
        super(Degrader_Constant_Precisions, self).__init__(config)
        self.species = [
            "OD",
            "RFP",
            "YFP",
            "CFP",
            "F530",
            "F480",
            "LuxR",
            "LasR",
            "AiiA",
            "C6",
            "C12",
        ]
        self.n_species = 11
        self.precisions = NeuralPrecisions(self.n_species, config.params.n_hidden_decoder_precisions, 4)
        self.version = 1

    def initialize_state(self, theta, _treatments):
        n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        zero = torch.zeros([n_batch, n_iwae])

        treatments_transformed = torch.clamp(torch.exp(_treatments) - 1.0, 1e-12, 1e6)
        c6a, c12a, ara_a = torch.unbind(treatments_transformed, axis=1)
        c6 = torch.transpose(c6a.repeat([n_iwae, 1]), 0, 1)
        c12 = torch.transpose(c12a.repeat([n_iwae, 1]), 0, 1)

        x0 = torch.stack(
            [
                theta.init_x,
                theta.init_rfp,
                theta.init_yfp,
                theta.init_cfp,
                zero,
                zero,
                theta.init_luxR,
                theta.init_lasR,
                theta.init_aiiA,
                c6,
                c12,
                theta.init_prec_x,
                theta.init_prec_rfp,
                theta.init_prec_yfp,
                theta.init_prec_cfp,
            ],
            axis=2,
        )
        return x0

    def gen_reaction_equations(self, config, theta, treatments, dev_1hot, condition_on_device=False):
        func = Degrader_Constant_RHS(
            config, theta, treatments, dev_1hot, condition_on_device, precisions=self.precisions, version=self.version,
        )
        self.aR = func.aR
        self.aS = func.aS
        return func

    def summaries(self, writer, epoch):
        variable_summaries(writer, epoch, self.aR, "aR.conditioned")
        variable_summaries(writer, epoch, self.aS, "aS.conditioned")
        self.precisions.summaries(writer, epoch)
