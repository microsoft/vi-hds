# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
from vihds.ode import OdeModel, OdeFunc, power
from vihds.precisions import ConstantPrecisions, NeuralPrecisions
from vihds.utils import variable_summaries
import torch

# pylint: disable = no-member, not-callable

class DR_Constant_RHS(OdeFunc):
    def __init__(self, config, theta, treatments, dev1_hot, precisions=None, version=1):
        super(DR_Constant_RHS, self).__init__(config, theta, treatments, dev1_hot)

        # Pass in a class instance for dynamic (neural) precisions. If None, then it is expected that you have 
        # latent variables for the precisions, as these will be assigned as part of BaseModel.expand_precisions_by_time()
        self.precisions = precisions
        
        self.n_batch = theta.get_n_batch()
        self.n_iwae = theta.get_n_samples()
        self.n_species = 8

        # tile treatments, one per iwae sample
        treatments_transformed = torch.clamp(torch.exp(treatments) - 1.0, 1e-12, 1e6)
        c6a, c12a = torch.unbind(treatments_transformed, axis=1)
        c6 = torch.transpose(c6a.repeat([self.n_iwae, 1]),0,1)
        c12 = torch.transpose(c12a.repeat([self.n_iwae, 1]),0,1)
        
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

        self.aR = theta.aR
        self.aS = theta.aS

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
            #self.fracLuxR = torch.clamp((power(KR6*c6, nR) + power(KR12*c12, nR)) / power(1.0 + KR6*c6 + KR12*c12, nR), 1e-6, 1.0)
            #self.fracLasR = torch.clamp((power(KS6*c6, nS) + power(KS12*c12, nS)) / power(1.0 + KS6*c6 + KS12*c12, nS), 1e-6, 1.0)
            self.fracLuxR = (power(KR6*c6, nR) + power(KR12*c12, nR)) / power(1.0 + KR6*c6 + KR12*c12, nR)
            self.fracLasR = (power(KS6*c6, nS) + power(KS12*c12, nS)) / power(1.0 + KS6*c6 + KS12*c12, nS)
        elif version == 2:
            eS6 = torch.clamp(theta.eS6, lb, ub)
            eR12 = torch.clamp(theta.eR12, lb, ub)
            self.fracLuxR = power(c6, nR) + power(eR12*c12, nR)
            self.fracLasR = power(eS6*c6, nS) + power(c12, nS)
        else:
            raise Exception("Unknown version of DR_Constant: %d" % version)

    def forward(self, t, state):
        x, rfp, yfp, cfp, f530, f480, luxR, lasR = torch.unbind(state[:,:,:self.n_species], axis=2)

        # Cells growing or not (not before lag-time)
        gr = self.r * torch.sigmoid(4.0 * (t -  self.tlag))

        # Specific growth and dilution
        g = (1.0 - x / self.K)
        gamma = gr * g

        # Promoter activity
        boundLuxR = luxR * luxR * self.fracLuxR
        boundLasR = lasR * lasR * self.fracLasR
        P76 = (self.e76 + self.KGR_76 * boundLuxR + self.KGS_76 * boundLasR) / (1.0 + self.KGR_76 * boundLuxR + self.KGS_76 * boundLasR)
        P81 = (self.e81 + self.KGR_81 * boundLuxR + self.KGS_81 * boundLasR) / (1.0 + self.KGR_81 * boundLuxR + self.KGS_81 * boundLasR)

        # Right-hand sides
        d_x = gamma * x
        d_rfp = self.rc - (gamma + self.drfp) * rfp
        d_yfp = self.rc * self.aYFP * P81 - (gamma + self.dyfp) * yfp
        d_cfp = self.rc * self.aCFP * P76 - (gamma + self.dcfp) * cfp
        d_f530 = self.rc * self.a530 - gamma * f530
        d_f480 = self.rc * self.a480 - gamma * f480
        d_luxR = self.rc * self.aR - (gamma + self.dR) * luxR
        d_lasR = self.rc * self.aS - (gamma + self.dS) * lasR

        dX = torch.stack([d_x, d_rfp, d_yfp, d_cfp, d_f530, d_f480, d_luxR, d_lasR], axis=2)
        if self.precisions is not None:
            dV = self.precisions(t, state, None, self.n_batch, self.n_iwae)
            return torch.cat([dX, dV], dim=2)
        else:
            return dX


class DR_Constant(OdeModel):
    def __init__(self, config):
        super(DR_Constant, self).__init__(config)
        self.precisions = ConstantPrecisions(['prec_x','prec_rfp','prec_yfp','prec_cfp'])
        self.species = ['OD', 'RFP', 'YFP', 'CFP', 'F530', 'F480', 'LuxR', 'LasR']
        self.n_species = 8
        self.device = config.device
        self.version = 1

    def condition_theta(self, theta, dev_1hot, writer, epoch):
        '''Condition on device information by mapping param_cond = f(param, d; \phi) where d is one-hot rep of device'''
        n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        ones = torch.tensor([1.0]).repeat([n_batch, n_iwae])
        theta.aR = self.device_conditioner(ones, 'aR', dev_1hot)
        theta.aS = self.device_conditioner(ones, 'aS', dev_1hot)
        return theta

    def initialize_state(self, theta, _treatments):
        n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        zero = torch.zeros([n_batch, n_iwae], device=self.device)
        x0 = torch.stack([theta.init_x, theta.init_rfp, theta.init_yfp, theta.init_cfp, zero, zero, theta.init_luxR, theta.init_lasR], axis=2)
        return x0

    def gen_reaction_equations(self, config, theta, treatments, dev_1hot):
        func = DR_Constant_RHS(config, theta, treatments, dev_1hot, version=self.version)
        self.aR = func.aR
        self.aS = func.aS
        return func

    def summaries(self, writer, epoch):
        variable_summaries(writer, epoch, self.aR, 'aR.conditioned')
        variable_summaries(writer, epoch, self.aS, 'aS.conditioned')


class DR_Constant_V2(DR_Constant):
    def __init__(self, config):
        super(DR_Constant_V2, self).__init__(config)
        self.version = 2


class DR_Constant_Precisions(DR_Constant):
    def __init__(self, config):
        super(DR_Constant_Precisions, self).__init__(config)
        self.species = ['OD', 'RFP', 'YFP', 'CFP', 'F530', 'F480', 'LuxR', 'LasR']
        self.n_species = 8
        self.precisions = NeuralPrecisions(self.n_species, config.params.n_hidden_decoder_precisions, 4)
        self.version = 1
        
    def initialize_state(self, theta, _treatments):
        n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        zero = torch.zeros([n_batch, n_iwae])
        x0 = torch.stack([theta.init_x, theta.init_rfp, theta.init_yfp, theta.init_cfp, zero, zero, theta.init_luxR, theta.init_lasR, theta.init_prec_x, theta.init_prec_rfp, theta.init_prec_yfp, theta.init_prec_cfp], axis=2)
        return x0

    def gen_reaction_equations(self, config, theta, treatments, dev_1hot):
        func = DR_Constant_RHS(config, theta, treatments, dev_1hot, precisions=self.precisions, version=self.version)
        self.aR = func.aR
        self.aS = func.aS
        return func

    def summaries(self, writer, epoch):
        variable_summaries(writer, epoch, self.aR, 'aR.conditioned')
        variable_summaries(writer, epoch, self.aS, 'aS.conditioned')
        self.precisions.summaries(writer, epoch)


class DR_Constant_Precisions_V2(DR_Constant_Precisions):
    def __init__(self, config):
        super(DR_Constant_Precisions_V2, self).__init__(config)
        self.version = 2