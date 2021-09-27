# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import torch
from torch import nn

# pylint: disable=no-member

from vihds.utils import variable_summaries

# The overall strategy here is that the precisions are specified in one of 3 ways:
# - Constant, using the ConstantPrecisions class
# - Neural components, dprec/dt = N1(x) - N2(x).prec, using NeuralPrecisions()
# - Neural components, prec = 1/var, dvar/dt = N1(x) - N2(x).var, using NeuralPrecisions(inverse=True)


class ConstantPrecisions(nn.Module):
    """Initialize constant precisions class"""

    def __init__(self, precision_vars):
        super(ConstantPrecisions, self).__init__()
        self.dynamic = False
        self.precision_vars = precision_vars
        # e.g.: precision_vars = ['prec_x', 'prec_rfp', 'prec_yfp', 'prec_cfp']

    def add_time_dimension(self, p, n_times):
        p = p.repeat([1, 1, 1, n_times])
        return p

    def expand(self, theta, n_times, x_states):
        precision_list = [getattr(theta, v) for v in self.precision_vars]
        precisions = torch.stack(precision_list, axis=-1)
        precisions = torch.unsqueeze(precisions, 3).repeat([1, 1, 1, n_times])
        return x_states, precisions

    def summaries(self, _writer, _epoch):
        pass


class NeuralPrecisions(nn.Module):
    """Initialize neural precisions layers"""

    def __init__(
        self, n_inputs, n_hidden_precisions, n_outputs, inverse=False, hidden_activation=nn.Tanh,
    ):
        super(NeuralPrecisions, self).__init__()
        print("- Initialising neural precisions with %d hidden layers" % n_hidden_precisions)
        self.dynamic = True
        self.inverse = inverse
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        n_inputs = self.n_inputs + 1

        if n_hidden_precisions < 1:
            self.prec_production = nn.Linear(n_inputs, n_outputs)  # bias_constraint = NonNeg()
            nn.init.xavier_uniform_(self.prec_production.weight)
            self.prec_degradation = nn.Linear(n_inputs, n_outputs)  # bias_constraint = NonNeg()
            nn.init.xavier_uniform_(self.prec_degradation.weight)
            self.prod = nn.Sequential(hidden_activation(), self.prec_production, nn.Sigmoid())
            self.degr = nn.Sequential(hidden_activation(), self.prec_degradation, nn.Sigmoid())
        else:
            self.prec_hidden = nn.Linear(n_inputs, n_hidden_precisions)
            nn.init.xavier_uniform_(self.prec_hidden.weight)
            inp_act = hidden_activation()
            # TODO: Add bias constraint
            self.prec_production = nn.Linear(n_hidden_precisions, n_outputs)  # bias_constraint = NonNeg()
            # self.prec_production.bias = nn.Parameter(torch.zeros_like(self.prec_production.bias).clamp(min=0.0))
            nn.init.xavier_uniform_(self.prec_production.weight, gain=0.5)
            self.prec_degradation = nn.Linear(n_hidden_precisions, n_outputs)  # bias_constraint = NonNeg()
            # self.prec_degradation.bias = nn.Parameter(torch.zeros_like(self.prec_degradation.bias).clamp(min=0.0))
            nn.init.xavier_uniform_(self.prec_degradation.weight, gain=1)
            self.prod = nn.Sequential(self.prec_hidden, inp_act, self.prec_production, nn.Sigmoid())
            self.degr = nn.Sequential(self.prec_hidden, inp_act, self.prec_degradation, nn.Sigmoid())

    def forward(self, t, state, constants, n_batch, n_iwae):
        reshaped_state = state[:, :, : -self.n_outputs]
        reshaped_var_state = state[:, :, -self.n_outputs :]
        t_expanded = t.repeat([n_batch, n_iwae, 1])
        if constants is not None:
            x = torch.cat([t_expanded, reshaped_state, constants], dim=2)
        else:
            x = torch.cat([t_expanded, reshaped_state], dim=2)
        xa = self.prod(x)
        xd = self.degr(x)
        vrs = xa - xd * reshaped_var_state
        return vrs

    def expand(self, theta, _n_times, x_states):
        if self.inverse:
            prec = 1.0 / x_states[:, :, -self.n_outputs :, :]
        else:
            prec = x_states[:, :, -self.n_outputs :, :]
        return x_states[:, :, : -self.n_outputs, :], prec

    def summaries(self, writer, epoch):
        """Tensorboard summaries for neural precisions"""
        if writer is not None:
            for name in ["prec_hidden", "prec_production", "prec_degradation"]:
                if hasattr(self, name):
                    module = getattr(self, name)
                    variable_summaries(writer, epoch, module.weight, name + "_weights", False)
                    variable_summaries(writer, epoch, module.bias, name + "_bias", False)
