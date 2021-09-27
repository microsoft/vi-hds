# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import torch
import torch.nn as nn
import torchdiffeq

from vihds.solvers import modified_euler_integrate, modified_euler_while
from vihds.utils import default_get_value, variable_summaries

# pylint: disable = no-member, not-callable

def power(x, a):
    #return (a * x.log()).exp()
    return x.pow(a)

class OdeFunc(nn.Module):
    def __init__(self, config, _theta, _conditions, _dev1_hot):
        super(OdeFunc, self).__init__()        

    def forward(self, t, y):
        raise NotImplementedError("TODO: write your own forward method")


class OdeModel(nn.Module):
    def __init__(self, config):
        super(OdeModel, self).__init__()
        self.device_depth = config.data.device_depth
        self.n_treatments = len(config.data.conditions)
        self.use_laplace = default_get_value(config.params, 'use_laplace', False, verbose=True)
        self.precisions = None
        self.species = None
        self.n_species = None
        self.relevance = config.data.relevance_vectors
        self.default_devices = config.data.default_devices

    def gen_reaction_equations(self, config, theta, conditions, dev1_hot):
        return OdeFunc(config, theta, conditions, dev1_hot)

    def device_conditioner(self, param, param_name, dev_1hot, use_bias=False, activation='relu'):
        n_iwae = param.shape[1]
        n_batch = param.shape[0]
        param_flat = torch.reshape(param, [n_iwae * n_batch, 1])
        n_inputs = dev_1hot.shape[1]
        conditioner = DeviceConditioner(n_inputs, use_bias=use_bias, activation=activation)

        # tile devices, one per iwae sample
        #dev_relevance = torch.tensor(dev_1hot * self.relevance[param_name])
        dev_relevance = dev_1hot * torch.tensor(self.relevance[param_name])
        param_cond = conditioner(dev_relevance).repeat([n_iwae, 1])
        if param_name in self.default_devices:
            out = param_flat * (1.0 + param_cond)
        else:
            out = param_flat * param_cond
        return out.reshape([n_batch, n_iwae])
        
    def condition_theta(self, theta, dev_1hot, writer, epoch):
        raise NotImplementedError("TODO: write your condition_theta")

    def initialize_state(self, theta, treatments):
        raise NotImplementedError("TODO: write your initialize_state")

    def simulate(self, config, times, theta, conditions, dev_1hot, condition_on_device=True):
        # Initialise ODE simulation with initial conditions and RHS function
        init_state = self.initialize_state(theta, conditions).to(config.device)
        d_states_d_t = self.gen_reaction_equations(config, theta, conditions.to(config.device), dev_1hot).to(config.device)

        # Evaluate ODEs using one of several solvers
        times = times.to(config.device)
        if config.params.solver == 'modeuler':
            sol = modified_euler_integrate(d_states_d_t, init_state, times)
        elif config.params.solver == 'modeulerwhile':
            sol = modified_euler_while(d_states_d_t, init_state, times)
        else:
            integrator = torchdiffeq.odeint_adjoint if config.params.adjoint_solver else torchdiffeq.odeint
            sol = integrator(d_states_d_t, init_state, times, method=config.params.solver)
        return sol.permute(1, 2, 3, 0)

    @classmethod
    def observe(cls, x_sample, _theta):
        x_predict = [
            x_sample[:, :, 0, :],
            x_sample[:, :, 0, :] * x_sample[:, :, 1, :],
            x_sample[:, :, 0, :] * (x_sample[:, :, 2, :] + x_sample[:, :, 4, :]),
            x_sample[:, :, 0, :] * (x_sample[:, :, 3, :] + x_sample[:, :, 5, :])]
        x_predict = torch.stack(x_predict, axis=-1).permute(0, 1, 3, 2)
        return x_predict

    def expand_precisions(self, theta, times, x_states):
        return self.precisions.expand(theta, len(times), x_states)

class DeviceConditioner(nn.Module):
    """
    Returns a 1D parameter conditioned on device
    ::NOTE:: condition_on_device is a closure over n_iwae, n_batch, dev_1hot_rep
    """
    def __init__(self, n_inputs, use_bias=False, activation='relu'):
        super(DeviceConditioner, self).__init__()
        self.cond = nn.Linear(n_inputs, 1, use_bias)
        nn.init.xavier_uniform_(self.cond.weight)
        nn.init.normal_(self.cond.weight, mean=2.0, std=1.5)
        if activation == 'relu':
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.cond(x)
        x = self.act(x)
        return x


class NeuralStates(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_states, n_latents):
        super(NeuralStates, self).__init__()
        self.n_latents = n_latents
        self.n_states = n_states
        self.states_hidden = nn.Linear(n_inputs, n_hidden)
        nn.init.xavier_uniform_(self.states_hidden.weight)
        self.hidden_act = nn.ReLU()
        self.states_production = nn.Linear(n_hidden, n_states)
        nn.init.xavier_uniform_(self.states_production.weight)
        self.prod_act = nn.Sigmoid()
        self.states_degradation = nn.Linear(n_hidden, n_states)
        nn.init.xavier_uniform_(self.states_degradation.weight)
        self.degr_act = nn.Sigmoid()

    def forward(self, x, constants):
        aug = torch.cat([x, constants], dim=2)
        hidden = self.hidden_act(self.states_hidden(aug))
        dx = self.prod_act(self.states_production(hidden)) - self.degr_act(self.states_degradation(hidden)) * x
        return dx
    
    def summaries(self, writer, epoch):
        '''Tensorboard summaries for neural states'''
        if (writer is not None):
            for name in ['states_hidden', 'states_production', 'states_degradation']:
                module = getattr(self, name)
                variable_summaries(writer, epoch, module.weight, name + "_weights", False)
                variable_summaries(writer, epoch, module.bias, name + "_bias", False)
