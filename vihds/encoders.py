# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
from torch.utils.data import ConcatDataset
import numpy as np
from collections import OrderedDict
from vihds.distributions import ChainedDistribution

# pylint: disable = no-member, not-callable
IDENTITY = "identity"
POSITIVE = "positive"

class ConditionalEncoder(nn.Module):
    def __init__(self, n_channels, n_obs, params):
        super(ConditionalEncoder, self).__init__()
        n_filters = params.n_filters
        filter_size = params.filter_size
        pool_size = params.pool_size
        self.n_outputs = params.n_hidden
        data_format = params.data_format
        lambda_l2 = params.lambda_l2
        lambda_l2_hidden = params.lambda_l2_hidden
    
        n_conv = n_obs - (filter_size - 1)
        #print("n_outputs_conv:", n_outputs_conv)
        n_pool = n_conv - (pool_size - 1)
        #print("n_outputs_pool:", n_outputs_pool)
        n_hidden_layer = n_pool * n_filters
        #print("n_outputs:", n_outputs)

        # Define layers
        self.conv = nn.Conv1d(n_channels, n_filters, filter_size)
        nn.init.orthogonal_(self.conv.weight)
        #TODO: Add kernel weight regularization somehow
        #kernel_regularizer=tf.keras.regularizers.l2(lambda_l2),
        self.pool = nn.AvgPool1d(pool_size, stride=1)
        self.lin  = nn.Linear(n_hidden_layer, self.n_outputs)
        nn.init.orthogonal_(self.lin.weight)
        #TODO: Add kernel weight regularization somehow
        #kernel_regularizer=tf.keras.regularizers.l2(lambda_l2_hidden))
        if params.transfer_func == "tanh":
            self.act = nn.Tanh()
        else:
            raise Exception("Unknown activation layer %s"%params["transfer_func"])
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        x = self.act(x)
        return x

class LocalAndGlobal:
    """Convenience class to hold any tuple of local, global-conditional and global values."""
    def __init__(self, loc, glob_cond, glob, const):
        '''Initialiser'''
        self.loc = loc
        self.glob_cond = glob_cond
        self.glob = glob
        self.const = const

    @classmethod
    def from_list(self, seq):
        return LocalAndGlobal(seq[0], seq[1], seq[2], seq[3])

    def to_list(self):
        return [self.loc, self.glob_cond, self.glob, self.const]

    def sum(self):
        return self.loc + self.glob_cond + self.glob + self.const

    def concat(self, name):
        """Returns a concatenation of the items."""
        concatenated = ChainedDistribution(name=name)
        for chained in self.to_list():
            for item_name, distribution in chained.distributions.items():
                concatenated.add_distribution(item_name, distribution, chained.slot_dependencies[item_name])
        return concatenated

    def diagnostic_printout(self, prefix):
        print('%s-LOCAL\n%s' % (prefix, self.loc))
        print('%s-GLOBAL-COND\n%s' % (prefix, self.glob_cond))
        print('%s-GLOBAL\n%s' % (prefix, self.glob))
        print('%s-CONSTANT\n%s' % (prefix, self.const))


#TODO: Establish if this is necessary here. Feels like a TensorFlow hack...
def constrain_parameter(free_param, free_to_constrained):
    if free_to_constrained == IDENTITY:
        constrained = free_param
    elif free_to_constrained == POSITIVE:
        constrained = free_param.exp()
    else:
        raise NotImplementedError("unknown free_to_constrained = %s" % free_to_constrained)
    return constrained

class Q_Distribution(nn.Module):
    def __init__(self, condition_data, condition_treatments, condition_devices, shapes, description, name):
        super(Q_Distribution, self).__init__()
        self.condition_data = condition_data
        self.condition_treatments = condition_treatments
        self.condition_devices = condition_devices
        self.n_inputs = 0
        if condition_data:
            self.n_inputs += shapes[0]
        if condition_treatments:
            self.n_inputs += shapes[1]
        if condition_devices:
            self.n_inputs += shapes[2]
        self.description = description
        self.name = name

    def forward(self):
        pass

class Q_Local(Q_Distribution):
    def __init__(self, shapes, description, distribution_name, use_bias, stop_grad):
        super(Q_Local, self).__init__(True, description.conditioning['treatments'], 
            description.conditioning['devices'], shapes, description, distribution_name)
        
        self.layers = nn.ModuleDict()
        for free_name in description.free_params:
            layer = nn.Linear(self.n_inputs, 1, use_bias)
            self.layers.update({free_name: layer})
            #TODO: Add regularization (tf.keras.regularizers.l2(0.01) by default)

    def forward(self, delta_obs, conds, devs):
        x = torch.Tensor([])
        if self.condition_data:
            x = torch.cat((x, delta_obs), 1)
        if self.condition_treatments:
            x = torch.cat((x, conds), 1)
        if self.condition_devices:
            x = torch.cat((x, devs), 1)
        params = OrderedDict()
        for free_name, constrained_name, free_to_constrained in zip(
                self.description.free_params, self.description.params, self.description.free_to_constrained):
            
            free_param = self.layers[free_name](x)
            #TODO: Torch equivalent of tf.stop_gradient
            #if stop_grad:
            #    free_param = tf.stop_gradient(free_param)  # eliminate score function term from autodiff
            constrained = constrain_parameter(free_param, free_to_constrained)
            params[free_name] = free_param
            params[constrained_name] = constrained
        
        for other_param_name, other_param_value in self.description.other_params.items():
            params[other_param_name] = other_param_value

        new_distribution = self.description.class_type(wait_for_assigned=True, variable=True)
        new_distribution.assign_free_and_constrained(**params)
        return new_distribution

class Q_Global_Cond(Q_Distribution):
    def __init__(self, shapes, description, name, use_bias, stop_grad):
        super(Q_Global_Cond, self).__init__(False, description.conditioning["treatments"], 
            description.conditioning["devices"], shapes, description, name)
        
        self.layers = nn.ModuleDict()
        for free_name in description.free_params:
            self.layers.update({free_name: nn.Linear(self.n_inputs, 1, use_bias)})

    def forward(self, _x, conds, devs):
        x = torch.Tensor([])  # Ignore the encoded data for global parameters
        if self.condition_treatments:
            x = torch.cat((x, conds), 1)
        if self.condition_devices:
            x = torch.cat((x, devs), 1)
        params = OrderedDict()
        for free_name, constrained_name, free_to_constrained in zip(
                self.description.free_params, self.description.params, self.description.free_to_constrained):
            
            free_param = self.layers[free_name](x)
            #TODO: Torch equivalent of tf.stop_gradient
            #if stop_grad:
            #    free_param = tf.stop_gradient(free_param)  # eliminate score function term from autodiff
            #TODO: Implement Tensorboard summaries
            #variable_summaries(free_param, 'nn_%s'%name, plot_histograms)
            constrained = constrain_parameter(free_param, free_to_constrained)
            params[free_name] = free_param
            params[constrained_name] = constrained
        
        for other_param_name, other_param_value in self.description.other_params.items():
            params[other_param_name] = other_param_value

        new_distribution = self.description.class_type(wait_for_assigned=True, variable=True)
        new_distribution.assign_free_and_constrained(**params)
        return new_distribution

class Q_Global(Q_Distribution):
    def __init__(self, description, name):
        super(Q_Global, self).__init__(False, False, False, (0,0,0), description, name)
        self.free_params = nn.ParameterDict()
        for free_name, init_free in zip(self.description.free_params, self.description.init_free_params):
            self.free_params.update({free_name: nn.Parameter(torch.Tensor([init_free]))})

    def forward(self):
        params = OrderedDict()
        for free_name, constrained_name, free_to_constrained in zip(self.description.free_params, 
            self.description.params, self.description.free_to_constrained):

            free_param = self.free_params[free_name]
            constrained = constrain_parameter(free_param, free_to_constrained)
            params[free_name] = free_param
            params[constrained_name] = constrained

        for other_param_name, other_param_value in self.description.other_params.items():
            params[other_param_name] = other_param_value

        new_distribution = self.description.class_type(wait_for_assigned=True, variable=False)
        new_distribution.assign_free_and_constrained(**params)
        return new_distribution

class Q_Constant(Q_Distribution):
    def __init__(self, description, name):
        super(Q_Constant, self).__init__(False, False, False, (0,0,0), description , name)

    def forward(self):
        params = OrderedDict()
        for free_name, init_free in zip(self.description.free_params, self.description.init_free_params):
            free_param = torch.Tensor([init_free])
            params[free_name] = free_param
        new_distribution = self.description.class_type(wait_for_assigned=True, variable=False)
        new_distribution.assign_free_and_constrained(**params)
        return new_distribution

def build_q_unconditioned(parameters, constant, verbose, stop_grad=False):
    q = nn.ModuleDict()
    if constant:
        attr = "_constant"
    else:
        attr = "_global"
    if not hasattr(parameters, attr):
        print("- Found no %s parameters"%attr[1:])
        return q

    distribution_descriptions = getattr(parameters, attr)
    for distribution_name in distribution_descriptions.list_of_params:
        description = getattr(distribution_descriptions, distribution_name)
        if constant:
            qi = Q_Constant(description, distribution_name)
        else:
            qi = Q_Global(description, distribution_name)
        q.update({distribution_name: qi})
    return q

def build_q_conditioned(parameters, shapes, local, verbose, stop_grad=False, plot_histograms=False):
    q = nn.ModuleDict()
    if local:
        attr = "_local"
    else:
        attr = "_global_cond"
    if not hasattr(parameters, attr):
        print("- Found no %s parameters" % attr[1:])
        return q

    distribution_descriptions = getattr(parameters, attr)
    for distribution_name in distribution_descriptions.list_of_params:
        description = getattr(distribution_descriptions, distribution_name)
        if local:
            qi = Q_Local(shapes, description, distribution_name, True, stop_grad)
        else:
            qi = Q_Global_Cond(shapes, description, distribution_name, False, stop_grad)
        q.update({distribution_name: qi})
    return q

def build_p_unconditioned(parameters, constant, verbose):
    if constant:
        attr = "_constant"
    else:
        attr = "_global"
    p = ChainedDistribution(name="p_global")
    if not hasattr(parameters, attr):
        print("- Found no %s parameters"%attr[1:])
        return p

    distribution_descriptions = getattr(parameters, attr)
    for distribution_name in distribution_descriptions.list_of_params:
        description = getattr(distribution_descriptions, distribution_name)
        params = OrderedDict()
        for pname, value in description.defaults.items():
            if pname not in params:
                if value is None:
                    params[pname] = None
                else:
                    params[pname] = torch.tensor([value])
        new_distribution = description.class_type(**params)
        p.add_distribution(distribution_name, new_distribution)

    return p

def build_p_conditioned(parameters, local, verbose):
    if local:
        attr = "_local"
    else:
        attr = "_global_cond"
    p = ChainedDistribution(name="p"+attr)
    if not hasattr(parameters, attr):
        print("- Found no %s params"%attr[1:])
        return p
    distribution_descriptions = getattr(parameters, attr)
    for distribution_name in distribution_descriptions.list_of_params:
        description = getattr(distribution_descriptions, distribution_name)
        params = OrderedDict()
        for pname, value in description.defaults.items():
            if pname not in params:
                if value is None:
                    params[pname] = None
                else:
                    params[pname] = torch.tensor([value])
        new_distribution = description.class_type(**params)
        p.add_distribution(distribution_name, new_distribution)
    return p

class Encoder(nn.Module):
    '''
    Variational autoencoder (VAE) for hierarchical parameters: 
    - Local (separate values for each data instance)
    - Group-level/global-conditioned (one value for each group, conditionedon device/group information)
    - Global (single inferred value)
    '''
    def __init__(self, parameters, data, verbose):
        super(Encoder, self).__init__()
        print("Initialising encoder")
        self.verbose = verbose
        self.parameters = parameters
        
        if isinstance(data.train.dataset, ConcatDataset):
            # Use the time vector with the fewest time-points for the encoder
            raise NotImplementedError("Can't handle multiple datasets yet")
            #self.n_species = data.train.dataset.datasets[0].n_species
            #self.n_times = np.min([d.n_times for d in data.train.dataset.datasets])
        else:
            self.n_species = data.train.dataset.n_species
            self.n_times   = data.train.dataset.n_times
        self.set_up_q(data)
        self.set_up_p()

    def set_up_q(self, dataset):
        self.conditional = ConditionalEncoder(self.n_species, self.n_times - 1, self.parameters.params_dict)
        shapes = (self.conditional.n_outputs, dataset.n_conditions, dataset.depth)
        self.q_local_defs = build_q_conditioned(self.parameters, shapes, True, self.verbose)
        shapes = (0, dataset.n_conditions, dataset.depth)
        self.q_global_cond_defs = build_q_conditioned(self.parameters, shapes, False, self.verbose)
            #kernel_regularizer=tf.keras.regularizers.l2(0.01)
        self.q_global_defs = build_q_unconditioned(self.parameters, False, self.verbose)
        self.q_constant_defs = build_q_unconditioned(self.parameters, True, self.verbose)

    def evaluate_q(self, data):
        q_local = ChainedDistribution(name="q_local")
        delta_obs = data.observations[:, :, 1:self.n_times] - data.observations[:, :, :self.n_times-1]
        encoded_data = self.conditional(delta_obs)
        for k, qi in self.q_local_defs.items():
            v = qi(encoded_data, data.inputs, data.dev_1hot)
            q_local.add_distribution(k, v)
        q_global_cond = ChainedDistribution(name="q_global_cond")
        for k, qi in self.q_global_cond_defs.items():
            v = qi(encoded_data, data.inputs, data.dev_1hot)
            q_global_cond.add_distribution(k, v)
        q_global = ChainedDistribution(name="q_global")
        for k, qi in self.q_global_defs.items():
            v = qi()
            q_global.add_distribution(k, v)
        q_constant = ChainedDistribution(name="q_constant")
        for k, qi in self.q_constant_defs.items():
            v = qi()
            q_constant.add_distribution(k, v)
        q = LocalAndGlobal(q_local, q_global_cond, q_global, q_constant).concat("q")
        
        return q
    
    def set_up_p(self):
        # prior: local: may have some dependencies in theta (in hierarchy, local, etc)
        p_local = build_p_conditioned(self.parameters, True, self.verbose)
        p_global_cond = build_p_conditioned(self.parameters, False, self.verbose)
        # prior: global should be fully defined in parameters
        p_global = build_p_unconditioned(self.parameters, False, self.verbose)
        p_constant = build_p_unconditioned(self.parameters, True, self.verbose)
        p_vals = LocalAndGlobal(p_local, p_global_cond, p_global, p_constant)
        self.p = p_vals.concat("p")

    def forward(self, data):
        q = self.evaluate_q(data)
        return q