# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.constraints import NonNeg
import numpy as np
import pdb

from solvers import modified_euler_integrate, integrate_while
from utils import default_get_value, variable_summaries
from procdata import ProcData

def power(x, a):
    return tf.exp(a * tf.log(x))

def log_prob_laplace(x_obs, x_post_sample, log_precisions, precisions):
    log_p_x = tf.log(0.5) + log_precisions - precisions * tf.abs(x_post_sample - x_obs)
    return log_p_x

def log_prob_gaussian(x_obs, x_post_sample, log_precisions, precisions):
    # https://en.wikipedia.org/wiki/Normal_distribution
    log_p_x = -0.5 * tf.log(2.0 * np.pi) + 0.5 * log_precisions - 0.5 * precisions * tf.square(x_post_sample - x_obs)
    return log_p_x

def expand_constant_precisions(precision_list):
    # e.g.: precision_list = [theta.prec_x, theta.prec_fp, theta.prec_fp, theta.prec_fp ]
    precisions = tf.stack(precision_list, axis=-1)
    log_precisions = tf.log(precisions)
    precisions = tf.expand_dims(precisions, 2)
    log_precisions = tf.expand_dims(log_precisions, 2)
    return log_precisions, precisions

def expand_decayed_precisions(precision_list): # pylint: disable=unused-argument
    raise NotImplementedError("TODO: expand_decayed_precisions")

class BaseModel(object):
    # We need an init_with_params method separate from the usual __init__, because the latter is
    # called automatically with no arguments by pyyaml on creation, and we need a way to feed
    # params (from elsewhere in the YAML structure) into it. It would really be better construct
    # it properly after the structure has been loaded.
    # pylint: disable=attribute-defined-outside-init
    def init_with_params(self, params, procdata : ProcData):
        self.params = params
        self.relevance = procdata.relevance_vectors
        self.default_devices = procdata.default_devices
        self.device_depth = procdata.device_depth
        self.n_treatments = len(procdata.conditions)
        self.use_laplace = default_get_value(self.params, 'use_laplace', False, verbose=True)
        self.precision_type = default_get_value(self.params, 'precision_type', 'constant', verbose=True)
        self.species = None
        self.nspecies = None
        #self.layers = []

    def gen_reaction_equations(self, theta, conditions, dev_1hot, condition_on_device=True):
        raise NotImplementedError("TODO: write your gen_reaction_equations")

    def get_precision_list(self, theta):
        return [theta.prec_x, theta.prec_rfp, theta.prec_yfp, theta.prec_cfp]

    def device_conditioner(self, param, param_name, dev_1hot, kernel_initializer='glorot_uniform', use_bias=False, activation=tf.nn.relu):
        """
        Returns a 1D parameter conditioned on device
        ::NOTE:: condition_on_device is a closure over n_iwae, n_batch, dev_1hot_rep
        """
        n_iwae = tf.shape(param)[1]
        n_batch = tf.shape(param)[0]
        param_flat = tf.reshape(param, [n_iwae * n_batch, 1])
        cond_nn = tf.keras.layers.Dense(1, use_bias=use_bias, activation=activation, kernel_initializer=kernel_initializer)
        # tile devices, one per iwae sample
        dev_1hot_rep = tf.tile(dev_1hot * self.relevance[param_name], [n_iwae, 1])
        param_cond = cond_nn(dev_1hot_rep)
        if param_name in self.default_devices:
            return tf.reshape(param_flat * (1.0 + param_cond), [n_batch, n_iwae])
        else:
            return tf.reshape(param_flat * param_cond, [n_batch, n_iwae])

    def initialize_state(self, theta, treatments):
        raise NotImplementedError("TODO: write your initialize_state")

    def simulate(self, theta, times, conditions, dev_1hot, solver, condition_on_device=True):
        init_state = self.initialize_state(theta, conditions)
        d_states_d_t = self.gen_reaction_equations(theta, conditions, dev_1hot, condition_on_device)
        if solver == 'modeuler':
            # Evaluate ODEs using Modified-Euler
            t_state, f_state = modified_euler_integrate(d_states_d_t, init_state, times)
            t_state_tr = tf.transpose(t_state, [0, 1, 3, 2])
            f_state_tr = tf.transpose(f_state, [0, 1, 3, 2])
        elif solver == 'modeulerwhile':
            # Evaluate ODEs using Modified-Euler
            t_state, f_state = integrate_while(d_states_d_t, init_state, times, algorithm='modeuler')
            t_state_tr = tf.transpose(t_state, [1, 2, 0, 3])
            f_state_tr = None
        elif solver == 'rk4':
            # Evaluate ODEs using 4th order Runge-Kutta
            t_state, f_state = integrate_while(d_states_d_t, init_state, times, algorithm='rk4')
            t_state_tr = tf.transpose(t_state, [1, 2, 0, 3])
            f_state_tr = None
        else:
            raise NotImplementedError("Solver <%s> is not implemented" % solver)
        return t_state_tr, f_state_tr

    @classmethod
    def observe(cls, x_sample, _theta):
        x_predict = [
            x_sample[:, :, :, 0],
            x_sample[:, :, :, 0] * x_sample[:, :, :, 1],
            x_sample[:, :, :, 0] * (x_sample[:, :, :, 2] + x_sample[:, :, :, 4]),
            x_sample[:, :, :, 0] * (x_sample[:, :, :, 3] + x_sample[:, :, :, 5])]
        x_predict = tf.stack(x_predict, axis=-1)
        return x_predict

    def add_time_dimension(self, p, x):
        time_steps = x.shape[1]
        p = tf.tile(p, [1, 1, time_steps, 1], name="time_added")
        return p

    def expand_precisions_by_time(self, theta, _x_predict, x_obs, _x_sample):
        precision_list = self.get_precision_list(theta)
        log_prec, prec = self.expand_precisions(precision_list)
        log_prec = self.add_time_dimension(log_prec, x_obs)
        prec = self.add_time_dimension(prec, x_obs)
        if self.precision_type == "decayed":
            time_steps = x_obs.shape[1]
            lin_timesteps = tf.reshape(tf.linspace(1.0, time_steps.value, time_steps.value), [1, 1, time_steps, 1])
            prec = prec / lin_timesteps
            log_prec = log_prec - tf.log(lin_timesteps)
        return log_prec, prec

    @classmethod
    def expand_precisions(cls, precision_list):
        return expand_constant_precisions(precision_list)

    def log_prob_observations(self, x_predict, x_obs, theta, x_sample):
        log_precisions, precisions = self.expand_precisions_by_time(theta, x_predict, x_obs, x_sample)
        # expand x_obs for the iw samples in x_post_sample
        x_obs_ = tf.expand_dims(x_obs, 1)
        lpfunc = log_prob_laplace if self.use_laplace else log_prob_gaussian
        log_prob = lpfunc(x_obs_, x_predict, log_precisions, precisions)
        # sum along the time and observed species axes
        #log_prob = tf.reduce_sum(log_prob, [2, 3])
        # sum along the time axis
        log_prob = tf.reduce_sum(log_prob, 2)
        return log_prob

class NeuralPrecisions(object):
    def __init__(self, nspecies, n_hidden_precisions, inputs = None, hidden_activation = tf.nn.tanh):
        '''Initialize neural precisions layers'''
        self.nspecies = nspecies
        if inputs is None:
            inputs = self.nspecies+1
        inp = Dense(n_hidden_precisions, activation = hidden_activation, use_bias=True, name = "prec_hidden", input_shape=(inputs,))
        act_layer = Dense(4, activation = tf.nn.sigmoid, name = "prec_act", bias_constraint = NonNeg())
        deg_layer = Dense(4, activation = tf.nn.sigmoid, name = "prec_deg", bias_constraint = NonNeg())
        self.act = Sequential([inp, act_layer])
        self.deg = Sequential([inp, deg_layer])

        for layer in [inp, act_layer, deg_layer]:
            weights, bias = layer.weights
            variable_summaries(weights, layer.name + "_kernel", False)
            variable_summaries(bias, layer.name + "_bias", False)

    def __call__(self, t, state, n_batch, n_iwae):
        reshaped_state = tf.reshape(state[:,:,:-4], [n_batch*n_iwae, self.nspecies])
        reshaped_var_state = tf.reshape(state[:,:,-4:], [n_batch*n_iwae, 4])
        t_expanded = tf.tile( [[t]], [n_batch*n_iwae, 1] )
        ZZ_vrs = tf.concat( [ t_expanded, reshaped_state ], axis=1 )
        vrs = tf.reshape(self.act(ZZ_vrs) - self.deg(ZZ_vrs)*reshaped_var_state, [n_batch, n_iwae, 4])
        return vrs