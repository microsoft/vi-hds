# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

import tensorflow as tf
import numpy as np
import pdb

from solvers import modified_euler_integrate, modified_euler_integrate_while
from utils import default_get_value

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
    def init_with_params(self, params, relevance):
        self.params = params
        self.relevance = relevance
        self.use_laplace = default_get_value(self.params, 'use_laplace', False, verbose=True)
        self.precision_type = default_get_value(self.params, 'precision_type', 'constant', verbose=True)
        self.species = ['OD', 'RFP', 'YFP', 'CFP']

    def get_list_of_constants(self, constants):
        raise NotImplementedError("TODO: write your get_list_of_constants")

    def gen_reaction_equations(self, theta, conditions, dev_1hot, condition_on_device=True):
        raise NotImplementedError("TODO: write your gen_reaction_equations")

    def get_precision_list(self, theta):
        return [theta.prec_x, theta.prec_rfp, theta.prec_yfp, theta.prec_cfp]

    def device_conditioner(self, param, param_name, dev_1hot, use_bias=False, activation=None):
        """
        Returns a 1D parameter conditioned on device
        ::NOTE:: condition_on_device is a closure over n_iwae, n_batch, dev_1hot_rep
        """
        # TODO: try e.g. activation=tf.nn.relu
        n_iwae = tf.shape(param)[1]
        n_batch = tf.shape(param)[0]
        # tile devices, one per iwae sample
        dev_1hot_rep = tf.tile(dev_1hot * self.relevance[param_name], [n_iwae, 1])
        param_flat = tf.reshape(param, [n_iwae * n_batch, 1])
        param_cond = tf.layers.dense(dev_1hot_rep, units=1, use_bias=use_bias,
                                     activation=activation, name='%s_decoder' % param_name)
        return tf.reshape(param_flat * tf.exp(param_cond), [n_batch, n_iwae])

    def initialize_state(self, theta, constants):
        constants_tensors = tf.expand_dims(tf.constant(self.get_list_of_constants(constants), dtype=tf.float32), 0)
        n_constants = constants_tensors.shape[1].value
        n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        init_state = tf.reshape(tf.tile(constants_tensors, [n_batch * n_iwae, 1]), (n_batch, n_iwae, n_constants))
        return init_state

    def simulate(self, theta, constants, times, conditions, dev_1hot, solver='dopri15', condition_on_device=True):
        init_state = self.initialize_state(theta, constants)
        d_states_d_t, dev_conditioned = self.gen_reaction_equations(theta, conditions, dev_1hot, condition_on_device)
        if solver == 'modeuler':
            # Evaluate ODEs using Modified-Euler
            t_state, f_state = modified_euler_integrate(d_states_d_t, init_state, times)
            t_state_tr = tf.transpose(t_state, [0, 1, 3, 2])
            f_state_tr = tf.transpose(f_state, [0, 1, 3, 2])
        elif solver == 'modeulerwhile':
            # Evaluate ODEs using Modified-Euler
            t_state, f_state = modified_euler_integrate_while(d_states_d_t, init_state, times)
            t_state_tr = tf.transpose(t_state, [1, 2, 0, 3])
            f_state_tr = None
        else:
            raise NotImplementedError("Solver <%s> is not implemented" % solver)
        return t_state_tr, f_state_tr, dev_conditioned

    @classmethod
    def observe(cls, x_sample, _theta, _constants):
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
        log_prob = tf.reduce_sum(log_prob, [2, 3])
        return log_prob