# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

from __future__ import absolute_import
from typing import Any, Dict, List, Optional
from collections import OrderedDict

# Standard data science imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.compat.v1 import placeholder

# Local imports
import procdata
import encoders
from decoders import ODEDecoder
import distributions as ds
from vi import GeneralLogImportanceWeights
from utils import default_get_value, variable_summaries

class Decoder:
    '''
    Decoder network
    '''

    def __init__(self, params: Dict[str, Any], placeholders: 'Placeholders', times: np.array,
                 encoder: 'Encoder', condition_on_device=True, plot_histograms=True):
        ode_decoder = ODEDecoder(params)
        # List(str), e.g. ['OD', 'RFP', 'YFP', 'CFP', 'F510', 'F430', 'LuxR', 'LasR']
        self.names = ode_decoder.ode_model.species # list(str)
        # self.x_sample: Tensor of float32, shape e.g. (?, ?, ?, 8)
        # self.x_post_sample: Tensor of float32, shape e.g. (?, ?, ?, 4)
        # self.device_conditioned: Dict[str,Tensor], keys e.g. 'aS', 'aR'.
        self.x_sample, self.x_post_sample = ode_decoder(
            placeholders.conds_obs, placeholders.dev_1hot, times, encoder.theta, encoder.clipped_theta,
            condition_on_device)

class Encoder:
    '''
    Encoder network
    '''

    def __init__(self, verbose: bool, parameters: 'Parameters', placeholders: 'Placeholders', x_delta_obs: tf.Tensor):
        # ChainedDistribution
        self.q = self.set_up_q(verbose, parameters, placeholders, x_delta_obs)
        # DotOperatorSamples
        self.theta = self.q.sample(placeholders.u, verbose)  # return a dot operating theta
        # List of (about 30) strings
        #self.theta_names = self.theta.keys
        self.theta_names = self.q.get_theta_names()
        if verbose:
            print('THETA ~ Q')
            print(self.theta)
        # tf.Tensor of float32. theta is in [batch, iwae_samples, theta_dim]
        self.log_q_theta = self.q.log_prob(self.theta)  # initial log density # [batch, iwae_samples]
        # log_q_global_cond_theta = q_vals.glob_cond.log_prob(theta)
        # log_q_global_theta      = q_vals.glob.log_prob(theta)
        # ChainedDistribution
        self.p = self.set_up_p(verbose, parameters)
        # DotOperatorSamples
        self.clipped_theta = self.p.clip(self.theta, stddevs=4)
        # Tensor of float
        self.log_p_theta = self.p.log_prob(self.theta)

    def set_up_p(self, verbose: bool, parameters: 'Parameters'):
        """Returns a ChainedDistribution"""
        p_vals = LocalAndGlobal(
            # prior: local: may have some dependencies in theta (in hierarchy, local, etc)
            ds.build_p_local(parameters, verbose, self.theta),
            ds.build_p_global_cond(parameters, verbose, self.theta),
            # prior: global should be fully defined in parameters
            ds.build_p_global(parameters, verbose, self.theta),
            ds.build_p_constant(parameters, verbose, self.theta))
        if verbose:
            p_vals.diagnostic_printout('P')
        return p_vals.concat("p")

    @classmethod
    def set_up_q(self, verbose, parameters, placeholders, x_delta_obs):
        # Constants
        q_constant = ds.build_q_constant(parameters, verbose)
        # q: global, device-dependent distributions
        q_global_cond = ds.build_q_global_cond(parameters, placeholders.dev_1hot, placeholders.conds_obs, verbose, plot_histograms=parameters.params_dict["plot_histograms"])
        # q: global, independent distributions
        q_global = ds.build_q_global(parameters, verbose)
        # q: local, based on amortized neural network
        if len(parameters.l.list_of_params) > 0:
            encode = encoders.ConditionalEncoder(parameters.params_dict)
            approx_posterior_params = encode(x_delta_obs)
            q_local = ds.build_q_local(parameters, approx_posterior_params, placeholders.dev_1hot, placeholders.conds_obs, verbose,
                        kernel_regularizer=tf.keras.regularizers.l2(0.01))
        else:
            q_local = ds.ChainedDistribution(name="q_local")
        q_vals = LocalAndGlobal(q_local, q_global_cond, q_global, q_constant)
        if verbose:
            q_vals.diagnostic_printout('Q')
        return q_vals.concat("q")


class SessionVariables:
    """Convenience class to hold the output of one of the Session.run calls used in training."""
    def __init__(self, seq):
        """seq: a sequence of 10 or 11 elements."""
        n = 10
        assert len(seq) == n or len(seq) == (n+1)
        (self.log_normalized_iws, self.normalized_iws, self.normalized_iws_reshape,
         self.x_post_sample, self.x_sample, self.elbo, self.vae_cost, self.precisions, self.theta_tensors, self.q_params) = seq[:n]
        self.summaries = seq[n] if len(seq) == (n+1) else None

    def as_list(self):
        result = [self.log_normalized_iws, self.normalized_iws, self.normalized_iws_reshape,
                  self.x_post_sample, self.x_sample, self.elbo, self.vae_cost, self.precisions, self.theta_tensors, self.q_params]
        if self.summaries is not None:
            result.append(self.summaries)
        return result


class LocalAndGlobal:
    """Convenience class to hold any tuple of local, global-conditional and global values."""

    def __init__(self, loc, glob_cond, glob, const):
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

    def create_placeholders(self, suffix):
        def as_placeholder(size, name):
            return placeholder(dtype=tf.float32, shape=(None, None, size), name=name)
        return LocalAndGlobal(
            as_placeholder(self.loc, "local_" + suffix),
            as_placeholder(self.glob_cond, "global_cond_" + suffix),
            as_placeholder(self.glob, "global_" + suffix),
            as_placeholder(self.const, "const_" + suffix))

    def concat(self, name,):
        """Returns a concatenation of the items."""
        concatenated = ds.ChainedDistribution(name=name)
        for chained in self.to_list():
            for item_name, distribution in chained.distributions.items():
                concatenated.add_distribution(item_name, distribution, chained.slot_dependencies[item_name])
        return concatenated

    def diagnostic_printout(self, prefix):
        print('%s-LOCAL\n%s' % (prefix, self.loc))
        print('%s-GLOBAL-COND\n%s' % (prefix, self.glob_cond))
        print('%s-GLOBAL\n%s' % (prefix, self.glob))
        print('%s-CONSTANT\n%s' % (prefix, self.const))

class Objective:
    '''A convenience class to hold variables related to the objective function of the network.'''
    def __init__(self, encoder, decoder, model, placeholders):
        self.log_p_observations_by_species = model.log_prob_observations(decoder.x_post_sample, placeholders.x_obs, encoder.theta, decoder.x_sample)
        self.log_p_observations = tf.reduce_sum(self.log_p_observations_by_species, 2)
        # let the model decide what precisions to use.
        # pylint:disable=fixme
        # TODO: will work for constant time precisions, but not for decayed. (get precisions after log_prob called)
        _log_precisions, self.precisions = model.expand_precisions_by_time(encoder.theta, decoder.x_post_sample,
                                                                           placeholders.x_obs, decoder.x_sample)
        self.log_unnormalized_iws = GeneralLogImportanceWeights(
            self.log_p_observations, encoder.log_p_theta, encoder.log_q_theta, beta=1.0)
        # [batch_size, num_iwae_samples]
        logsumexp_log_unnormalized_iws = tf.reduce_logsumexp(self.log_unnormalized_iws, axis=1, keepdims=True)
        # w_logmeanexp = w_logsumexp - tf.log(tf.cast(args.train_samples, tf.float32))
        self.vae_cost = -tf.reduce_mean(self.log_unnormalized_iws)
        # corresponds to `model_loss`
        iwae_cost = -tf.reduce_mean(logsumexp_log_unnormalized_iws -
                                    tf.log(tf.cast(tf.shape(self.log_p_observations)[1],
                                                   tf.float32)))  # mean over batch
        self.elbo = -iwae_cost
        # log_ws:= log_unnormalized_important_weights
        # ys:= log_normalized_importance_weights
        # ws:= normalized_importance_weights
        # w_logsumexp:= logsumexp_log_unnormalized_important_weights
        self.log_normalized_iws = self.log_unnormalized_iws - logsumexp_log_unnormalized_iws
        self.normalized_iws = tf.exp(self.log_normalized_iws)
        self.normalized_iws_reshape = tf.reshape(self.normalized_iws, shape=[-1])
        #log_unnormalized_iws_reshape = tf.reshape(log_unnormalized_important_weights, shape=[-1])

class Placeholders:
    '''A convenience class of placeholder tensors, associated with actual values on each batch of training.'''
    def __init__(self, data_pair, n_vals):
        # PLACEHOLDERS: represent stuff we must supply to the computational graph at each iteration,
        # e.g. batch of data or random numbers
        #: None means we can dynamically set this number (nbr of batch, nbr of IW samples)
        self.x_obs = placeholder(dtype=tf.float32, shape=(None, data_pair.n_time, data_pair.n_species), name='species')
        self.dev_1hot = placeholder(dtype=tf.float32, shape=(None, data_pair.depth), name='device_1hot')
        self.conds_obs = placeholder(dtype=tf.float32, shape=(None, data_pair.n_conditions), name='conditions')
        # for beta VAE
        self.beta = placeholder(dtype=tf.float32, shape=(None), name='beta')
        u_vals = n_vals.create_placeholders("random_bits")
        self.u = tf.concat(u_vals.to_list(), axis=-1, name='u_local_global_stacked')

class TrainingLogData:
    '''A convenience class of data collected for logging during training'''
    def __init__(self):
        self.training_elbo_list = []
        self.validation_elbo_list = []
        self.batch_feed_time = 0.0
        self.batch_train_time = 0.0
        self.total_train_time = 0.0
        self.total_test_time = 0.0
        self.n_test = 0
        self.max_val_elbo = -float('inf')

class TrainingStepper:
    '''Class to hold variables needed for the training loop.'''
    def __init__(self, dreg: bool, encoder: Encoder, objective: Objective, params_dict: Dict[str, Any]):
        # using gradient descent with a learning rate schedule (these are HYPER parameters!)
        global_step = tf.Variable(0, trainable=False)
        self.plot_histograms = params_dict["plot_histograms"]
        self.tb_gradients = params_dict["tb_gradients"]
        boundaries = default_get_value(params_dict, "learning_boundaries", [1000, 2000, 5000])
        values = [float(f) for f in default_get_value(params_dict, "learning_rates", [1e-2, 1e-3, 1e-4, 2 * 1e-5])]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        # Alternatives for opt_func, with momentum e.g. 0.50, 0.75
        #   momentum = default_get_value(params_dict, "momentum", 0.0)
        #   opt_func = tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
        #   opt_func = tf.train.RMSPropOptimizer(learning_rate)
        opt_func = tf.train.AdamOptimizer(learning_rate)
        self.train_step = self.build_train_step(dreg, encoder, objective, opt_func)

    @classmethod
    def create_dreg_gradients(self, encoder, objective, trainable_params):
        normalized_weights = tf.stop_gradient(
            tf.nn.softmax(objective.log_unnormalized_iws, axis=1))  # [batch_size, num_iwae]
        sq_normalized_weights = tf.square(normalized_weights)  # [batch_size, num_iwae]
        stopped_log_q_theta = encoder.q.log_prob(encoder.theta, stop_grad=True)
        stopped_log_weights = GeneralLogImportanceWeights(objective.log_p_observations, encoder.log_p_theta,
                                                          stopped_log_q_theta,
                                                          beta=1.0)
        neg_iwae_grad = tf.reduce_sum(sq_normalized_weights * stopped_log_weights, axis=1)  # [batch_size]
        iwae_grad = -tf.reduce_mean(neg_iwae_grad)
        grads = tf.gradients(iwae_grad, trainable_params)
        return grads

    #@classmethod
    def build_train_step(self, dreg, encoder, objective, opt_func):
        '''Returns a computation that is run in the tensorflow session.'''
        # This path is for b_use_correct_iwae_gradients = True. For False, we would just
        # want to return opt_func.minimize(objective.vae_cost)
        trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if dreg:
            grads = self.create_dreg_gradients(encoder, objective, trainable_params)
            print("Set up Doubly Reparameterized Gradient (dreg)")
        else:
            # ... so, get list of params we will change with grad and ...
            # ... compute the VAE elbo gradient, using special stop grad function to prevent propagating gradients
            # through ws (ie just a copy)
            grads = tf.gradients(objective.vae_cost, trainable_params)
            print("Set up non-dreg gradient")
            # grads = [tf.clip_by_value(g, -0.1, 0.1) for g in iwae_grads]
        if self.tb_gradients:
            with tf.name_scope('Gradients'):
                for p,g in zip(trainable_params, grads):
                    variable_summaries(g, p.name.split(':')[0], self.plot_histograms)
        # TODO(dacart): check if this should go above "optimizer =" or be deleted.
        #clipped_grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        # This depends on update rule implemented in AdamOptimizer:
        optimizer = opt_func.apply_gradients(zip(grads, trainable_params))
        return optimizer

