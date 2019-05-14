# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import absolute_import
from typing import Any, Dict, List, Optional
from collections import OrderedDict

# Standard data science imports
import numpy as np
import pandas as pd
import tensorflow as tf

# Local imports
from data import procdata
from inference.distributions import ChainedDistribution
from inference.encoders import ConditionalConvolutionalEncoder
from inference.decoders import ODEDecoder
from inference.distributions import (build_q_local,
                                         build_q_global,
                                         build_q_global_cond,
                                         build_p_local,
                                         build_p_global,
                                         build_p_global_cond)
from inference.vi import GeneralLogImportanceWeights
from utils import default_get_value

class Dataset(object):

    def __init__(self, D: OrderedDict, original_data_ids: List[int], conditions: Optional[List[str]] = None,
                 use_default_device: bool = False, default_device: str = 'Pcat_Y81C76'):
        '''
        :param D: OrderedDict with keys 'X', 'C', 'Observations' and 'Time'
        :param original_data_ids: TODO(dacart): document these
        :param conditions:
        :param use_default_device:
        :param default_device:
        '''
        self.procdata = procdata.ProcData()
        # Data dictionary
        self.D = D
        # List of conditions (strings)
        self.conditions = conditions or ['C12', 'C6']
        # Number of conditions
        self.n_conditions = len(self.conditions)
        # Whether to use the default device, and what it is
        self.use_default_device = use_default_device
        self.default_device = default_device
        # Number of parameters in device? TODO(dacart) verify
        self.device_depth = self.procdata.n_s_components + self.procdata.n_r_components
        # Keep track of which ids from the original data this dataset comes from
        self.original_data_ids = original_data_ids
        # Numpy array of float, shape e.g. (234,86,4)
        self.X = D['X']
        c_panda = pd.DataFrame(D['C']['values'], columns=D['C']['columns'])
        # Numpy array of float, shape e.g. (234,86)
        self.C = D['C']['values'].astype(np.float32)
        # Numpy array of float, shape e.g. (234,2)
        self.treatments = c_panda[self.conditions].values.astype(np.float32)
        # print("Log + 1 transform conditions")
        self.treatments = np.log(1.0 + self.treatments)
        self.n_treatments = len(self.treatments)
        # Numpy array of float, shape e.g. (234,)
        # TODO(dacart) why is this converted from float64 to float32 then to int?
        self.devices = c_panda['Device'].values.astype(np.float32).astype(int)
        # more numpy data
        # Numpy array of float, shape e.g. (234,86,3)
        self.delta_X = np.diff(self.X, axis=-1)
        # Set up set.dev_1hot, numpy array of float, shape e.g. (234,11)
        self.dev_1hot = None
        self.prepare_devices(self.use_default_device)

    #     ONE-HOT ENCODING OF DEVICE ID        #
    def prepare_devices(self, use_default_device):
        self.dev_1hot = np.zeros((self.size(), self.device_depth))
        self.dev_1hot = self.procdata.get_cassettes(self.devices)
        if use_default_device:
            print("Adding Default device: %s")
            self.dev_1hot = self.procdata.add_default_device(self.dev_1hot, self.default_device)

    def size(self):
        return len(self.X)

    def create_feed_dict(self, placeholders, u_value):
        return {placeholders.x_obs: self.X,
                placeholders.dev_1hot: self.dev_1hot,
                placeholders.conds_obs: self.treatments,
                placeholders.beta: 1.0,
                placeholders.u: u_value}

    def create_feed_dict_for_index(self, placeholders, index, beta_val, u_value):
        return {placeholders.x_obs: self.X[index],
                placeholders.dev_1hot: self.dev_1hot[index],
                placeholders.conds_obs: self.treatments[index],
                placeholders.beta: beta_val,
                placeholders.u: u_value}

class Decoder:
    '''
    Decoder network
    '''

    def __init__(self, verbose: bool, params: Dict[str, Any], placeholders: 'Placeholders', times: np.array,
                 encoder: 'Encoder'):
        ode_decoder = ODEDecoder(params)
        # List(str), e.g. ['OD', 'RFP', 'YFP', 'CFP', 'F510', 'F430', 'LuxR', 'LasR']
        self.names = ode_decoder.ode_model.species # list(str)
        # self.x_sample: Tensor of float32, shape e.g. (?, ?, ?, 8)
        # self.x_post_sample: Tensor of float32, shape e.g. (?, ?, ?, 4)
        # self.device_conditioned: Dict[str,Tensor], keys e.g. 'aS', 'aR'.
        self.x_sample, self.x_post_sample, self.device_conditioned = ode_decoder(
            placeholders.conds_obs, placeholders.dev_1hot, times, encoder.theta, encoder.clipped_theta,
            condition_on_device=True)
        self.set_up_summaries(verbose)

    def set_up_summaries(self, verbose: bool):
        for para, var in self.device_conditioned.items():
            with tf.name_scope('%s.conditioned' % para):
                if verbose:
                    print("- Creating tensorboard outputs for %s" % para)
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

class Encoder:
    '''
    Encoder network
    '''

    def __init__(self, verbose: bool, parameters: 'Parameters', placeholders: 'Placeholders',
                 x_delta_obs: tf.Tensor):
        # ChainedDistribution
        self.q = self.set_up_q(verbose, parameters, placeholders, x_delta_obs)
        # DotOperatorSamples
        self.theta = self.q.sample(placeholders.u, verbose)  # return a dot operating theta
        # List of (about 30) strings
        self.theta_names = self.theta.keys
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
            build_p_local(parameters, self.theta, verbose),
            build_p_global_cond(parameters, self.theta, verbose),
            # prior: global should be fully defined in parameters
            build_p_global(parameters, self.theta, verbose))
        if verbose:
            p_vals.diagnostic_printout('P')
        return p_vals.concat("p")

    @classmethod
    def set_up_q(cls, verbose, parameters, placeholders, x_delta_obs):
        encode = ConditionalConvolutionalEncoder(parameters.params_dict)
        approx_posterior_params = encode(placeholders.x_obs, x_delta_obs, placeholders.dev_1hot, placeholders.conds_obs)
        q_vals = LocalAndGlobal(
            # q: local, based on amortized neural network
            build_q_local(parameters, approx_posterior_params, placeholders.dev_1hot, placeholders.conds_obs, verbose,
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            # q: global, device-dependent distributions
            build_q_global_cond(parameters, placeholders.dev_1hot, placeholders.conds_obs, verbose),
            # q: global, independent distributions
            build_q_global(parameters, verbose))
        if verbose:
            q_vals.diagnostic_printout('Q')
        return q_vals.concat("q")


class SessionVariables:
    """Convenience class to hold the output of one of the Session.run calls used in training."""
    def __init__(self, seq):
        """seq: a sequence of 8 or 9 elements."""
        assert len(seq) == 8 or len(seq) == 9
        (self.log_normalized_iws, self.normalized_iws, self.normalized_iws_reshape,
         self.x_post_sample, self.x_sample, self.elbo, self.precisions, self.theta_tensors) = seq[:8]
        self.summaries = seq[8] if len(seq) == 9 else None

    def as_list(self):
        result = [self.log_normalized_iws, self.normalized_iws, self.normalized_iws_reshape,
                  self.x_post_sample, self.x_sample, self.elbo, self.precisions, self.theta_tensors]
        if self.summaries is not None:
            result.append(self.summaries)
        return result


class LocalAndGlobal:
    """Convenience class to hold any triple of local, global-conditional and global values."""

    def __init__(self, loc, glob_cond, glob):
        self.loc = loc
        self.glob_cond = glob_cond
        self.glob = glob

    @classmethod
    def from_list(cls, seq):
        return LocalAndGlobal(seq[0], seq[1], seq[2])

    def to_list(self):
        return [self.loc, self.glob_cond, self.glob]

    def sum(self):
        return self.loc + self.glob_cond + self.glob

    def create_placeholders(self, suffix):
        def as_placeholder(size, name):
            return tf.placeholder(dtype=tf.float32, shape=(None, None, size), name=name)
        return LocalAndGlobal(
            as_placeholder(self.loc, "local_" + suffix),
            as_placeholder(self.glob_cond, "global_cond_" + suffix),
            as_placeholder(self.glob, "global_" + suffix))

    def concat(self, name,):
        """Returns a concatenation of the items. Copied from distributions.concat_local_global.
        TODO(dacart): replace distributions.concat_local_global with this method throughout."""
        concated = ChainedDistribution(name=name)
        for chained in self.to_list():
            for item_name, distribution in chained.distributions.items():
                concated.add_distribution(item_name, distribution, chained.slot_dependencies[item_name])
        return concated

    def diagnostic_printout(self, prefix):
        print('%s-LOCAL\n%s' % (prefix, self.loc))
        print('%s-GLOBAL-COND\n%s' % (prefix, self.glob_cond))
        print('%s-GLOBAL\n%s' % (prefix, self.glob))

class Objective:
    '''A convenience class to hold variables related to the objective function of the network.'''
    def __init__(self, encoder, decoder, model, placeholders):
        self.log_p_observations = model.log_prob_observations(decoder.x_post_sample, placeholders.x_obs, encoder.theta,
                                                              decoder.x_sample)
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
    '''A convenience class of tf.placeholder tensors, associated with actual values on each batch of training.'''
    def __init__(self, data_pair, n_vals):
        # PLACEHOLDERS: represent stuff we must supply to the computational graph at each iteration,
        # e.g. batch of data or random numbers
        #: None means we can dynamically set this number (nbr of batch, nbr of IW samples)
        self.x_obs = tf.placeholder(dtype=tf.float32, shape=(None, (data_pair.n_time), (data_pair.n_species)),
                                    name='species')
        self.dev_1hot = tf.placeholder(dtype=tf.float32, shape=(None, (data_pair.depth)), name='device_1hot')
        self.conds_obs = tf.placeholder(dtype=tf.float32, shape=(None, (data_pair.n_conditions)), name='conditions')
        # for beta VAE
        self.beta = tf.placeholder(dtype=tf.float32, shape=(None), name='beta')
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

class TrainingStepper:
    '''Class to hold variables needed for the training loop.'''
    def __init__(self, dreg: bool, encoder: Encoder, objective: Objective, params_dict: Dict[str, Any]):
        # using gradient descent with a learning rate schedule (these are HYPER parameters!)
        global_step = tf.Variable(0, trainable=False)
        boundaries = default_get_value(params_dict, "learning_boundaries", [1000, 2000, 5000])
        values = [float(f) for f in default_get_value(params_dict, "learning_rates", [1e-2, 1e-3, 1e-4, 2 * 1e-5])]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        # Alternatives for opt_func, with momentum e.g. 0.50, 0.75
        #   momentum = default_get_value(params_dict, "momentum", 0.0)
        #   opt_func = tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
        #   opt_func = tf.train.RMSPropOptimizer(learning_rate)
        # Initially the gradients can be too big, clip them.
        # All three of the following are tf.Tensor, shape (?,1):
        self.logsumexp_log_p = tf.reduce_logsumexp(objective.log_p_observations, axis=1, keepdims=True)
        self.logsumexp_log_p_theta = tf.reduce_logsumexp(encoder.log_p_theta, axis=1, keepdims=True)
        self.logsumexp_log_q_theta = tf.reduce_logsumexp(encoder.log_q_theta, axis=1, keepdims=True)
        opt_func = tf.train.AdamOptimizer(learning_rate)
        self.train_step = self.build_train_step(dreg, encoder, objective, opt_func)

    @classmethod
    def build_train_step(cls, dreg, encoder, objective, opt_func):
        '''Returns a computation that is run in the tensorflow session.'''
        # This path is for b_use_correct_iwae_gradients = True. For False, we would just
        # want to return opt_func.minimize(objective.vae_cost)
        trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if dreg:
            grads = cls.create_dreg_gradients(encoder, objective, trainable_params)
        else:
            # ... so, get list of params we will change with grad and ...
            # ... compute the VAE elbo gradient, using special stop grad function to prevent propagating gradients
            # through ws (ie just a copy)
            grads = tf.gradients(objective.vae_cost, trainable_params)
            print("Set up non-dreg gradient")
            # grads = [tf.clip_by_value(g, -0.1, 0.1) for g in iwae_grads]
        # This depends on update rule implemented in AdamOptimizer:
        optimizer = opt_func.apply_gradients(zip(grads, trainable_params))
        # TODO(dacart): check if this should go above "optimizer =" or be deleted.
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        return optimizer

    @classmethod
    def create_dreg_gradients(cls, encoder, objective, trainable_params):
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
        print("Set up Doubly Reparameterized Gradient (dreg)")
        return grads


class DatasetPair(object):
    '''A holder for a training and validation set and various associated parameters.'''
    # pylint: disable=too-many-instance-attributes,too-few-public-methods

    def __init__(self, train: Dataset, val: Dataset): # use_default_device=False):
        '''
        :param train: a Dataset containing the training data
        :param val: a Dataset containing the validation data
        '''
        # Dataset of the training data
        self.train = train
        # Dataset of the validation data
        self.val = val
        # List of scaling statistics (floats, built by scale_data)
        self.scales = []
        self.scale_data()
        # Number of training instances (int)
        self.n_train = self.train.size()
        # Number of validation instances (int)
        self.n_val = self.val.size()
        # Number of species we're training on (int)
        self.n_species = self.train.X.shape[2]
        # Number of time points we're training on
        self.n_time = self.train.X.shape[1]
        # TODO(dacart) document this
        self.depth = self.train.device_depth
        # Number of conditions we're training on
        self.n_conditions = self.train.n_conditions
        # Numpy array of time-point values (floats), length self.n_time
        self.times = self.train.D['Time'].astype(np.float32)

    def scale_data(self):
        od, rfp, yfp, cfp = 0, 1, 2, 3  # constants for defining data
        outputs = [od, rfp, yfp, cfp]
        def compute_scaling_statistic(array):
            return np.max(array)
        for output_idx in outputs:
            # First scale the data according to the train
            output_statistic = compute_scaling_statistic(self.train.X[:, :, output_idx])
            self.scales.append(output_statistic)
            self.train.X[:, :, output_idx] /= self.scales[output_idx].astype(np.float32)
            self.val.X[:, :, output_idx] /= self.scales[output_idx].astype(np.float32)
            # mins = np.min(np.min(self.train.X[:, :, output_idx], axis=1))
            # Second, shift so smallest value for each time series is 0
            self.train.X[:, :, output_idx] -= np.min(self.train.X[:, :, output_idx], axis=1)[:, np.newaxis]
            self.val.X[:, :, output_idx] -= np.min(self.val.X[:, :, output_idx], axis=1)[:, np.newaxis]