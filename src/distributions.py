# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

from abc import ABC, abstractmethod
from collections import OrderedDict
import os
import pdb
import numpy as np
import tensorflow as tf

from utils import variable_summaries

SQRT2 = np.sqrt(2.0)
LOG2PI = np.log(2 * np.pi)
IDENTITY = "identity"
POSITIVE = "positive"
A_FOR_ERF = 8.0 / (3.0 * np.pi) * (np.pi - 3.0) / (4.0 - np.pi)

def erf_approx(x):
    return tf.tanh(x)

def erfinv_approx(x):
    return tf.atanh(x)

def constrain_parameter(tf_free_param, free_to_constrained, distribution_name, constrained_name):
    if free_to_constrained == IDENTITY:
        tf_constrained_param = tf.identity(tf_free_param, name='%s_%s_identity' % (distribution_name, constrained_name))
    elif free_to_constrained == POSITIVE:
        tf_constrained_param = tf.exp(tf_free_param, name='%s_%s' % (distribution_name, constrained_name))
    else:
        raise NotImplementedError("unknown free_to_constrained = %s" % (free_to_constrained))

    return tf_constrained_param

def build_q_local(PARAMETERS, hidden, devs, conds, verbose, kernel_regularizer, use_bias=True, stop_grad=False, plot_histograms=False):
    assert hasattr(PARAMETERS, "l"), "require local parameters"
    distribution_descriptions = PARAMETERS.l

    # make a distribution that has "log_prob(theta)" and "sample()"
    q_local = ChainedDistribution(name="q_local")

    for distribution_name in distribution_descriptions.list_of_params:
        if verbose:
            print("- build_q_local::%s" % distribution_name)
        description = getattr(distribution_descriptions, distribution_name)
        conditioning = description.defaults['c']  # <-- not a tensor
        params = OrderedDict()
        for free_name, constrained_name, free_to_constrained in zip(
                description.free_params, description.params, description.free_to_constrained):
            # when appropriate, concatenate the dependencies
            to_concat = [hidden]
            if conditioning is not None:  # collect tensors to concat
                if conditioning['treatments']:
                    to_concat.append(conds)
                if conditioning['devices']:
                    to_concat.append(devs)
            hidden_conditioned = tf.concat(to_concat, axis=1)

            # filter to whatever we want to condition
            name = '%s_%s'%(distribution_name,free_name)
            free_param = tf.keras.layers.Dense(1, use_bias=use_bias, kernel_regularizer=kernel_regularizer, name=name)
            tf_free_param = free_param(hidden_conditioned)
            if stop_grad:
                tf_free_param = tf.stop_gradient(tf_free_param)  # eliminate score function term from autodiff
            tf_constrained_param = constrain_parameter(
                tf_free_param, free_to_constrained, distribution_name, constrained_name)

            params[free_name] = tf_free_param
            params[constrained_name] = tf_constrained_param

        for other_param_name, other_param_value in description.other_params.items():
            params[other_param_name] = other_param_value

        new_distribution = description.class_type(wait_for_assigned=True, variable=True)
        new_distribution.assign_free_and_constrained(**params)

        q_local.add_distribution(distribution_name, new_distribution)

    return q_local

def build_q_global_cond(PARAMETERS, devs, conds, verbose, kernel_regularizer=None, use_bias=False, stop_grad=False, plot_histograms=False):
    # make a distribution that has "log_prob(theta)" and "sample()"
    q_global_cond = ChainedDistribution(name="q_global_cond")

    if not hasattr(PARAMETERS, "g_c"):
        print("- Found no global conditional params")
        return q_global_cond

    distribution_descriptions = PARAMETERS.g_c

    for distribution_name in distribution_descriptions.list_of_params:
        description = getattr(distribution_descriptions, distribution_name)

        conditioning = description.defaults['c']  # <-- not a tensor

        if verbose:
            print("- build_q_global_cond::%s"%distribution_name)
        params = OrderedDict()
        for free_name, constrained_name, free_to_constrained in zip(
                description.free_params, description.params, description.free_to_constrained):
            to_concat = []
            if conditioning is not None:  # collect tensors to concat
                if verbose:
                    print("- Conditioning parameter %s.%s" % (distribution_name, free_name))
                if conditioning['treatments']:
                    to_concat.append(conds)
                if conditioning['devices']:
                    to_concat.append(devs)

            mlp_inp = tf.concat(to_concat, axis=1)
            name = '%s_%s' % (distribution_name, free_name)
            # map sample from prior with conditioning information through 1-layer NN
            free_param = tf.keras.layers.Dense(1, use_bias=use_bias, kernel_regularizer=kernel_regularizer, name=name)
            tf_free_param = free_param(mlp_inp)
            #tf_free_param = tf.layers.dense(mlp_inp, units=1, use_bias=use_bias, kernel_regularizer=kernel_regularizer, 
            if stop_grad:
                tf_free_param = tf.stop_gradient(tf_free_param)
            variable_summaries(tf_free_param, 'nn_%s'%name, plot_histograms)
            tf_constrained_param = constrain_parameter(tf_free_param, free_to_constrained, distribution_name, constrained_name)

            params[free_name] = tf_free_param
            params[constrained_name] = tf_constrained_param

        for other_param_name, other_param_value in description.other_params.items():
            params[other_param_name] = other_param_value

        new_distribution = description.class_type(wait_for_assigned=True, variable=True)
        new_distribution.assign_free_and_constrained(**params)

        q_global_cond.add_distribution(distribution_name, new_distribution)

    return q_global_cond

def build_q_global(PARAMETERS, verbose, stop_grad=False):
    # make a distribution that has "log_prob(theta)" and "sample()"
    q_global = ChainedDistribution(name="q_global")

    if not hasattr(PARAMETERS, "g"):
        print("- Found no global parameters")
        return q_global

    distribution_descriptions = PARAMETERS.g

    for distribution_name in distribution_descriptions.list_of_params:
        description = getattr(distribution_descriptions, distribution_name)
        if verbose:
            print("- build_q_global::%s" % distribution_name)
        params = OrderedDict()
        for free_name, constrained_name, free_to_constrained, init_free in zip(
                description.free_params, description.params, description.free_to_constrained,
                description.init_free_params):

            tf_free_param = tf.Variable(init_free, name='%s.%s'%(distribution_name,free_name))
            if stop_grad:
                tf_free_param = tf.stop_gradient(tf_free_param)
            tf_constrained_param = constrain_parameter(
                tf_free_param, free_to_constrained, distribution_name, constrained_name)

            params[free_name] = tf_free_param
            params[constrained_name] = tf_constrained_param

        for other_param_name, other_param_value in description.other_params.items():
            params[other_param_name] = other_param_value

        new_distribution = description.class_type(wait_for_assigned=True, variable=False)
        new_distribution.assign_free_and_constrained(**params)

        q_global.add_distribution(distribution_name, new_distribution)

    return q_global

def build_q_constant(PARAMETERS, verbose, stop_grad=False):
    # make a distribution that has "log_prob(theta)" and "sample()"
    q_constant = ChainedDistribution(name="q_constant")

    if not hasattr(PARAMETERS, "c"):
        print("- Found no constant parameters")
        return q_constant

    distribution_descriptions = PARAMETERS.c

    for distribution_name in distribution_descriptions.list_of_params:
        description = getattr(distribution_descriptions, distribution_name)
        if verbose:
            print("- build_q_constant::%s" % distribution_name)
        params = OrderedDict()
        for free_name, constrained_name, free_to_constrained, init_free in zip(
                description.free_params, description.params, description.free_to_constrained,
                description.init_free_params):

            tf_free_param = tf.Variable(init_free, name='%s.%s'%(distribution_name,free_name), trainable=False)
            params[free_name] = tf_free_param

        for other_param_name, other_param_value in description.other_params.items():
            params[other_param_name] = other_param_value

        new_distribution = description.class_type(wait_for_assigned=True, variable=False)
        new_distribution.assign_free_and_constrained(**params)

        q_constant.add_distribution(distribution_name, new_distribution)

    return q_constant

def build_p_global(PARAMETERS, verbose, theta=None):
    # p_global: generative model with fixed distribution parameters; ie the top-level distributions

    # make a distribution that has "log_prob(theta)" and "sample()"
    p_global = ChainedDistribution(name="p_global")

    if not hasattr(PARAMETERS, "g"):
        print("- Found no global parameters")
        return p_global

    assert hasattr(PARAMETERS, "g"), "require global parameters"
    distribution_descriptions = PARAMETERS.g

    for distribution_name in distribution_descriptions.list_of_params:
        if verbose:
            print("- build_p_global::%s"%distribution_name)

        description = getattr(distribution_descriptions, distribution_name)

        params = OrderedDict()
        slots = OrderedDict()
        # check for dependencies
        for (constrained_name, dependency) in zip(description.params, description.dependencies):
            if dependency is None:
                continue
            # TODO(dacart): arguments for format string missing
            #if verbose:
            #    print("build_p_global: found dependency for %s = %s" % )
            if theta is None:
                if verbose:
                    print("build_p_global: empty slot for dependency!")
                params[constrained_name] = None
                slots[constrained_name] = dependency
            else:
                params[constrained_name] = getattr(theta, dependency)

        # for each default param not already found via dependency, add to params
        for constrained_name, default_value in description.defaults.items():
            if constrained_name not in params:
                params[constrained_name] = default_value

        new_distribution = description.class_type(**params)
        p_global.add_distribution(distribution_name, new_distribution, slots)

    return p_global

def build_p_constant(PARAMETERS, verbose, theta=None):
    # p_global: generative model with fixed distribution parameters; ie the top-level distributions

    # make a distribution that has "log_prob(theta)" and "sample()"
    p_constant = ChainedDistribution(name="p_constant")

    if not hasattr(PARAMETERS, "c"):
        print("- Found no constant parameters")
        return p_constant

    assert hasattr(PARAMETERS, "c"), "require constant parameters"
    distribution_descriptions = PARAMETERS.c

    for distribution_name in distribution_descriptions.list_of_params:
        if verbose:
            print("- build_p_constant::%s"%distribution_name)

        description = getattr(distribution_descriptions, distribution_name)

        params = OrderedDict()
        slots = OrderedDict()
        # check for dependencies
        for (constrained_name, dependency) in zip(description.params, description.dependencies):
            if dependency is None:
                continue
            # TODO(dacart): arguments for format string missing
            #if verbose:
            #    print("build_p_constant: found dependency for %s = %s" % )
            if theta is None:
                if verbose:
                    print("build_p_constant: empty slot for dependency!")
                params[constrained_name] = None
                slots[constrained_name] = dependency
            else:
                params[constrained_name] = getattr(theta, dependency)

        # for each default param not already found via dependency, add to params
        for constrained_name, default_value in description.defaults.items():
            if constrained_name not in params:
                params[constrained_name] = default_value

        new_distribution = description.class_type(**params)
        p_constant.add_distribution(distribution_name, new_distribution, slots)

    return p_constant

def build_p_global_cond(PARAMETERS, verbose, theta=None):
    p_global_cond = ChainedDistribution(name="p_global_cond")

    if not hasattr(PARAMETERS, "g_c"):
        print("- Found no global conditional params")
        return p_global_cond

    assert hasattr(PARAMETERS, "g_c"), "require global conditioned parameters"
    distribution_descriptions = PARAMETERS.g_c

    for distribution_name in distribution_descriptions.list_of_params:
        if verbose:
            print("- build_p_global_cond::%s"%distribution_name)

        description = getattr(distribution_descriptions, distribution_name)
        #conditioning = description.defaults['c']  # <-- not a tensor

        params = OrderedDict()
        slots = OrderedDict()
        # check for dependencies
        for (constrained_name, dependency) in zip(description.params, description.dependencies):

            if dependency is not None:
                #if verbose:
                #    print("build_p_global_cond: found dependency for %s = %s" % )
                if theta is None:
                    if verbose:
                        print("build_p_global_cond: empty slot for dependency!")
                    params[constrained_name] = None
                    slots[constrained_name] = dependency
                else:
                    params[constrained_name] = getattr(theta, dependency)

        # for each default param not already found via dependency, add to params
        for constrained_name, default_value in description.defaults.items():
            if constrained_name not in params:
                params[constrained_name] = default_value

        new_distribution = description.class_type(**params)
        p_global_cond.add_distribution(distribution_name, new_distribution, slots)

    return p_global_cond

def build_p_local(PARAMETERS, verbose, theta=None):
    assert hasattr(PARAMETERS, "l"), "require local parameters"
    distribution_descriptions = PARAMETERS.l

    # make a distribution that has "log_prob(theta)" and "sample()"
    p_local = ChainedDistribution(name="p_local")

    for distribution_name in distribution_descriptions.list_of_params:
        if verbose:
            print("- build_p_local::%s"%distribution_name)

        description = getattr(distribution_descriptions, distribution_name)

        params = OrderedDict()
        slots = OrderedDict()

        # check for dependencies
        for (constrained_name, dependency) in zip(description.params, description.dependencies):

            if dependency is not None:
                print("- build_p_local: found dependency for %s = %s" % (constrained_name, dependency))
                #params[constrained_name] = getattr(theta, dependency)
                if theta is None:
                    print("- build_p_local: empty slot for dependency!")
                    params[constrained_name] = None
                    slots[constrained_name] = dependency
                else:
                    params[constrained_name] = getattr(theta, dependency)

        # for each default param not already found via dependency, add to params
        for constrained_name, default_value in description.defaults.items():
            if constrained_name not in params:
                params[constrained_name] = default_value

        new_distribution = description.class_type(**params)
        p_local.add_distribution(distribution_name, new_distribution, slots)

    return p_local

class DotOperatorSamples(object):
    def __init__(self):
        self.samples = OrderedDict()
        self.keys = []
        self.values = []

    def add(self, distribution_name, distribution_sample):
        assert not hasattr(self, distribution_name), "DotOperatorSamples already has %s" % distribution_name
        self.samples[distribution_name] = distribution_sample
        self.keys.append(distribution_name)
        self.values.append(distribution_sample)
        setattr(self, distribution_name, distribution_sample)

    def __str__(self):
        s = ""
        for distribution_name, distribution_sample in self.samples.items():
            s += "%s = %s\n" % (distribution_name, distribution_sample)
        return s

    def get_n_batch(self):
        return tf.shape(self.values[0])[0]
        #return self.samples.values()[0].shape[0]

    def get_n_samples(self):
        return tf.shape(self.values[0])[1]
        #return tf.shape(list(self.samples)[0][1])[1]
        #return self.samples.values()[0].shape[1]

    def get_tensors(self):
        return self.values #[tensor for tensor in self.samples.values()]

    # Commented out as not called.
    #def set_tensors(self, tensor_list):
    #    # TODO(dacart) what is going on here?
    #    for _tensor in self.samples.values():
    #        pass
    #    # return [tensor for tensor in self.samples.values()]

class ChainedDistribution(object):
    def __init__(self, name="unknown"):
        self.name = name
        self.distributions = OrderedDict()
        self.slot_dependencies = OrderedDict()

    def log_prob(self, theta, stop_grad=False):  # wraps log_prob for individual distributions
        log_probs = []
        for distribution, sample_value in theta.samples.items():
            if distribution in self.distributions:
                log_probs.append(self.distributions[distribution].log_prob(sample_value, stop_grad))

        # stacking issue?
        if log_probs:
            return tf.reduce_sum(tf.stack(log_probs, -1), -1)
        # return 0.0 so we can broadcast empty logprob with another non-empty
        return 0.0

    def monte_carlo_mode(self, theta, tile_it=True):
        # TODO(dacart): use or lose tile_it (in related methods too)
        # get logprob for each theta
        logprobs = self.log_prob(theta)
        # mode index
        return tf.math.argmax(logprobs, axis=0) # pylint: disable=no-member

    def monte_carlo_theta_at_mode(self, theta, tile_it=True):
        mode = self.monte_carlo_mode(theta, tile_it=tile_it)
        # select from theta at the mode, return another theta
        # if tile_it==True, then copy mode across samples dimension
        return self.select_at_index(theta, mode, tile_it=tile_it)

    def select_at_index(self, theta, mode, tile_it):
        theta_at_mode = DotOperatorSamples()
        for distribution, sample_value in theta.samples.items():
            if distribution in self.distributions:
                mode_value = sample_value[mode]
                if tile_it:
                    mode_value = tf.tile(mode_value, [1, len(sample_value)])
                theta_at_mode.add(distribution, mode_value)
        return theta_at_mode

    def clip(self, theta, stddevs=3, skip=None):
        clipped_theta = DotOperatorSamples()
        for distribution, sample_value in theta.samples.items():
            skip_this = skip is not None and distribution in skip
            if not skip_this:
                clipped_value = self.distributions[distribution].clip(sample_value, stddevs=stddevs)
            else:
                clipped_value = sample_value
            clipped_theta.add(distribution, clipped_value)
        return clipped_theta

    def log_prob_mat(self, theta, stop_grad=False):
        log_probs = []
        for distribution, sample_value in theta.samples.items():
            if distribution in self.distributions:
                log_probs.append(self.distributions[distribution].log_prob(sample_value, stop_grad))
        return tf.stack(log_probs, -1)

    def order_distributions(self):
        names = self.distributions.keys()
        slots = self.slot_dependencies.values()

        orders = OrderedDict()

        while len(orders) < len(names):
            # add distribution if all depenedencies are in orders already
            for name, s in zip(names, slots):
                if name not in orders:

                    # what are dependencies for this
                    dependencies = s.values()

                    all_dependencies = True
                    for dependency in dependencies:
                        if dependency not in orders:
                            print("%s is waiting for %s" % (name, dependency))
                            all_dependencies = False
                            break

                    if all_dependencies is True:
                        orders[name] = len(orders)

        return orders

    def sample(self, list_of_u, verbose, stop_grad=False):
        distribution_id_order = self.order_distributions()

        assert list_of_u.shape[-1] == len(self.distributions), \
            "ChainedDistribution (%s #= %d):: must give a list of u's, one for each distribution." % (
                self.name, list_of_u.shape[-1])
        samples = DotOperatorSamples()

        idx = 0

        if verbose:
            print(distribution_id_order)
        for name, idx in distribution_id_order.items():
        #for name, distribution in zip(self.distributions.iterkeys(), self.distributions.itervalues()):
        #for name, distribution, u in zip(self.distributions.iterkeys(), self.distributions.itervalues(), list_of_u):
            #name = self.distributions.keys()[idx]
            distribution = self.distributions[name]

            if distribution.slots_are_pending():
                print("while sampling, found pending slot for %s"%name)
                distribution.fill_slots(self.slot_dependencies[name], samples)
                assert distribution.slots_are_pending() is False, "STILL pending slot for %s"%name

            if name == "dummy":
                pdb.set_trace()
            theta = distribution.sample(list_of_u[:, :, idx], stop_grad)
            samples.add(name, theta)
        return samples

    def add_distribution(self, key, value, slots=None):
        assert hasattr(self, key) is False, "ChainedDistribution (%s) already has %s" % (self.name, key)
        self.distributions[key] = value
        setattr(self, key, value)
        self.slot_dependencies[key] = slots or {}

    def get_tensors(self):
        tensors = []
        for distribution in self.distributions.values():
            tensors.extend(distribution.get_tensors())
        return tensors

    def attach_summaries(self, plot_histograms):
        """ For each tensor in each distribution add a summary node """
        for name, distribution in self.distributions.items():
            distribution.attach_summaries(name, plot_histograms)

    def add_noise(self, var):
        """ For each tensor in each distribution add random noise to parameter """
        #for _, distribution in zip(self.distributions.iterkeys(), self.distributions.itervalues()):
        for _, distribution in self.distributions.items():
            distribution.add_noise(var)

    def get_tensor_names(self):
        names = []
        #for name, distribution in zip(self.distributions.iterkeys(), self.distributions.itervalues()):
        for name, distribution in self.distributions.items():
            names.extend(distribution.get_tensor_names(name))

        return names

    def get_theta_names(self):
        return list(self.distributions.keys())

    def __str__(self):
        return self.pretty_print()

    def pretty_print(self):
        s = ""
        for key in self.distributions:
            s += "%s = %s slots=[%s]\n" % (key, getattr(self, key), str(self.slot_dependencies[key]))
        return s

    # def get_tensors_by_name(self, theta_names):
    #     tensors = []
    #     for key in theta_names:
    #         tensors.extend(self.distributions[key].get_tensors())
    #     return tensors
    
    # def get_tensor_names(self, theta_names):
    #     names = []
    #     for key in theta_names:
    #         names.extend(self.distributions[key].get_tensor_names(key))
    #     return names


class TfCrnDistribution(ABC):
    def __init__(self, variable: bool):
        self.variable = variable
        self.waiting_slots = {}

    def slots_are_pending(self):
        return any(self.waiting_slots.values())

    def clip(self, sample, stddevs=3):
        return sample

    @abstractmethod
    def get_tensors(self):
        pass

    @abstractmethod
    def log_prob(self, x, stop_grad):
        pass

    @abstractmethod
    def sample(self, u, stop_grad):
        pass

    @abstractmethod
    def get_tensor_names(self, name):
        pass

    @abstractmethod
    def attach_summaries(self, name, plot_histograms):
        pass

    # TODO(dacart): make the arguments to assign_free_and_constrained consistent,
    # then it can be an abstract method too.
    #
    # @abstractmethod
    # def assign_free_and_constrained(...):
    #   pass

class TfConstant(TfCrnDistribution):
    
    def __init__(self, c=None, value=None, wait_for_assigned=False, variable=False):
        super(TfConstant, self).__init__(variable)
        self.value = value
        self.nbr_params = 1
        self.param_names = ["value"]

    def assign_free_and_constrained(self, value):
        self.value = value
        if self.value is not None:
            self.waiting_slots["value"] = False

    def fill_slots(self, slots, samples):
        if self.waiting_slots["value"] is True:
            self.value = getattr(samples, slots['value'])
            self.waiting_slots["value"] = False

    def sample(self, u, stop_grad):  # TODO reshape
        return tf.zeros_like(u) + self.value

    def log_prob(self, x, stop_grad):
        return tf.zeros_like(x)

    def __str__(self):
        s = "%s " % (self.__class__)
        for p_name in self.param_names:
            s += "%s = %s  " % (p_name, getattr(self, p_name))

        return s

    def get_tensors(self):
        return [self.value]

    def attach_summaries(self, name, plot_histograms):
        ()
        # self._attach_summary_ops(self.prec, 'prec', name)

    def get_tensor_names(self, name):
        return ["%s.value" % name]#, "%s.prec" % name]

class TfNormal(TfCrnDistribution):

    def __init__(self, mu=None, c=None, sigma=None, prec=None, variable=True, wait_for_assigned=False):
        super(TfNormal, self).__init__(variable)

        self.waiting_slots["mu"] = True
        self.waiting_slots["prec"] = True
        if wait_for_assigned is True:
            self.mu = mu
            self.sigma = sigma
            self.prec = prec

        else:
            self.mu = mu
            if sigma is None:
                if prec is not None:
                    sigma = 1.0/tf.sqrt(prec)
            else:
                # a sigma param is passed in

                #if prec is not None:
                #    #assert prec is None,  "Need sigma or precision, not both."
                prec = 1.0/(sigma*sigma)

            self.sigma = sigma
            self.prec = prec

        # if we have values for slots, we arent waiting for them
        if self.prec is not None:
            self.waiting_slots["prec"] = False
        if self.mu is not None:
            self.waiting_slots["mu"] = False

        # TODO: remove these guys
        self.nbr_params = 2
        self.param_names = ["mu", "prec"]

    def assign_free_and_constrained(self, mu, log_prec, prec):
        self.mu = mu
        self.log_prec = log_prec
        self.prec = prec
        if prec is not None:
            self.sigma = tf.cast(1.0/tf.sqrt(prec), tf.float32)

        # if we have values for slots, we arent waiting for them
        if self.prec is not None:
            self.waiting_slots["prec"] = False
        if self.mu is not None:
            self.waiting_slots["mu"] = False

    def fill_slots(self, slots, samples):
        if self.waiting_slots["mu"] is True:
            self.mu = getattr(samples, slots['mu'])
            self.waiting_slots["mu"] = False

        if self.waiting_slots["prec"] is True:
            self.prec = getattr(samples, slots['prec'])
            self.waiting_slots["prec"] = False
            self.sigma = tf.cast(1.0/tf.sqrt(self.prec), tf.float32)

    def sample(self, u, stop_grad):
        if stop_grad == True:
            return tf.stop_gradient(self.mu) + tf.stop_gradient(self.sigma)*u
        return self.mu + self.sigma*u

    def clip(self, x, stddevs=3):
        lower = self.mu - stddevs*self.sigma
        upper = self.mu + stddevs*self.sigma
        x = tf.clip_by_value(x, lower, upper)
        return x

    def log_prob(self, x, stop_grad):
        if stop_grad == True:
            prec = tf.stop_gradient(self.prec)
            mu = tf.stop_gradient(self.mu)
        else:
            prec = self.prec
            mu = self.mu
        return -LOG2PI + 0.5 * tf.log(prec + 1e-12) - 0.5 * prec * tf.square(mu - x)
        #return -LOG2PI + 0.5*tf.log(self.prec + 1e-12) -0.5*self.prec*tf.square(self.mu-x)

    def __str__(self):
        s = "%s " % self.__class__
        for p_name in self.param_names:
            s += "%s = %s  " % (p_name, getattr(self, p_name))
        return s

    def get_tensors(self):
        return [self.mu, self.prec]

    def attach_summaries(self, name, plot_histograms):
        if self.variable:
            variable_summaries(self.mu, name + '.mu', plot_histograms)
            variable_summaries(self.prec, name + '.prec', plot_histograms)
        else:
            with tf.name_scope(name):
                tf.summary.scalar('mu', tf.reduce_mean(self.mu))
                tf.summary.scalar('prec', tf.reduce_mean(self.prec))


    def get_tensor_names(self, name):
        return ["%s.mu" % name, "%s.prec" % name]

class TfLogNormal(TfNormal):

    def sample(self, u, stop_grad):
        log_sample = super(TfLogNormal, self).sample(u, stop_grad)
        return tf.exp(log_sample)

    def log_prob(self, x, stop_grad):
        log_x = tf.log(x+1e-12)
        return super(TfLogNormal, self).log_prob(log_x, stop_grad) - log_x

    def clip(self, x, stddevs=3):
        lower = tf.exp(self.mu - stddevs*self.sigma)
        upper = tf.exp(self.mu + stddevs*self.sigma)
        x = tf.clip_by_value(x, lower, upper)
        return x

class TfTruncatedNormal(TfNormal):

    def __init__(self, mu=None, c=None, sigma=None, prec=None, a=None, b=None, wait_for_assigned=False):
        if wait_for_assigned:
            self.mu = mu
            self.sigma = sigma
            self.prec = prec
            self.a = a # left boundary
            self.b = b # right boundary

        else:
            self.mu = mu
            if sigma is None:
                assert prec is not None, "Need sigma or precision"
                sigma = 1.0/tf.sqrt(prec)
            else:
                assert prec is None, "Need sigma or precision, not both."
                prec = 1.0/(sigma*sigma)

            self.sigma = sigma
            self.prec = prec

            if a is None:
                a = -np.inf
            if b is None:
                b = np.inf
            self.a = a
            self.b = b

            self.A = (self.a - self.mu) #/ self.sigma
            self.B = (self.b - self.mu) #/ self.sigma
            #tf.case([(tf.less(-np.inf, self.a), lambda: self.Phi(self.A))], default=lambda: tf.constant(0.0))
            self.PhiA = self.Phi(self.A)
            #tf.case([(tf.less(self.b,  np.inf), lambda: self.Phi(self.B))], default=lambda: tf.constant(1.0))
            self.PhiB = self.Phi(self.B)
            self.PhiA = tf.cast(self.PhiA, tf.float32)
            self.PhiB = tf.cast(self.PhiB, tf.float32)

            self.Z = self.PhiB - self.PhiA
            self.logZ = tf.log(self.Z)

        # TODO: remove these guys
        self.nbr_params = 4
        self.param_names = ["mu", "prec", "a", "b"]

    def assign_free_and_constrained(self, mu, log_prec, prec, a, b):
        self.mu = mu
        self.log_prec = log_prec
        self.prec = prec
        self.sigma = 1.0/tf.sqrt(prec)
        self.a = a
        self.b = b

        self.A = (self.a - self.mu) #/ self.sigma
        self.B = (self.b - self.mu) #/ self.sigma
        #tf.case([(tf.less(-np.inf, self.a), lambda: self.Phi(self.A))], default=lambda: tf.constant(0.0))
        self.PhiA = self.Phi(self.A)
        #tf.case([(tf.less(self.b,  np.inf), lambda: self.Phi(self.B))], default=lambda: tf.constant(1.0))
        self.PhiB = self.Phi(self.B)
        self.PhiA = tf.cast(self.PhiA, tf.float32)
        self.PhiB = tf.cast(self.PhiB, tf.float32)

        #self.PhiA = tf.minimum(self.PhiA, 0.98)
        #self.PhiA = tf.cond(tf.less(-np.inf, self.A), lambda: tf.cast(self.Phi(self.A), tf.float32), lambda: 0.0)
        #self.PhiB = tf.cond(tf.less(self.B,  np.inf), lambda: tf.cast(self.Phi(self.B), tf.float32), lambda: 0.0)
        # if self.A > -np.inf:
        #     self.PhiA = tf.cast(self.Phi(self.A), tf.float32)
        # else:
        #     self.PhiA = 0.0

        # if self.B < np.inf:
        #     self.PhiB = tf.cast(self.Phi(self.B), tf.float32)
        # else:
        #     self.PhiB = 1.0

        self.Z = self.PhiB - self.PhiA
        self.logZ = tf.log(self.Z)

    def sample(self, u, stop_grad):
        raise NotImplementedError("Sample for TfTruncatedNormal hasn't been implemented with stop_grad argument yet ")
        # phi_u = self.Phi(u)
        # standardized_u = self.PhiInverse(self.PhiA + phi_u*self.Z)
        # s = self.mu + self.sigma*standardized_u
        # print("TRUNCATED SAMPLES: ", s.shape, self.mu.shape, self.sigma.shape, u.shape, self.Z.shape,
        #       self.A.shape, self.B.shape, self.PhiA.shape, self.PhiB.shape)
        # return tf.clip_by_value(s, self.a, self.b)


    def log_prob(self, x, stop_grad):
        raise NotImplementedError("log_prob for TfTruncatedNormal hasn't been implemented with stop_grad argument yet ")
        # log_prob_full_support = super(TfTruncatedNormal, self).log_prob(x)        
        # return log_prob_full_support - self.logZ

    def Phi(self, eta):
        #return 0.5*(1 + tf.erf(eta / SQRT2))
        return 0.5*(1 + erf_approx(eta / SQRT2))
        # return 0.5*(1 + erf_approx(eta / SQRT2))

    def PhiInverse(self, u):
        return SQRT2*erfinv_approx(2*u-1.0)
        #return SQRT2*tf.erfc(u)

    def __str__(self):
        s = "%s " % (self.__class__)
        for p_name in self.param_names:
            s += "%s = %s  " % (p_name, getattr(self, p_name))
        return s

    def get_tensors(self):
        return [self.mu, self.prec]

    def get_tensor_names(self, name):
        return ["%s.mu" % name, "%s.prec" % name]

class TfKumaraswamy(TfCrnDistribution):

    # TODO(dacart): set self.prec somehow, as it's needed below.
    def __init__(self, a=None, b=None, zmin=0.0, zmax=1.0, wait_for_assigned=False):
        if not wait_for_assigned:
            self.a = a
            self.b = b
            self.one_over_a = 1.0 / self.a
            self.one_over_b = 1.0 / self.b
            self.log_a = tf.log(self.a)
            self.log_b = tf.log(self.b)

            self.zmin = zmin # left boundary
            self.zmax = zmax # right boundary
            self.zrange = self.zmax-self.zmin


        # TODO: remove these guys
        self.nbr_params = 4
        self.param_names = ["a", "b", "zmin", "zmax"]


    def assign_free_and_constrained(self, log_a, log_b, a, b, zmin, zmax):
        self.a = tf.clip_by_value(a, 0.0001, 1.0/0.0001)
        self.b = tf.clip_by_value(b, 0.0001, 1.0/0.0001)
        self.one_over_a = 1.0 / self.a
        self.one_over_b = 1.0 / self.b
        self.log_a = tf.log(self.a)
        self.log_b = tf.log(self.b)
        self.zmin = zmin # left boundary
        self.zmax = zmax # right boundary
        self.zrange = self.zmax-self.zmin

    def standard_sample(self, u, stop_grad):
        raise NotImplementedError("standard_sample for TfKumaraswamy hasn't been implemented with stop_grad argument yet")

    def sample(self, u, stop_grad):
        raise NotImplementedError("sample for TfKumaraswamy hasn't been implemented with stop_grad argument yet")

    def log_prob(self, x, stop_grad):
        raise NotImplementedError("log_prob for TfKumaraswamy hasn't been implemented with stop_grad argument yet")

    # convert
    def std_normal_2_uniform(self, eta):
        return 0.5*(1 + tf.erf(eta / SQRT2))

    def PhiInverse(self, u):
        return SQRT2*erfinv_approx(2*u-1.0)
        #return SQRT2*tf.erfc(u)

    def __str__(self):
        s = "%s " % (self.__class__)
        for p_name in self.param_names:
            s += "%s = %s  " % (p_name, getattr(self, p_name))
        return s

    def get_tensors(self):
        raise NotImplementedError("Haven't determined how to set mu and prec yet")
        # TODO(dacart): set self.mu and self.prec somewhere
        #return [self.mu, self.prec]

    def get_tensor_names(self, name):
        return ["%s.mu" % name, "%s.prec" % name]