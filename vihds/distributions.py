# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import OrderedDict
import os
import pdb
import numpy as np

# import tensorflow as tf
import torch
from torch import nn

from vihds.utils import variable_summaries

# pylint: disable = no-member

SQRT2 = np.sqrt(2.0)
LOG2PI = np.log(2 * np.pi)
IDENTITY = "identity"
POSITIVE = "positive"
A_FOR_ERF = 8.0 / (3.0 * np.pi) * (np.pi - 3.0) / (4.0 - np.pi)


def erf_approx(x):
    return x.tanh()


def erfinv_approx(x):
    return x.atanh()


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
        return self.values[0].size()[0]

    def get_n_samples(self):
        return self.values[0].size()[1]

    def get_tensors(self):
        return self.values  # [tensor for tensor in self.samples.values()]


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
            return torch.stack(log_probs, -1).sum(-1)
        # return 0.0 so we can broadcast empty logprob with another non-empty
        return 0.0

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
        return torch.stack(log_probs, -1)

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

    def sample(self, list_of_u, device, stop_grad=False):
        distribution_id_order = self.order_distributions()

        assert list_of_u.shape[-1] == len(self.distributions), (
            "ChainedDistribution (%s #= %d):: must give a list of u's, one for each distribution."
            % (self.name, list_of_u.shape[-1])
        )
        samples = DotOperatorSamples()

        idx = 0

        for name, idx in distribution_id_order.items():
            distribution = self.distributions[name]

            if distribution.slots_are_pending():
                print("while sampling, found pending slot for %s" % name)
                distribution.fill_slots(self.slot_dependencies[name], samples)
                assert distribution.slots_are_pending() is False, "STILL pending slot for %s" % name

            if name == "dummy":
                pdb.set_trace()
            theta = distribution.sample(list_of_u[:, :, idx], stop_grad).to(device)
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

    def attach_summaries(self, writer, epoch, plot_histograms):
        """ For each tensor in each distribution add a summary node """
        for name, distribution in self.distributions.items():
            distribution.attach_summaries(writer, epoch, name, plot_histograms)

    def add_noise(self, var):
        """ For each tensor in each distribution add random noise to parameter """
        for _, distribution in self.distributions.items():
            distribution.add_noise(var)

    def get_tensor_names(self):
        names = []
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
            s += "%s = %s slots=[%s]\n" % (key, getattr(self, key), str(self.slot_dependencies[key]),)
        return s


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
    def attach_summaries(self, writer, epoch, name, plot_histograms):
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
            self.value = getattr(samples, slots["value"])
            self.waiting_slots["value"] = False

    def sample(self, u, stop_grad):  # TODO reshape
        return torch.zeros_like(u) + self.value

    def log_prob(self, x, stop_grad):
        return torch.zeros_like(x)

    def __str__(self):
        s = "%s " % (self.__class__)
        for p_name in self.param_names:
            s += "%s = %s  " % (p_name, getattr(self, p_name))

        return s

    def get_tensors(self):
        return [self.value]

    def attach_summaries(self, writer, epoch, name, plot_histograms):
        ()
        # self._attach_summary_ops(self.prec, 'prec', name)

    def get_tensor_names(self, name):
        return ["%s.value" % name]  # , "%s.prec" % name]


class TfNormal(TfCrnDistribution):
    def __init__(
        self, mu=None, c=None, sigma=None, prec=None, variable=True, wait_for_assigned=False,
    ):
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
                    sigma = 1.0 / prec.sqrt()
            else:
                # a sigma param is passed in

                # if prec is not None:
                #    #assert prec is None,  "Need sigma or precision, not both."
                prec = 1.0 / (sigma * sigma)

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
            self.sigma = 1.0 / prec.sqrt()

        # if we have values for slots, we arent waiting for them
        if self.prec is not None:
            self.waiting_slots["prec"] = False
        if self.mu is not None:
            self.waiting_slots["mu"] = False

    def fill_slots(self, slots, samples):
        if self.waiting_slots["mu"] is True:
            self.mu = getattr(samples, slots["mu"])
            self.waiting_slots["mu"] = False

        if self.waiting_slots["prec"] is True:
            self.prec = getattr(samples, slots["prec"])
            self.waiting_slots["prec"] = False
            self.sigma = 1.0 / self.prec.sqrt()

    def sample(self, u, stop_grad):
        if stop_grad == True:
            return self.mu.detach() + self.sigma.detach() * u
        return self.mu + self.sigma * u

    def clip(self, x, stddevs=3):
        lower = (self.mu - stddevs * self.sigma).data[0]
        upper = (self.mu + stddevs * self.sigma).data[0]
        x = x.clamp(lower, upper)
        return x

    def log_prob(self, x, stop_grad):
        if stop_grad == True:
            prec = self.prec.detach()
            mu = self.mu.detach()
        else:
            prec = self.prec
            mu = self.mu
        return -LOG2PI + 0.5 * (prec + 1e-12).log() - 0.5 * prec * (mu - x).pow(2)

    def __str__(self):
        s = "%s " % self.__class__
        for p_name in self.param_names:
            s += "%s = %s  " % (p_name, getattr(self, p_name))
        return s

    def get_tensors(self):
        return [self.mu, self.prec]

    def attach_summaries(self, writer, epoch, name, plot_histograms):
        if self.variable:
            variable_summaries(writer, epoch, self.mu, name + ".mu", plot_histograms)
            variable_summaries(writer, epoch, self.prec, name + ".prec", plot_histograms)
        else:
            writer.add_scalar("%s/mu" % name, self.mu.mean(), epoch)
            writer.add_scalar("%s/prec" % name, self.prec.mean(), epoch)

    def get_tensor_names(self, name):
        return ["%s.mu" % name, "%s.prec" % name]


class TfLogNormal(TfNormal):
    def sample(self, u, stop_grad):
        log_sample = super(TfLogNormal, self).sample(u, stop_grad)
        return log_sample.exp()

    def log_prob(self, x, stop_grad):
        log_x = (x + 1e-12).log()
        return super(TfLogNormal, self).log_prob(log_x, stop_grad) - log_x

    def clip(self, x, stddevs=3):
        lower = (self.mu - stddevs * self.sigma).exp().data[0]
        upper = (self.mu + stddevs * self.sigma).exp().data[0]
        x = x.clamp(lower, upper)
        return x


class TfTruncatedNormal(TfNormal):
    def __init__(
        self, mu=None, c=None, sigma=None, prec=None, a=None, b=None, wait_for_assigned=False,
    ):
        if wait_for_assigned:
            self.mu = mu
            self.sigma = sigma
            self.prec = prec
            self.a = a  # left boundary
            self.b = b  # right boundary

        else:
            self.mu = mu
            if sigma is None:
                assert prec is not None, "Need sigma or precision"
                sigma = 1.0 / prec.sqrt()
            else:
                assert prec is None, "Need sigma or precision, not both."
                prec = 1.0 / (sigma * sigma)

            self.sigma = sigma
            self.prec = prec

            if a is None:
                a = -np.inf
            if b is None:
                b = np.inf
            self.a = a
            self.b = b

            self.A = self.a - self.mu  # / self.sigma
            self.B = self.b - self.mu  # / self.sigma
            self.PhiA = self.Phi(self.A)
            self.PhiB = self.Phi(self.B)

            self.Z = self.PhiB - self.PhiA
            self.logZ = self.Z.log()

        # TODO: remove these guys
        self.nbr_params = 4
        self.param_names = ["mu", "prec", "a", "b"]

    def assign_free_and_constrained(self, mu, log_prec, prec, a, b):
        self.mu = mu
        self.log_prec = log_prec
        self.prec = prec
        self.sigma = 1.0 / prec.sqrt()
        self.a = a
        self.b = b

        self.A = self.a - self.mu  # / self.sigma
        self.B = self.b - self.mu  # / self.sigma
        self.PhiA = self.Phi(self.A)
        self.PhiB = self.Phi(self.B)

        self.Z = self.PhiB - self.PhiA
        self.logZ = self.Z.log()

    def sample(self, u, stop_grad):
        raise NotImplementedError("Sample for TfTruncatedNormal hasn't been implemented with stop_grad argument yet ")

    def log_prob(self, x, stop_grad):
        raise NotImplementedError("log_prob for TfTruncatedNormal hasn't been implemented with stop_grad argument yet ")

    def Phi(self, eta):
        return 0.5 * (1 + erf_approx(eta / SQRT2))

    def PhiInverse(self, u):
        return SQRT2 * erfinv_approx(2 * u - 1.0)

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
            self.log_a = self.a.log()
            self.log_b = self.b.log()

            self.zmin = zmin  # left boundary
            self.zmax = zmax  # right boundary
            self.zrange = self.zmax - self.zmin

        # TODO: remove these guys
        self.nbr_params = 4
        self.param_names = ["a", "b", "zmin", "zmax"]

    def assign_free_and_constrained(self, log_a, log_b, a, b, zmin, zmax):
        self.a = a.clamp(0.0001, 1.0 / 0.0001)
        self.b = b.clamp(0.0001, 1.0 / 0.0001)
        self.one_over_a = 1.0 / self.a
        self.one_over_b = 1.0 / self.b
        self.log_a = self.a.log()
        self.log_b = self.b.log()
        self.zmin = zmin  # left boundary
        self.zmax = zmax  # right boundary
        self.zrange = self.zmax - self.zmin

    def standard_sample(self, u, stop_grad):
        raise NotImplementedError(
            "standard_sample for TfKumaraswamy hasn't been implemented with stop_grad argument yet"
        )

    def sample(self, u, stop_grad):
        raise NotImplementedError("sample for TfKumaraswamy hasn't been implemented with stop_grad argument yet")

    def log_prob(self, x, stop_grad):
        raise NotImplementedError("log_prob for TfKumaraswamy hasn't been implemented with stop_grad argument yet")

    # convert
    def std_normal_2_uniform(self, eta):
        return 0.5 * (1 + (eta / SQRT2).erf())

    def PhiInverse(self, u):
        return SQRT2 * erfinv_approx(2 * u - 1.0)
        # return SQRT2*torch.erfc(u)

    def __str__(self):
        s = "%s " % (self.__class__)
        for p_name in self.param_names:
            s += "%s = %s  " % (p_name, getattr(self, p_name))
        return s

    def get_tensors(self):
        raise NotImplementedError("Haven't determined how to set mu and prec yet")
        # TODO(dacart): set self.mu and self.prec somewhere
        # return [self.mu, self.prec]

    def get_tensor_names(self, name):
        return ["%s.mu" % name, "%s.prec" % name]
