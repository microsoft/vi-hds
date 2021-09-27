# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import yaml
import torch
import numpy as np

def is_empty(a):
    if a:
        return False
    else:
        return True

def stat_cuda(msg):
    print('- %-12s  ' % (msg+':'), end='')
    print('|  allocated: %4dM  (max: %4dM)  |  cached: %4dM  (max: %4dM)' % (
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_cached() / 1024 / 1024,
        torch.cuda.max_memory_cached() / 1024 / 1024
    ))

def variable_summaries(writer, epoch, var, name, plot_histograms=False):
    """Attach summaries to a scalar node using Tensorboard"""
    mean = var.mean()
    writer.add_scalar(name+'/mean', mean, epoch)
    stddev = (var - mean).pow(2).mean().sqrt()
    writer.add_scalar(name+'/stddev', stddev, epoch)
    writer.add_scalar(name+'/max', var.max(), epoch)
    writer.add_scalar(name+'/min', var.min(), epoch)
    if plot_histograms: 
        writer.add_histogram(name+'/histogram', var, epoch)

def default_get_value(dct, key, default_value, verbose=False):
    if key in dct:
        return dct[key]
    if verbose:
        print("%s using default %s" % (key, str(default_value)))
    return default_value


class TrainingLogData:
    '''A convenience class of data collected for logging during training'''
    def __init__(self):
        '''Initialiser'''
        self.training_elbo_list = []
        self.validation_elbo_list = []
        self.batch_feed_time = 0.0
        self.batch_train_time = 0.0
        self.total_train_time = 0.0
        self.total_test_time = 0.0
        self.n_test = 0
        self.max_val_elbo = -float('inf')


class Results:
    '''A class to store results of running the encoder-decoder and computing the ELBO'''
    def __init__(self):
        self.species_names = None
        #self.times = None
        self.q_names = None
        self.q_values = None
        self.theta = None
        self.elbo = None
        self.iw_predict_mu = None
        self.iw_predict_std = None
        self.iw_states = None

    def init(self, species_names, q, theta, elbo, normalized_iws, x_predict, x_states, precisions):
        '''Initialiser'''
        self.species_names = species_names
        #self.times = times.detach().cpu().numpy()
        self.q_names = q.get_tensor_names()
        self.q_values = np.array([x.detach().cpu().numpy() for x in q.get_tensors()], dtype=object)
        self.theta = np.array([x.detach().cpu().numpy() for x in theta.get_tensors()])
        self.elbo = elbo.detach().cpu().numpy()
        importance_weights = normalized_iws.detach().cpu().numpy()[:, :, np.newaxis, np.newaxis]
        x_predict = x_predict.detach().cpu().numpy()
        x_states = x_states.detach().cpu().numpy()
        precisions = precisions.detach().cpu().numpy()

        self.iw_predict_mu  = np.sum(importance_weights * x_predict, 1)
        self.iw_predict_std = np.sqrt(np.sum(importance_weights * (x_predict**2 + 1.0 / precisions), 1) - self.iw_predict_mu**2)
        self.iw_states      = np.sum(importance_weights * x_states, 1)
        self.iw_variance    = np.sum(importance_weights / precisions, 1)
        
    def dump(self, location='.vihds_cache'):
        os.makedirs(location, exist_ok=True)
        # String lists
        def savetxt(base, data):
            np.savetxt(os.path.join(location, base + '.csv'), np.array(data, dtype=str), delimiter=",", fmt="%s")
        savetxt('species_names', self.species_names)
        savetxt('q_names', self.q_names)
        # Numpy arrays
        def save(base, data):
            np.save(os.path.join(location, base + '.npy'), data)
        #save('times', self.times)
        save('q_values', self.q_values)
        save('theta', self.theta)
        save('elbo', self.elbo)
        save('iw_predict_mu', self.iw_predict_mu)
        save('iw_predict_std', self.iw_predict_std)
        save('iw_states', self.iw_states)
        save('iw_variance', self.iw_variance)

        #save('normalized_iws', self.normalized_iws)
        #save('x_predict', self.x_predict)
        #save('x_states', self.x_states)
        #save('precisions', self.precisions)   
        
    def load(self, location='.vihds_cache'):
        # String lists
        def loadtxt(base):
            return np.loadtxt(os.path.join(location, base + '.csv'), dtype=str, delimiter=",")
        self.species_names = loadtxt('species_names')
        self.q_names = loadtxt('q_names')
        # Numpy arrays
        def load(base):
            return np.load(os.path.join(location, base + '.npy'), allow_pickle=True)
        #self.times = load('times')
        self.q_values = load('q_values')
        self.theta = load('theta')
        self.elbo = load('elbo')
        self.iw_predict_mu = load('iw_predict_mu')
        self.iw_predict_std = load('iw_predict_std')
        self.iw_states = load('iw_states')
        self.iw_variance = load('iw_variance')

        # self.normalized_iws = load('normalized_iws')
        # self.x_predict = load('x_predict')
        # self.x_states = load('x_states')
        # self.precisions = load('precisions')