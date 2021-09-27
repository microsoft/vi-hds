# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import os
import time
import numpy as np
import math
import functools
from munch import munchify

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# pylint: disable=no-member,not-callable

from vihds import plotting, utils
from vihds.config import Config
from vihds.encoders import LocalAndGlobal
from vihds.utils import TrainingLogData, Results

def log_prob_observations(model, x_predict, x_obs, precisions, use_laplace=False):    
    # expand x_obs for the iw samples in x_predict
    x_obs_ = torch.unsqueeze(x_obs, 1)
    lpfunc = log_prob_laplace if use_laplace else log_prob_gaussian
    log_prob = lpfunc(x_obs_, x_predict, precisions)
    # sum along the time and observed species axes
    #log_prob = torch.reduce_sum(log_prob, [2, 3])
    # sum along the time axis
    log_prob = torch.sum(log_prob, 3)
    return log_prob

def log_prob_laplace(x_obs, x_predict, precisions):
    log_p_x = torch.log(0.5) + precisions.log() - precisions * torch.abs(x_predict - x_obs)
    return log_p_x

def log_prob_gaussian(x_obs, x_predict, precisions):
    # https://en.wikipedia.org/wiki/Normal_distribution
    log_p_x = -0.5 * (math.log(2.0 * math.pi) - precisions.log() + precisions * (x_predict - x_obs).pow(2))
    return log_p_x

def batch_to_device(times, device, d):
    d['times'] = times.to(device)
    d['dev_1hot'] = d['dev_1hot'].to(device)
    d['inputs'] = d['inputs'].to(device)
    d['observations'] = d['observations'].to(device)
    return munchify(d)

def collate_merged(times, device, batch):
    device_list = [torch.tensor(b['devices']) for b in batch]
    devices = torch.stack(device_list)
    dev_1hot = torch.stack([torch.Tensor(b['dev_1hot']) for b in batch])
    inputs = torch.stack([torch.Tensor(b['inputs']) for b in batch])
    observations = torch.stack([torch.Tensor(b['observations']) for b in batch])

    dd = { 'devices': devices, 'dev_1hot': dev_1hot, 'inputs': inputs, 'observations': observations }    
    return batch_to_device(times, device, dd)

class Training():
    '''Class for orchestrating training of a latent space for dynamical systems using IWAE'''
    def __init__(self, args, settings: Config, data, parameters, model):
        '''Initialise a training routine'''
        # Store arguments
        self.args = args
        self.settings = settings
        self.dataset_pair = data
        self.model = model
        # Prepare the ADAM optimizer
        self.optimizer = torch.optim.Adam(model.parameters(recurse=True), lr=settings.params.learning_rate)
        # Define learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, settings.params.learning_boundaries, gamma=settings.params.learning_gamma)
        # Count the parameters
        n_vals = LocalAndGlobal.from_list(parameters.get_parameter_counts())
        self.model.n_theta = n_vals.sum()
        # Number of instances to put in a training batch.
        self.n_batch = min(settings.params.n_batch, data.n_train)
        # Values to split index batches
        #self.ds_indices = [d - 1 for d in data.train.dataset.cumulative_sizes]
        # Total number of data-points
        #self.n_data = data.train.dataset.cumulative_sizes[-1]

        # Prepare the full training and validation datasets for proper quantification        
        self.train_data = batch_to_device(data.train.dataset.times, settings.device, data.train.dataset[data.train.indices])
        self.valid_data = batch_to_device(data.test.dataset.times, settings.device, data.test.dataset[data.test.indices])
        #self.train_data = [batch_to_device(d.times, settings.device, d) for d in data.train.dataset.datasets]
        #self.valid_data = [batch_to_device(d.times, settings.device, d) for d in data.test.dataset.datasets]

        # Training and test loaders
        self.train_loader = DataLoader(dataset=data.train, batch_size=self.n_batch, shuffle=True,
            collate_fn=functools.partial(collate_merged, data.train.dataset.times, settings.device))

        if settings.trainer is not None:
            # Model path for storing results and tensorboard summaries
            held_out_name = args.heldout or '%d_of_%d' % (args.split, args.folds)
            self.train_path = os.path.join(self.settings.trainer.tb_log_dir, 'train_%s' % held_out_name)
            self.valid_path = os.path.join(self.settings.trainer.tb_log_dir, 'valid_%s' % held_out_name)
            os.makedirs(self.train_path, exist_ok=True)
            os.makedirs(self.valid_path, exist_ok=True)
        else:
            self.train_path = None
            self.valid_path = None
        self.empty_cache = True

    def cost(self, batch_data, batch_results, theta, q, p, full_output=False, writer=None, epoch=None):
        '''Cost function for a VAE model'''
        x_states, x_predict, precisions = batch_results
        log_p_by_species = log_prob_observations(self.model, x_predict, batch_data.observations, 
            precisions, self.settings.params.use_laplace)
        log_p_observations = log_p_by_species.sum(dim=2)        
        log_q_theta = q.log_prob(theta)     # Encoder, q(theta | x, d)
        log_p_theta = p.log_prob(theta)     # Decoder, p(theta | d)
        n_iwae = log_p_observations.shape[1]

        # Evaluate the importance weights 
        log_unnormalized_iws = log_p_observations + log_p_theta - log_q_theta       # log wi = log [p(x | theta) p(theta) / q (theta | x)]
        logsumexp_log_unnormalized_iws = log_unnormalized_iws.logsumexp(axis=1, keepdim=True)     # log \sum wi

        # Either use the vae_cost or the iwae_cost
        #vae_cost = -log_unnormalized_iws.mean()
        iwae_cost = -(logsumexp_log_unnormalized_iws - math.log(n_iwae)).mean()    # E(log \sum_i^k wi - log k)
        elbo = -iwae_cost
        
        #return munchify( {"loss":vae_cost, "iwae_cost":iwae_cost})
        if full_output:
            log_normalized_iws = log_unnormalized_iws - logsumexp_log_unnormalized_iws
            normalized_iws = log_normalized_iws.exp()
            if writer is not None:
                self._update_summaries(writer, epoch, q, log_unnormalized_iws, normalized_iws, log_p_observations, 
                    log_p_by_species, elbo, log_p_theta, log_q_theta)
            output = Results()
            output.init(self.model.decoder.state_names, q, theta, elbo, 
                normalized_iws, x_predict, x_states, precisions)
            return output
        else:
            return munchify({'elbo':iwae_cost})

    def _update_summaries(self, writer, epoch, q, log_unnormalized_iws, normalized_iws, log_p_observations, log_p_by_species, elbo, log_p_theta, log_q_theta):
        '''Update the Tensorboard summaries'''
        ts_to_vis = 1
        plot_histograms = self.settings.params.plot_histograms
        q.attach_summaries(writer, epoch, plot_histograms)  # global and local parameters of q distribution

        # Importance weights
        unnormed_iw = log_unnormalized_iws[ts_to_vis, :]
        normed_iw = normalized_iws[ts_to_vis, :]   # not in log space
        utils.variable_summaries(writer, epoch, unnormed_iw, 'IWS_unn_log', plot_histograms)
        utils.variable_summaries(writer, epoch, normed_iw, 'IWS_normed', plot_histograms)
        writer.add_scalar('IWS_normed/nonzeros', normed_iw.nonzero()[0][0], epoch)

        # ELBO
        writer.add_scalar('ELBO/elbo', elbo, epoch)
        # log(P) and also a per-species breakdown
        writer.add_scalar('ELBO/log_p', log_p_observations.logsumexp(axis=1).mean(), epoch)  # [batch, 1]
        for i,plot in enumerate(self.settings.data.signals):
            log_p_by_species_i = log_p_by_species[:,:,i].logsumexp(axis=1).mean()
            writer.add_scalar('ELBO/log_p_'+plot, log_p_by_species_i, epoch)
        # Priors        
        writer.add_scalar('ELBO/log_prior', log_p_theta.logsumexp(axis=1).mean(), epoch)
        writer.add_scalar('ELBO/loq_q', log_q_theta.logsumexp(axis=1).mean(), epoch)

    def _plot_prediction_summary(self, dataset, output, epoch, writer):
        fig = plotting.plot_prediction_summary(self.settings.data.devices, output.species_names, dataset.times, 
            dataset.observations.cpu().detach().numpy(), output.iw_predict_mu, output.iw_predict_std, dataset.devices, '-')
        writer.add_figure('Summary', fig, global_step=epoch)

    def _plot_species(self, dataset, output, epoch, writer):
        devices = list(range(len(self.settings.data.devices)))
        fig = plotting.species_summary(output.species_names, dataset.inputs.cpu().detach().numpy(), dataset.devices, 
            dataset.times, output.iw_states, devices, self.settings.data)
        writer.add_figure('Species', fig, global_step=epoch)

    def _plot_variance(self, dataset, output, epoch, writer):
        devices = list(range(len(self.settings.data.devices)))
        fig = plotting.species_summary(self.settings.data.signals, dataset.inputs.cpu().detach().numpy(), dataset.devices, 
            dataset.times, output.iw_variance, devices, self.settings.data, normalise=False)
        writer.add_figure('Precisions', fig, global_step=epoch)

    def _plot_weighted_theta_figure(self, train_output, valid_output, valid_writer, epoch, sample):
        name = 'Theta-Resample' if sample else 'Theta-Uniform'
        fig = plotting.plot_weighted_theta(self.encoder.theta_names, train_output.normalized_iws, train_output.theta, 
            self.dataset_pair.train.devices, valid_output.normalized_iws, valid_output.theta, self.dataset_pair.val.devices,
            columns2use=self.settings.params.theta_columns, sample=sample)
        valid_writer.add_figure('Theta/'+name, fig, global_step=epoch)

    def _evaluate_elbo_and_plot(self, epoch, log_data, train_writer, valid_writer):
        print("epoch %4d"%epoch, end='', flush=True)
        log_data.n_test += 1
        test_start = time.time()
        plot = (self.args.plot_epoch > 0) and (np.mod(epoch, self.args.plot_epoch) == 0)
        
        # Training
        train_results, theta, q, p = self.model(self.train_data, self.args.train_samples, writer=train_writer, epoch=epoch)
        train_output = self.cost(self.train_data, train_results, theta, q, p, full_output=True, writer=train_writer, epoch=epoch)
        print(" | train (iwae-elbo = %0.4f, time = %0.2f, total = %0.2f)"%(train_output.elbo, log_data.total_train_time / epoch, log_data.total_train_time), end='', flush=True)
        if train_writer is not None:
            if plot:
                self._plot_prediction_summary(self.train_data, train_output, epoch, train_writer)
                #self._plot_species(self.train_data, train_output, epoch, train_writer)
                if self.model.decoder.ode_model.precisions.dynamic:
                    self._plot_variance(self.train_data, train_output, epoch, train_writer)
            train_writer.flush()
        
        # Validation
        valid_results, theta, q, p = self.model(self.valid_data, self.args.test_samples, writer=valid_writer, epoch=epoch)
        valid_output = self.cost(self.valid_data, valid_results, theta, q, p, full_output=True, writer=valid_writer, epoch=epoch)
        if valid_writer is not None:
            if plot:
                self._plot_prediction_summary(self.valid_data, valid_output, epoch, valid_writer)
                #self._plot_species(self.valid_data, valid_output, epoch, valid_writer)
                if self.model.decoder.ode_model.precisions.dynamic:
                    self._plot_variance(self.valid_data, valid_output, epoch, valid_writer)
            valid_writer.flush()
        log_data.total_test_time += time.time() - test_start
        print(" | val (iwae-elbo = %0.4f, time = %0.2f, total = %0.2f)"%(valid_output.elbo, log_data.total_test_time / log_data.n_test, log_data.total_test_time))
        
        if valid_output.elbo > log_data.max_val_elbo:
            log_data.max_val_elbo = valid_output.elbo
            valid_output.dump()
            self.empty_cache = False
        
        log_data.training_elbo_list.append(train_output.elbo)
        log_data.validation_elbo_list.append(valid_output.elbo)

        return valid_output
    
    def _run_batch(self, epoch_start, batch, log_data):
        # keep track of this time, sometimes (not here) this can be inefficient
        log_data.batch_feed_time += time.time() - epoch_start
        train_start = time.time()
        
        batch_results, theta, q, p = self.model(batch, self.args.train_samples)
        elbo = self.cost(batch, batch_results, theta, q, p).elbo
        if torch.isnan(elbo):
            print("Cannot proceed with ELBO = nan. Exiting.") 
            return False
        elbo.backward()
        # take a training step
        self.optimizer.step()
        self.optimizer.zero_grad()
        # see how long it took
        log_data.batch_train_time += time.time() - train_start
        return True
        
    def run(self):
        # Tensorboard writers
        if self.settings.trainer is not None:
            train_writer = SummaryWriter(self.train_path)
            valid_writer = SummaryWriter(self.valid_path)
        else:
            train_writer = None
            valid_writer = None
        
        log_data = TrainingLogData()
        print("---------------------------")
        if self.args.heldout:
            split_name = 'heldout device = %s' % self.args.heldout
        else:
            split_name = 'split %d of %d' % (self.args.split, self.args.folds)
        print("Training: %s"%split_name)
        iterating = True
        epoch = 1
        while iterating is True and (epoch < self.args.epochs+1):
            self.model.train()
            epoch_start = time.time()
            for batch in self.train_loader:
                if iterating:
                    iterating = self._run_batch(epoch_start, batch, log_data)
            log_data.total_train_time += time.time() - epoch_start
            # Occasionally evaluate ELBO on train and val, using more IW samples
            if iterating and (np.mod(epoch, self.args.test_epoch) == 0):
                self.model.eval()
                valid_output = self._evaluate_elbo_and_plot(epoch, log_data, train_writer, valid_writer)
            self.scheduler.step()
            epoch += 1
        if self.settings.trainer is not None:
            train_writer.close()
            valid_writer.close()
        
        # Reload results from best validation elbo score
        if self.empty_cache:
            print("Exiting with no results in cache")
            return None
        valid_output.load()
        valid_output.elbo_list = log_data.validation_elbo_list
        return valid_output