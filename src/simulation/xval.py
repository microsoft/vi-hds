# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import numpy as np
import tensorflow as tf

from plotting import (plot_prediction_summary,
                            xval_species_summary,
                            xval_fit_individual,
                            xval_treatments)
from utils import make_summary_image_op
import matplotlib.pyplot as pp # pylint:disable=wrong-import-order

class XvalMerge(object):

    def __init__(self, args, trainer):
        self.epoch = args.epochs
        self.name = args.experiment
        self.label = args.experiment
        self.splits = []
        self.X_post_sample = []
        self.X_sample = []
        self.precisions = []
        self.log_normalized_iws = []
        self.elbo = []

        # from data_pair.val
        self.data_ids = []
        self.X_obs = []
        self.devices = []
        self.treatments = []
        self.trainer = trainer

        # Attributes initialized elsewhere
        self.chunk_sizes = None
        self.ids = None
        self.names = None
        self.times = None
        self.xval_writer = None

    def add(self, split_idx, data_pair, val_results):
        if split_idx == 1:
            self.names = val_results["names"]
            self.times = val_results["times"]
        self.splits.append(split_idx)
        self.data_ids.append(data_pair.val.original_data_ids)
        self.X_obs.append(data_pair.val.X)
        self.treatments.append(data_pair.val.treatments)
        self.devices.append(data_pair.val.devices)
        self.X_sample.append(val_results["x_sample"])
        self.X_post_sample.append(val_results["x_post_sample"])
        self.precisions.append(val_results["precisions"])
        self.log_normalized_iws.append(val_results["log_normalized_iws"])
        self.elbo.append(val_results["elbo"])

    def finalize(self):
        print('Preparing cross-validation results')
        self.chunk_sizes = np.array([len(ids) for ids in self.data_ids])
        self.ids = np.hstack(self.data_ids)
        self.X_obs = np.concatenate(self.X_obs, 0)
        self.X_sample = np.concatenate(self.X_sample, 0)
        self.X_post_sample = np.concatenate(self.X_post_sample, 0)
        self.log_normalized_iws = np.concatenate(self.log_normalized_iws, 0)
        self.precisions = np.concatenate(self.precisions, 0)
        self.treatments = np.concatenate(self.treatments, 0)
        self.devices = np.concatenate(self.devices, 0)
        self.elbo = np.array(self.elbo)
        #self.xval_elbo = np.sum(self.elbo * self.chunk_sizes / float(self.chunk_sizes.sum()), 0)

    def save(self, location=None):
        if location is None:
            location = self.trainer.savedir
        def save(base, data):
            np.save(os.path.join(location, base), data)
        print("Saving to: %s"%location)
        save("xval_result_times", self.times)
        np.savetxt(os.path.join(location, "xval_result_names.txt"), np.array(self.names, dtype=str), delimiter=" ", fmt="%s")
        save("xval_result_chunk_sizes", self.chunk_sizes)
        save("xval_result_ids", self.ids)
        save("xval_result_X_obs", self.X_obs)
        save("xval_result_X_sample", self.X_sample)
        save("xval_result_X_post_sample", self.X_post_sample)
        save("xval_result_log_normalized_iws", self.log_normalized_iws)
        save("xval_result_precisions", self.precisions)
        save("xval_result_treatments", self.treatments)
        save("xval_result_devices", self.devices)
        np.savetxt(os.path.join(location, "xval_result_elbo.txt"), self.elbo, fmt="%.4f")
        #np.savetxt(os.path.join(location, "xval_result_xval_elbo.txt",
        #           np.array([self.xval_elbo], dtype=float), fmt="%.4f")

    def prepare_treatment(self):
        if len(self.precisions.shape) == 3:
            PREC = self.precisions[:,:,np.newaxis,:]
            PREC = np.tile(PREC, [1,1,self.X_post_sample.shape[2],1])
        else:
            PREC = self.precisions
        STD = 1.0 / np.sqrt(PREC)

        self.PREDICT = self.X_post_sample[:,:,-1,:]
        self.STD = STD[:,:,-1,:]
        self.log_ws = self.log_normalized_iws[:, :, None, None]
        self.importance_weights = np.exp(np.squeeze(self.log_ws))

    def load(self, location=None):
        if location is None:
            location = self.trainer.tb_log_dir
        def load(base):
            return np.load(os.path.join(location, base))

        self.times = load("xval_result_times.npy")
        self.names = np.loadtxt(os.path.join(location, "xval_result_names.txt"), dtype=str, delimiter=" ")
        self.chunk_sizes = load("xval_result_chunk_sizes.npy")
        self.ids = load("xval_result_ids.npy")
        self.X_obs = load("xval_result_X_obs.npy")
        self.X_sample = load("xval_result_X_sample.npy")
        self.X_post_sample = load("xval_result_X_post_sample.npy")
        self.log_normalized_iws = load("xval_result_log_normalized_iws.npy")
        self.precisions = load("xval_result_precisions.npy")
        self.treatments = load("xval_result_treatments.npy")
        self.devices = load("xval_result_devices.npy")
        self.elbo = np.loadtxt(os.path.join(location, "xval_result_elbo.txt"), dtype=float)
        #self.xval_elbo = np.loadtxt(os.path.join(location, "xval_result_xval_elbo.txt", dtype=float))
        self.prepare_treatment()

    def make_writer(self, location=None):
        if location is None:
            location = self.trainer.tb_log_dir
        self.xval_writer = tf.summary.FileWriter(os.path.join(location, 'xval'))

    def close_writer(self):
        self.xval_writer.close()
    
    def save_figs(self, f, tag):
        f.savefig(os.path.join(self.trainer.tb_log_dir,'%s.png'%tag),bbox_inches='tight')
        f.savefig(os.path.join(self.trainer.tb_log_dir,'%s.pdf'%tag),bbox_inches='tight')

    def make_images(self, procdata):
        print("Making summary figure")
        f1 = plot_prediction_summary(procdata, self.names, self.times, self.X_obs, self.X_post_sample,
            self.precisions, self.devices, self.log_normalized_iws, '-')
        plot_op1 = make_summary_image_op(f1, 'Summary')
        self.xval_writer.add_summary(plot_op1, self.epoch)
        self.save_figs(f1,'xval_fit')
        pp.close(f1)
        self.xval_writer.flush()

        print("Making treatment figure")
        devices = [2,3,4,5,6,7]
        pretty_devices = ['Pcat-Pcat','RS100-S32','RS100-S34','R33-S32','R33-S34','R33-S175']
        f2 = xval_treatments(self, procdata, devices, pretty_devices)
        plot_op2 = make_summary_image_op(f2, 'Treatment')
        self.xval_writer.add_summary(plot_op2, self.epoch)
        self.save_figs(f2,'xval_treatments')
        pp.close(f2)
        self.xval_writer.flush()

        print("Making species figure")
        nvars = np.shape(self.X_sample)[3] - 4
        f3 = xval_species_summary(self, procdata, devices, pretty_devices, nvars, True)
        plot_op3 = make_summary_image_op(f3, 'Species')
        self.xval_writer.add_summary(plot_op3, self.epoch)
        self.save_figs(f3,'xval_species')
        pp.close(f3)
        self.xval_writer.flush()

        print("Making all device figures")
        for u in devices:
            file = xval_fit_individual(self, procdata, u)
            device = procdata.device_lookup[u]
            print("- %s" % device)
            self.save_figs(file, 'xval_%s' % device)
            plot_op = make_summary_image_op(file, 'Device %s' % device)
            self.xval_writer.add_summary(plot_op, self.epoch)
            pp.close(file)
            self.xval_writer.flush()