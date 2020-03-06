# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

import os
import numpy as np
import tensorflow as tf
import procdata
import plotting
from utils import make_summary_image_op, Trainer
import matplotlib.pyplot as pp # pylint:disable=wrong-import-order

class XvalMerge(object):

    def __init__(self, args, data_settings):
        
        self.separated_inputs = data_settings["separate_conditions"]
        self.device_names = data_settings["devices"]
        self.conditions = data_settings["conditions"]
        self.elbo = []
        self.elbo_list = []
        self.epoch = args.epochs
        self.name = args.experiment
        self.label = args.experiment
        self.log_normalized_iws = []
        self.precisions = []
        self.q_names = []
        self.q_values = []
        self.splits = []
        self.theta = []
        self.X_post_sample = []
        self.X_sample = []
        # from data_pair.val
        self.data_ids = []
        self.devices = []
        self.treatments = []
        self.trainer = trainer = Trainer(args, add_timestamp=True)
        self.X_obs = []
        # Attributes initialized elsewhere
        self.chunk_sizes = None
        self.ids = None
        self.names = None
        self.times = None
        self.xval_writer = None

    def add(self, split_idx, data_pair, val_results):
        if split_idx == 1:
            self.q_names = val_results["q_names"]
            self.names = val_results["names"]
            self.times = val_results["times"]
        self.elbo.append(val_results["elbo"])
        self.elbo_list.append(val_results["elbo_list"])
        self.log_normalized_iws.append(val_results["log_normalized_iws"])
        self.precisions.append(val_results["precisions"])
        self.q_values.append(val_results["q_values"])
        self.splits.append(split_idx)
        self.theta.append(val_results["theta"])
        self.X_post_sample.append(val_results["x_post_sample"])
        self.X_sample.append(val_results["x_sample"])
        
        self.data_ids.append(data_pair.val.original_data_ids)
        self.devices.append(data_pair.val.devices)
        self.treatments.append(data_pair.val.treatments)
        self.X_obs.append(data_pair.val.X)

    def finalize(self):
        print('Preparing cross-validation results')
        self.elbo = np.array(self.elbo)
        self.elbo_list = np.array(self.elbo_list)
        self.log_normalized_iws = np.concatenate(self.log_normalized_iws, 0)
        self.precisions = np.concatenate(self.precisions, 0)
        #self.q_values = [np.hstack(q) for q in np.array(self.q_values).transpose()]
        #self.q_values = np.hstack(self.q_values)
        self.q_values = [np.concatenate([np.array(q[i], ndmin=1) for q in self.q_values]) for i,_ in enumerate(self.q_names)]
        self.X_post_sample = np.concatenate(self.X_post_sample, 0)
        self.X_sample = np.concatenate(self.X_sample, 0)

        self.devices = np.concatenate(self.devices, 0)
        self.treatments = np.concatenate(self.treatments, 0)
        self.X_obs = np.concatenate(self.X_obs, 0)
        
        self.chunk_sizes = np.array([len(ids) for ids in self.data_ids])
        self.ids = np.hstack(self.data_ids)
    
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

    def save(self):
        location = self.trainer.tb_log_dir
        def save(base, data):
            np.save(os.path.join(location, base), data)
        def savetxt(base, data):
            np.savetxt(os.path.join(location, base), np.array(data, dtype=str), delimiter=" ", fmt="%s")
        print("Saving to: %s"%location)
        save("xval_result_elbo", self.elbo)
        save("xval_result_elbo_list", self.elbo_list)
        save("xval_result_log_normalized_iws", self.log_normalized_iws)
        save("xval_result_precisions", self.precisions)
        savetxt("xval_result_q_names.txt", self.q_names)
        save("xval_result_q_values", self.q_values)
        save("xval_result_theta", self.theta)
        save("xval_result_X_post_sample", self.X_post_sample)
        save("xval_result_X_sample", self.X_sample)
        
        savetxt("xval_result_device_names.txt", self.device_names)
        save("xval_result_devices", self.devices)
        save("xval_result_treatments", self.treatments)
        save("xval_result_X_obs", self.X_obs)

        save("xval_result_chunk_sizes", self.chunk_sizes)
        save("xval_result_ids", self.ids)
        savetxt("xval_result_names.txt", self.names)
        save("xval_result_times", self.times)        

    def load(self, location=None):
        if location is None:
            location = self.trainer.tb_log_dir
        def load(base):
            return np.load(os.path.join(location, base))
        def loadtxt(base):
            return np.loadtxt(os.path.join(location, base), dtype=str, delimiter=" ")
        self.elbo = load("xval_result_elbo.npy")
        self.elbo_list = load("xval_result_elbo_list.npy")
        self.log_normalized_iws = load("xval_result_log_normalized_iws.npy")
        self.precisions = load("xval_result_precisions.npy")
        self.q_names = loadtxt("xval_result_q_names.txt")
        self.q_values = load("xval_result_q_values.npy")
        self.theta = load("xval_result_theta.npy")
        self.X_sample = load("xval_result_X_sample.npy")
        self.X_post_sample = load("xval_result_X_post_sample.npy")
        
        self.device_names = loadtxt("xval_result_device_names.txt")
        self.devices = load("xval_result_devices.npy")
        self.treatments = load("xval_result_treatments.npy")
        self.X_obs = load("xval_result_X_obs.npy")

        self.chunk_sizes = load("xval_result_chunk_sizes.npy")
        self.ids = load("xval_result_ids.npy")
        self.names = loadtxt("xval_result_names.txt")
        self.times = load("xval_result_times.npy")
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
        device_ids = list(range(len(procdata.device_names)))
        
        print("Making summary figure")
        f1 = plotting.plot_prediction_summary(procdata, self.names, self.times, self.X_obs, self.X_post_sample,
            self.precisions, self.devices, self.log_normalized_iws, '-')
        self.save_figs(f1,'xval_fit')
        plot_op1 = make_summary_image_op(f1, 'Summary', 'Summary')
        self.xval_writer.add_summary(tf.Summary(value=[plot_op1]), self.epoch)
        pp.close(f1)
        self.xval_writer.flush()

        if self.separated_inputs is True:
            print("Making treatment figure")
            f2 = plotting.xval_treatments(self, procdata, device_ids)
            self.save_figs(f2,'xval_treatments')
            plot_op2 = make_summary_image_op(f2, 'Treatment', 'Treatment')
            self.xval_writer.add_summary(tf.Summary(value=[plot_op2]), self.epoch)
            pp.close(f2)
            self.xval_writer.flush()

        print("Making species figure")
        f_species = plotting.species_summary(procdata, self.names, self.treatments, self.devices, self.times, self.X_sample, self.importance_weights, device_ids, fixYaxis = True)
        self.save_figs(f_species,'xval_species')
        plot_op_species = make_summary_image_op(f_species, 'Species', 'Species')
        self.xval_writer.add_summary(tf.Summary(value=[plot_op_species]), self.epoch)
        pp.close(f_species)
        self.xval_writer.flush()

        print("Making global parameters figure")
        f_gparas = plotting.xval_global_parameters(self)
        if f_gparas is not None:
            self.save_figs(f_gparas,'xval_global_parameters')
            plot_op_gparas = make_summary_image_op(f_gparas, 'Parameters', 'Globals')
            self.xval_writer.add_summary(tf.Summary(value=[plot_op_gparas]), self.epoch)
            pp.close(f_gparas)
            self.xval_writer.flush()

        print("Making variable parameters figure")
        f_vparas = plotting.xval_variable_parameters(self)
        if f_vparas is not None:
            self.save_figs(f_vparas,'xval_variable_parameters')
            plot_op_vparas = make_summary_image_op(f_vparas, 'Parameters', 'Variable')
            self.xval_writer.add_summary(tf.Summary(value=[plot_op_vparas]), self.epoch)
            pp.close(f_vparas)
            self.xval_writer.flush()

        print("Making summary device figures")
        summaries = []
        for u in device_ids:
            print("- %s" % procdata.pretty_devices[u])
            device = procdata.device_names[u]
            f4 = plotting.xval_fit_summary(self, u, separatedInputs=self.separated_inputs)
            self.save_figs(f4, 'xval_summary_%s' % device)
            summaries.append(make_summary_image_op(f4, device, 'Device (Summary)'))
            pp.close(f4)        
        self.xval_writer.add_summary(tf.Summary(value=summaries), self.epoch)
        self.xval_writer.flush()
        
        print("Making individual device figures")
        indivs = []
        for u in device_ids:
            print("- %s" % procdata.pretty_devices[u])
            device = procdata.device_names[u]
            f5 = plotting.xval_fit_individual(self, u)
            self.save_figs(f5, 'xval_individual_%s' % device)
            indivs.append(make_summary_image_op(f5, device, 'Device (Individual)'))                
            pp.close(f5)
        self.xval_writer.add_summary(tf.Summary(value=indivs), self.epoch)
        self.xval_writer.flush()

