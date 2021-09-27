# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from vihds import plotting


class XvalMerge(object):
    def __init__(self, args, settings):
        self.epoch = args.epochs
        self.elbo = []
        self.elbo_list = []
        self.q_names = []
        self.q_values = []
        self.splits = []
        self.theta = []
        # self.normalized_iws = []
        # self.precisions = []
        # self.X_predict = []
        # self.X_states = []
        self.iw_predict_mu = []
        self.iw_predict_std = []
        self.iw_states = []
        # from data_pair.test
        self.data_ids = []
        self.devices = []
        self.treatments = []
        self.X_obs = []
        # Attributes initialized elsewhere
        self.chunk_sizes = None
        self.ids = None
        self.species_names = None
        self.times = None
        self.xval_writer = None
        self.settings = settings.data
        self.trainer = settings.trainer

    def add(self, split_idx, data_pair, val_results):
        if split_idx == 1:
            self.q_names = val_results.q_names
            self.species_names = val_results.species_names
            self.times = data_pair.train.dataset.times
        self.elbo.append(val_results.elbo)
        self.elbo_list.append(val_results.elbo_list)
        self.q_values.append(val_results.q_values)
        self.splits.append(split_idx)
        self.theta.append(val_results.theta)
        # self.normalized_iws.append(val_results.normalized_iws)
        # self.precisions.append(val_results.precisions)
        # self.X_predict.append(val_results.x_predict)
        # self.X_states.append(val_results.x_states)
        self.iw_predict_mu.append(val_results.iw_predict_mu)
        self.iw_predict_std.append(val_results.iw_predict_std)
        self.iw_states.append(val_results.iw_states)

        self.data_ids.append(data_pair.test.indices)
        dataset = data_pair.test.dataset[data_pair.test.indices]
        self.devices.append(dataset["devices"])
        self.treatments.append(dataset["inputs"].cpu().detach().numpy())
        self.X_obs.append(dataset["observations"].cpu().detach().numpy())

    def finalize(self):
        print("Preparing cross-validation results")
        self.elbo = np.array(self.elbo)
        self.elbo_list = np.array(self.elbo_list)
        self.q_values = [
            np.concatenate([np.array(q[i], ndmin=1) for q in self.q_values]) for i, _ in enumerate(self.q_names)
        ]
        # self.normalized_iws = np.concatenate(self.normalized_iws, 0)
        # self.precisions = np.concatenate(self.precisions, 0)
        # self.X_predict = np.concatenate(self.X_predict, 0)
        # self.X_states = np.concatenate(self.X_states, 0)
        self.iw_predict_mu = np.concatenate(self.iw_predict_mu, 0)
        self.iw_predict_std = np.concatenate(self.iw_predict_std, 0)
        self.iw_states = np.concatenate(self.iw_states, 0)

        self.devices = np.concatenate(self.devices, 0)
        self.treatments = np.concatenate(self.treatments, 0)
        self.X_obs = np.concatenate(self.X_obs, 0)

        self.chunk_sizes = np.array([len(ids) for ids in self.data_ids], dtype=object)
        self.ids = np.hstack(self.data_ids)

    def prepare(self):
        '''Importance-weighted means and stds over time'''
        importance_weights = self.normalized_iws[:, :, np.newaxis, np.newaxis]
        self.iw_predict_mu = np.sum(importance_weights * self.X_predict, 1)
        self.iw_predict_std = np.sqrt(np.sum(importance_weights * (self.X_predict**2 + 1.0 / self.precisions), 1)
                                      - self.iw_predict_mu**2)
        self.iw_states = np.sum(importance_weights * self.X_states, 1)

    def save(self):
        location = self.trainer.tb_log_dir
        print("Saving results to %s" % location)

        def save(base, data):
            np.save(os.path.join(location, base + ".npy"), data)

        def savetxt(base, data):
            np.savetxt(
                os.path.join(location, base + ".txt"), np.array(data, dtype=str), delimiter=" ", fmt="%s",
            )

        print("Saving to: %s" % location)
        save("xval_elbo", self.elbo)
        save("xval_elbo_list", self.elbo_list)
        savetxt("xval_q_names", self.q_names)
        save("xval_q_values", self.q_values)
        save("xval_theta", self.theta)

        save("xval_iw_predict_mu", self.iw_predict_mu)
        save("xval_iw_predict_std", self.iw_predict_std)
        save("xval_iw_states", self.iw_states)
        # save("xval_normalized_iws", self.normalized_iws)
        # save("xval_precisions", self.precisions)
        # save("xval_X_predict", self.X_predict)
        # save("xval_X_states", self.X_states)

        savetxt("xval_device_names", self.settings.devices)
        save("xval_devices", self.devices)
        save("xval_treatments", self.treatments)
        save("xval_X_obs", self.X_obs)

        save("xval_chunk_sizes", self.chunk_sizes)
        save("xval_ids", self.ids)
        savetxt("xval_names", self.species_names)
        save("xval_times", self.times)

    def load(self, location=None):
        if location is None:
            location = self.trainer.tb_log_dir
        print("Loading results from %s" % location)

        def load(base):
            return np.load(os.path.join(location, base + ".npy"), allow_pickle=True)

        def loadtxt(base):
            return np.loadtxt(os.path.join(location, base + ".txt"), dtype=str, delimiter=" ")

        self.elbo = load("xval_elbo")
        self.elbo_list = load("xval_elbo_list")
        self.q_names = loadtxt("xval_q_names")
        self.q_values = load("xval_q_values")
        self.theta = load("xval_theta")
        # self.normalized_iws = load("xval_normalized_iws")
        # self.precisions = load("xval_precisions")
        # self.X_states = load("xval_X_states")
        # self.X_predict = load("xval_X_predict")
        self.iw_predict_mu = load("xval_iw_predict_mu")
        self.iw_predict_std = load("xval_iw_predict_std")
        self.iw_states = load("xval_iw_states")

        # self.device_names = loadtxt("xval_device_names.txt")
        self.devices = load("xval_devices")
        self.treatments = load("xval_treatments")
        self.X_obs = load("xval_X_obs")

        self.chunk_sizes = load("xval_chunk_sizes")
        self.ids = load("xval_ids")
        self.species_names = loadtxt("xval_names")
        self.times = load("xval_times")

    def make_writer(self, location=None):
        if location is None:
            location = self.trainer.tb_log_dir
        self.xval_writer = SummaryWriter(os.path.join(location, "xval"))

    def close_writer(self):
        self.xval_writer.close()

    def save_figs(self, f, tag):
        # pp.close(f)
        f.savefig(os.path.join(self.trainer.tb_log_dir, "%s.png" % tag), bbox_inches="tight")
        f.savefig(os.path.join(self.trainer.tb_log_dir, "%s.pdf" % tag), bbox_inches="tight")

    def mark_completed(self, node_name):
        location = self.trainer.tb_log_dir
        filepath = os.path.join(location, "completed.txt")
        with open(filepath, "w") as file:
            file.write(node_name)
            file.close()

    def make_images(self):
        device_ids = list(range(len(self.settings.devices)))

        print("Making summary figure")
        f_summary = plotting.plot_prediction_summary(
            self.settings.devices,
            self.species_names,
            self.times,
            self.X_obs,
            self.iw_predict_mu,
            self.iw_predict_std,
            self.devices,
            "-",
        )
        self.save_figs(f_summary, "xval_fit")
        self.xval_writer.add_figure("Summary", f_summary, self.epoch)
        self.xval_writer.flush()

        if self.settings.separate_conditions is True:
            print("Making treatment figure")
            f_treatments = plotting.xval_treatments(self, device_ids)
            self.save_figs(f_treatments, "xval_treatments")
            self.xval_writer.add_figure("Treatment", f_treatments, self.epoch)
            self.xval_writer.flush()

        print("Making species figure")
        f_species = plotting.species_summary(
            self.species_names, self.treatments, self.devices, self.times, self.iw_states, device_ids, self.settings,
        )
        self.save_figs(f_species, "xval_species")
        self.xval_writer.add_figure("Species", f_species, self.epoch)
        self.xval_writer.flush()

        print("Making global parameters figure")
        f_gparas = plotting.xval_global_parameters(self)
        if f_gparas is not None:
            self.save_figs(f_gparas, "xval_global_parameters")
            self.xval_writer.add_figure("Parameters/Globals", f_gparas, self.epoch)
            self.xval_writer.flush()

        print("Making variable parameters figure")
        f_vparas = plotting.xval_variable_parameters(self)
        if f_vparas is not None:
            self.save_figs(f_vparas, "xval_variable_parameters")
            self.xval_writer.add_figure("Parameters/Variable", f_vparas, self.epoch)
            self.xval_writer.flush()

        print("Making summary device figures")
        for u in device_ids:
            print("- %s" % self.settings.pretty_devices[u])
            device = self.settings.devices[u]
            f_summary_i = plotting.xval_fit_summary(self, u, separatedInputs=self.settings.separate_conditions)
            self.save_figs(f_summary_i, "xval_summary_%s" % device)
            self.xval_writer.add_figure("Device_Summary/" + device, f_summary_i, self.epoch)
        self.xval_writer.flush()

        print("Making individual device figures")
        for u in device_ids:
            print("- %s" % self.settings.pretty_devices[u])
            device = self.settings.devices[u]
            if self.settings.separate_conditions is True:
                # TODO: Check just 1 treatment? 2treatments function fails when there is just 1 treatment
                f_indiv_i = plotting.xval_individual_2treatments(self, u)
            else:
                f_indiv_i = plotting.xval_individual(self, u)
            self.save_figs(f_indiv_i, "xval_individual_%s" % device)
            self.xval_writer.add_figure("Device_Individual/" + device, f_indiv_i, self.epoch)
        self.xval_writer.flush()
