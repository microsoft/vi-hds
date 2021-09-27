# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from data import procdata
from vihds.config import Config

# pylint: disable=not-callable, no-member


def onehot(i, n):
    """One-hot vector specifiying position i, with length n"""
    v = np.zeros((n))
    if i is not None:
        v[i] = 1
    return v


def depth(group_values):
    return len(set([g for g in group_values if g is not None]))


def get_cassettes(devices, settings):
    """
    devices: list of device indices (positions in self.device_names above)
    Returns a matrix of ones and zeros, where there are ones wherever
    the device (first index) contains the component (second index), with
    component indices taken from S components then R components.
    Each row of the matrix is a cassette.
    """
    rows = []
    for d in devices:
        device_name = settings.device_idx_to_device_name[d]
        vs = [onehot(cm[device_name], depth(cm.values())) for p, cm in settings.component_maps.items()]
        rows.append(np.hstack(vs))
        # r_matrix[idx, r_value] = 1
    if settings.dtype == "float32":
        return np.array(rows).astype(np.float32)
    elif settings.dtype == "float64":
        return np.array(rows).astype(np.float64)
    else:
        raise Exception("Unknown dtype %s" % settings.dtype)


def scale_data(X, settings: Config):
    n_outputs = np.shape(X)[1]
    if settings.normalize is None:
        scales = [np.max(X[:, i, :]).astype(np.float32) for i in range(n_outputs)]
    else:
        scales = settings.normalize
    for i, scale in enumerate(scales):
        # First scale the data
        X[:, i, :] /= scale
        # Second, shift so smallest value for each time series is 0
        if settings.subtract_background:
            mins = np.min(X[:, i, :], axis=1)[:, np.newaxis]
            X[:, i, :] -= mins
    return X, scales


class TimeSeriesDataset(Dataset):
    """A class to facilitate loading batches of time-series observations"""

    def __init__(self, data_settings, params):
        """
        Args:
            file (string): Path to the csv file with time-series observations.
            root_dir (string): Directory with the files.
        """
        # TODO: Create a generic approach to storing and loading data
        # self.parser = data_settings.load
        self.parser = procdata.load

        self.data_settings = data_settings
        self.params = params
        self.n_times = None
        self.n_species = None

    def _preprocess(self, devices, inputs, times, observations):
        self.devices = devices
        # One-hot encoding of device IDs for each of the L observations: (np.ndarray; L)
        self.dev_1hot = torch.tensor(get_cassettes(devices, self.data_settings))
        # Transformed values of C input conditions, for each of the L observations: (np.ndarray; L x C)
        self.inputs = torch.tensor(np.log(1.0 + inputs))
        # Time-points for the observations: (np.ndarray: T)
        self.times = torch.tensor(times)
        self.n_times = len(times)
        # L observations of time-series (length T) with S observables: (np.ndarray; L x T x S):
        obs, self.scales = scale_data(observations, self.data_settings)
        self.observations = torch.tensor(obs)
        self.n_species = np.shape(observations)[1]

    def init_single(self, f):
        devices, inputs, times, observations = self.parser(f, self.data_settings)
        self._preprocess(devices, inputs, times, observations)

    def init_multiple_merge(self):
        devices, inputs, times_list, observations_list = zip(
            *[self.parser(f, self.data_settings) for f in self.data_settings.files]
        )
        times, observations = merge_observations(times_list, observations_list)
        # filter_nonempty = [datasets[i] for i in range(len(datasets)) if datasets[i] is not None]
        # dataset = reduce(merge_files, filter_nonempty)
        self._preprocess(np.concatenate(devices), np.concatenate(inputs), times, observations)

    def __len__(self):
        return len(self.devices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            "devices": self.devices[idx],
            "dev_1hot": self.dev_1hot[idx],
            "inputs": self.inputs[idx],
            "observations": self.observations[idx],
        }


########################################################
# Methods to enable the merging of datasets.
# Currently assumes equal timepoints, which is not ideal.
########################################################


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def merge_observations(times_list, observations_list):
    times_arr = np.asarray(times_list)
    obs_arr = np.asarray(observations_list)
    n_list = np.array([len(t) for t in times_arr])
    loc = np.argmin(n_list)
    chosen_times = times_arr[loc]
    for i, (t, obs) in enumerate(zip(times_arr, obs_arr)):
        locs = list(map(lambda ti: find_nearest(t, ti), chosen_times))
        obs_arr[i] = obs[:, :, locs]
    return chosen_times, np.concatenate(obs_arr)


class TimeSeriesDatasetPair(object):
    """A holder for a training and validation set and various associated parameters."""

    # pylint: disable=too-many-instance-attributes,too-few-public-methods

    def __init__(self, train_dataset: Subset, test_dataset: Subset, data_settings):
        """
        :param train: a Dataset containing the training data
        :param val: a Dataset containing the validation data
        """
        # Dataset of the training data
        self.train = train_dataset
        # Dataset of the validation data
        self.test = test_dataset
        # Number of training instances (int)
        self.n_train = len(train_dataset)
        # Number of validation instances (int)
        self.n_test = len(test_dataset)

        # Number of group-level parameters (summed over all groups; int)
        self.depth = data_settings.device_depth
        # Number of conditions we're training on
        self.n_conditions = len(data_settings.conditions)


def build_datasets(args, config):
    """
    Construct the data workspace
    Arguments:
        data: a dictionary of the form {'devices': [...], 'files': [...]}
            where the devices are names like Pcat_Y81C76 and the files are csv basenames.
    Sets self.dataset_pair to hold the training and evaluation datasets."""

    data_settings = config.data
    # Load all the data in all the files, merging appropriately
    if data_settings.merge:
        dataset = TimeSeriesDataset(data_settings, config.params)
        dataset.init_multiple_merge()
        # datasets.append(dataset)
    else:
        # raise NotImplementedError('TODO: Enable non-merged time-series data')
        datasets = []
        for f in config.data.files:
            d = TimeSeriesDataset(config.data, config.params)
            d.init_single(f)
            datasets.append(d)

        # Create a concatenation of datasets, which will be processed separately in a forward pass through the VAE model
        dataset = ConcatDataset(datasets)

    # Train/test partition
    np.random.seed(args.seed)
    if args.heldout:
        # We specified a holdout device to act as the validation set.
        # d_train, d_val, train_ids, val_ids = split_holdout_device(config.data, loaded_data, args.heldout)
        # train_dataset = _Dataset(config, d_train, train_ids)
        # val_dataset = _Dataset(config, d_val, val_ids)
        # dataset_pair = TimeSeriesDatasetPair(train_dataset, val_dataset)
        raise NotImplementedError("TODO: implement heldout device")
    else:
        loaded_data_length = len(dataset)
        indices = np.random.permutation(loaded_data_length)
        val_chunks = np.array_split(indices, args.folds)
        assert len(val_chunks) == args.folds, "Bad chunks"
        # All the ids from 0 to W-1 inclusive, in order.
        all_ids = np.arange(loaded_data_length, dtype=int)
        # split runs from 1 to args.folds, so the index we need is one less.
        # val_ids is the indices of data items to be used as validation data.
        val_ids = np.sort(val_chunks[args.split - 1])
        train_ids = np.setdiff1d(all_ids, val_ids)

        # A DatasetPair object: two Datasets (one train, one val) plus associated information.
        train = Subset(dataset, train_ids)
        val = Subset(dataset, val_ids)
        dataset_pair = TimeSeriesDatasetPair(train, val, data_settings)

    return dataset_pair
