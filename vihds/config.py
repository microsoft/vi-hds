# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import os
import datetime
import re
import shutil
from munch import munchify
from collections import OrderedDict
import numpy as np
import yaml
import torch

# pylint: disable=no-member


def _tidy_args(args):
    print("Processing command-line arguments")
    print("-", args)
    if args.test_epoch > args.epochs:
        print("- Setting test_epoch to %d" % args.epochs)
        args.test_epoch = args.epochs
    if args.plot_epoch > args.epochs:
        print("- Setting plot_epoch to %d" % args.epochs)
        args.plot_epoch = args.epochs

    """Try to fix the PyTorch and Numpy random seeds to achieve deterministic behaviour.
        2020-03-31: looks like it is working."""
    seed = args.seed
    if seed is not None:
        print("- Setting: np.random.seed({})".format(seed))
        np.random.seed(seed)
        print("- Setting: torch.manual_seed({})".format(seed))
        torch.manual_seed(seed)

    return args


def locate_yml(folder):
    matches = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".yaml"):
                matches.append(os.path.join(root, file))
    return matches[0]


def _unique_dir_name(name, results_dir="Results"):
    now = datetime.datetime.now().replace(microsecond=0).isoformat()
    time_code = re.sub("[^A-Za-z0-9]+", "", now)  # current date and time concatenated into str for logging
    experiment = name + "_" + time_code
    return os.path.join(results_dir, experiment)


def apply_defaults_params(config):
    defaults = munchify(
        {
            "solver": "midpoint",
            "adjoint_solver": False,
            "use_laplace": False,
            "n_filters": 10,
            "filter_size": 10,
            "pool_size": 5,
            "lambda_l2": 0.001,
            "lambda_l2_hidden": 0.001,
            "n_hidden": 50,
            "n_hidden_decoder": 50,
            "n_batch": 36,
            "data_format": "channels_last",
            "precision_type": "constant",
            "precision_alpha": 1000.0,
            "precision_beta": 1.0,
            "init_prec": 0.00001,
            "init_latent_species": 0.001,
            "transfer_func": "tanh",
            "n_hidden_decoder_precisions": 20,
            "n_growth_layers": 4,
            "tb_gradients": False,
            "plot_histograms": False,
            "learning_boundaries": [250, 500],
            "learning_rate": 0.01,
            "learning_gamma": 0.2,
        }
    )
    for k in config:
        defaults[k] = config[k]
    return defaults


def depth(group_values):
    return len(set([g for g in group_values if g is not None]))


def proc_data(data_settings):
    # Group-level parameter assignments for each device
    groups_list = [[k, v] for k, v in data_settings.groups.items()]
    data_settings.component_maps = OrderedDict()
    for k, group in groups_list:
        data_settings.component_maps[k] = OrderedDict(zip(data_settings.devices, group))
    # Total number of group-level parameters
    data_settings.device_depth = sum([depth(cm.values()) for k, cm in data_settings.component_maps.items()])
    # Relevance vectors for decoding multi-hot vector into multiple one-hot vectors
    data_settings.relevance_vectors = OrderedDict()
    k1 = 0
    for k, group in groups_list:
        k2 = depth(group) + k1
        rv = np.zeros(data_settings.device_depth)
        rv[k1:k2] = 1.0
        # print("Relevance for %s: "%k + str(rv))
        if k in data_settings.default_devices:
            rv[k1 + data_settings.default_devices[k]] = 0.0
        data_settings.relevance_vectors[k] = rv.astype(np.float32)
        k1 = k2
    # Manually curated device list: map from device names to 0.0, 1.0, ...
    data_settings.device_map = dict(zip(data_settings.devices, (float(v) for v in range(len(data_settings.devices)))))
    # Map from device indices (as ints) to device names
    data_settings.device_idx_to_device_name = dict(enumerate(data_settings.devices))
    # Map from device indices (as floats) to device names
    data_settings.device_lookup = {v: k for k, v in data_settings.device_map.items()}
    return data_settings


def apply_defaults_data(config):
    ndevices = len(config.devices)
    defaults = munchify(
        {
            "groups": {"default": [0] * ndevices},
            "default_devices": dict(),
            "normalize": None,
            "merge": True,
            "subtract_background": True,
            "separate_conditions": False,
            "dtype": "float32",
        }
    )
    for k in config:
        defaults[k] = config[k]
    defaults.data_dir = get_data_directory()
    return proc_data(defaults)


class Config(object):
    def __init__(self, args):
        args = _tidy_args(args)
        if args.yaml is None:
            return None
        with open(args.yaml, "r") as stream:
            config = munchify(yaml.safe_load(stream))
            # return Munch.fromYAML(stream)
        self.data = apply_defaults_data(config.data)
        # self.models = config.models
        # self.experiments = {}
        # for node, data_settings in config.experiments.items():
        #     self.experiments[node] = apply_defaults_data(data_settings)
        self.params = apply_defaults_params(config.params)
        if args.precision_hidden_layers is not None:
            self.params.n_hidden_decoder_precisions = args.precision_hidden_layers
        self.model = config.model
        self.seed = args.seed
        if (args.gpu is not None) & torch.cuda.is_available():
            print("- GPU mode computation")
            self.device = torch.device("cuda:" + str(args.gpu))
            if self.data.dtype == "float32":
                torch.set_default_tensor_type("torch.cuda.FloatTensor")
            elif self.data.dtype == "float64":
                torch.set_default_tensor_type("torch.cuda.DoubleTensor")
            else:
                raise Exception("Unknown dtype %s" % self.data.dtype)
        else:
            print("- CPU mode computation")
            self.device = torch.device("cpu")
            if self.data.dtype == "float32":
                torch.set_default_tensor_type("torch.FloatTensor")
            elif self.data.dtype == "float64":
                torch.set_default_tensor_type("torch.DoubleTensor")
            else:
                raise Exception("Unknown dtype %s" % self.data.dtype)
        self.trainer = None  # Trainer(args, log_dir=log_dir, add_timestamp=True)


def get_data_directory():
    """Returns directory where observation datasets are stored (default: "data")"""
    data_dir = os.getenv("INFERENCE_DATA_DIR")
    if data_dir:
        return data_dir
    else:
        return "data"


def get_results_directory():
    """
    Returns mount directory of remote machine on local, where inference results are to be stored
    (default: "results")
    """
    results_dir = os.getenv("INFERENCE_RESULTS_DIR")
    if results_dir:
        return results_dir
    else:
        return "results"


class Trainer(object):
    """Collection functions and attributes for training a Model"""

    def __init__(self, args, log_dir=None, add_timestamp=False):
        self.results_dir = get_results_directory()
        self.experiment = args.experiment
        self.yaml_file_name = args.yaml
        if log_dir is None:
            self.create_logging_dirs(add_timestamp)
        else:
            self.tb_log_dir = log_dir

    def _unique_dir_name(self, experiment, add_timestamp):
        now = datetime.datetime.now().isoformat()
        time_code = re.sub("[^A-Za-z0-9]+", "", now)  # current date and time concatenated into str for logging
        if add_timestamp is True:
            experiment += "_" + time_code
        return os.path.join(self.results_dir, experiment)

    def create_logging_dirs(self, add_timestamp=False):
        self.tb_log_dir = self._unique_dir_name(self.experiment, add_timestamp)
        os.makedirs(self.tb_log_dir, exist_ok=True)
        shutil.copyfile(
            self.yaml_file_name, os.path.join(self.tb_log_dir, os.path.basename(self.yaml_file_name)),
        )
