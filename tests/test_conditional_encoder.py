# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

import numpy as np
import tensorflow.compat.v1 as tf # type: ignore

# Call tests in this file by running "pytest" on the directory containing it. For example:
#   cd ~/vi-hds
#   pytest tests

import src.utils as utils
from src.procdata import apply_defaults
from src.convenience import LocalAndGlobal, Placeholders
from src.encoders import ConditionalEncoder
from src.parameters import Parameters
from src.run_xval import Runner, create_parser

def test_conditional_encoder():

    # Load a spec (YAML)
    parser = create_parser(False)
    args = parser.parse_args(['./specs/dr_constant_icml.yaml'])
    spec = utils.load_config_file(args.yaml)  # spec is a dict of dicts of dicts
    para_settings = utils.apply_defaults(spec['params'])
    data_settings = apply_defaults(spec["data"])

    # Set up model runner
    trainer = utils.Trainer(args, add_timestamp=True)
    self = Runner(args, 0, trainer)
    self.params_dict = para_settings
    self._prepare_data(data_settings)
    self.n_batch = min(self.params_dict['n_batch'], self.dataset_pair.n_train)

    # Import priors from YAML
    parameters = Parameters()
    parameters.load(self.params_dict)

    print("----------------------------------------------")
    if self.args.verbose:
        print("parameters:")
        parameters.pretty_print()
    n_vals = LocalAndGlobal.from_list(parameters.get_parameter_counts())
    self.n_theta = n_vals.sum()

    self.placeholders = Placeholders(self.dataset_pair, n_vals)

    # feed_dicts are used to supply placeholders, these are for the entire train/val dataset, there is a batch one below.
    self._create_feed_dicts()

    # time-series of species differences: x_delta_obs is BATCH x (nTimes-1) x nSpecies
    x_delta_obs = self.placeholders.x_obs[:, 1:, :] - self.placeholders.x_obs[:, :-1, :]

    # Define encoder
    encode = ConditionalEncoder(parameters.params_dict)
    approx_posterior_params = encode(x_delta_obs)

    # Run TF session to extract an ODE simulation using modified Euler and RK4
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 
    [xd, q] = sess.run([x_delta_obs, approx_posterior_params], feed_dict=self.train_feed_dict)
    print("xd:", np.shape(xd))
    print("q:", np.shape(q))

    assert np.shape(q) == (234,50), 'Shape of encoder output'