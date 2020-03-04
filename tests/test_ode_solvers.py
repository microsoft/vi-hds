# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

import numpy as np
import tensorflow as tf
import os
import subprocess
import sys
import tempfile
import re

# Call tests in this file by running "pytest" on the directory containing it. For example:
#   cd ~/vi-hds
#   pytest tests

import models.dr_constant
import utils
import procdata
import distributions
from run_xval import Runner, create_parser

# Load a spec (YAML)
parser = create_parser(False)
args = parser.parse_args(['./specs/dr_constant_icml.yaml'])
spec = utils.load_config_file(args.yaml)  # spec is a dict of dicts of dicts
para_settings = utils.apply_defaults(spec['params'], args)
data_settings = procdata.apply_defaults(spec["data"])
model = para_settings['model']

# Load the parameter priors
shared = dict([(k, np.exp(v['mu'])) for k, v in para_settings['shared'].items()])
priors = dict()
priors.update(para_settings['global'])
priors.update(para_settings['global_conditioned'])
priors.update(para_settings['local'])

# Define a parameter sample that is the mode of each LogNormal prior
theta = distributions.DotOperatorSamples()
for k, v in priors.items():
    if k != "conditioning":
        if 'mu' in v: 
            sample_value = np.exp(v['mu'])         
        else: 
            sample_value = shared[v['distribution']]
        theta.add(k, np.tile(sample_value, [1,1]).astype(np.float32))

# Add the constants separately
for k, v in para_settings['constant'].items():
    theta.add(k, np.tile(v, [1,1]).astype(np.float32))

# Set up model runner
trainer = utils.Trainer(args, add_timestamp=True)
self = Runner(args, 0, trainer)
self.params_dict = para_settings
self._prepare_data(data_settings)
self.n_batch = min(self.params_dict['n_batch'], self.dataset_pair.n_train)

# Set various attributes of the model
model = self.params_dict["model"]
model.init_with_params(self.params_dict, self.procdata.relevance_vectors, self.procdata.default_devices)

# Define simulation variables and run simulator
times = np.linspace(0.0, 20.0, 101).astype(np.float32)
conditions = np.array([[1.0, 1.0]]).astype(np.float32)
dev_1hot = np.expand_dims(np.zeros(7).astype(np.float32),0)
sol_rk4 = model.simulate(theta, times, conditions, dev_1hot, 'rk4')[0]
sol_mod = model.simulate(theta, times, conditions, dev_1hot, 'modeulerwhile')[0]

# Run TF session to extract an ODE simulation using modified Euler and RK4
sess = tf.Session()
sess.run(tf.global_variables_initializer()) 
[mod, rk4] = sess.run([sol_mod, sol_rk4])
print(np.shape(mod))

# Ensure that the relative error is no bigger than 5%
Y0 = mod[0][0][-1]
#print(Y0)
Y1 = rk4[0][0][-1]
#print(Y1)
assert np.nanmax(np.abs((Y0 - Y1) / Y0)) < 0.05, 'Difference between Modified Euler and RK4 solvers greater than 5%'