# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

import tensorflow as tf
static_rnn = tf.nn.static_rnn

class ODEDecoder(object):
    def __init__(self, params):
        self.solver = params["solver"]
        self.ode_model = params["model"]

    def __call__(self, conds_obs, dev_1hot, times, thetas, clipped_thetas, condition_on_device):
        x_sample, _f_sample, dev_conditioned = self.ode_model.simulate(
            clipped_thetas, times, conds_obs, dev_1hot, self.solver, condition_on_device)
        # TODO: why just params here and not clipped params?
        x_predict = self.ode_model.observe(x_sample, thetas)
        return x_sample, x_predict, dev_conditioned

RNNDecoder = None  # import hack
# class RNNDecoder(object):
#     def __init__(self, n_time, n_species, n_theta, n_latent_species, params):
#         self.n_time = n_time
#         self.n_species = n_species
#         self.n_theta = n_theta
#         self.n_latent_species = n_latent_species
#         self.n_hidden = n_species + n_latent_species
#         self.n_input = n_theta
#         self.post_simulation = params["model"].post_simulation
#         self.constants = params["constants"]

#     def vectorize_thetas(self, thetas):
#         theta = thetas.get_tensors()  # these are the samples
#         # n_batch x n_samples -->
#         stacked_thetas = tf.stack(theta, axis=-1)  # correct?
#         shapes = tf.shape(stacked_thetas)
#         v_thetas = stacked_thetas.reshape(shapes[0]*shapes[1], -1)
#         return v_thetas

#     def simulate(self, conds_obs, times, thetas, clipped_thetas):
#         state = self.cell.zero_state()  # needs args ?
#         outputs = []
#         z = self.vectorize_thetas(thetas)
#         for t in range(self.n_time + 1):
#             output, state = self.cell(z, state)
#             if t > 0:
#                 outputs.append(output)
#         x_sample = tf.stack(outputs)[:self.n_species]  # use a stack operation here ?
#         # such that time dimension is last
#         x_post_sample = self.post_simulation(x_sample, thetas, self.constants)
#         return x_sample, x_post_sample

#     def __call__(self, latent_placeholder):
#         self.cell = LSTMCell(num_units=self.n_hidden)
#         self.rnn = static_rnn(self.cell, latent_placeholder, sequence_length=1)
