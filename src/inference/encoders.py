# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

def init_u(n_in):
    return tf.Variable(xavier_init(n_in, 1))

def init_w(n_in):
    return tf.Variable(xavier_init(n_in, 1))

def init_b(n_in):
    return tf.Variable(xavier_init(n_in, 1))

class ConditionalConvolutionalEncoder(object):
    def __init__(self, params):
        #self.seed = params["tensorflow_seed"]
        self.n_filters = params["n_filters"]
        self.filter_size = params["filter_size"]
        self.pool_size = params["pool_size"]
        self.n_hidden = params["n_hidden"]
        self.data_format = params["data_format"]
        self.n_batch = params["n_batch"]
        self.lambda_l2 = params["lambda_l2"]
        self.lambda_l2_hidden = params["lambda_l2_hidden"]
        self.transfer_func = params["transfer_func"]

    def __call__(self, x_obs, x_delta_obs, dev_1hot, conds_obs, name='hidden'):

        self.conv2 = tf.layers.conv1d(
            x_delta_obs,
            filters=self.n_filters,
            kernel_size=self.filter_size,
            data_format=self.data_format,
            name='conv2_species',
            kernel_regularizer=tf.keras.regularizers.l2(self.lambda_l2),
            kernel_initializer=tf.orthogonal_initializer())

        # keep stride to 1, otherwise get saw-like filters
        self.pool2 = tf.layers.average_pooling1d(
            inputs=self.conv2,
            pool_size=self.pool_size,
            strides=1,
            data_format=self.data_format,
            name='pool2_species')

        encoded = tf.layers.dense(
            tf.layers.flatten(self.pool2),
            activation=self.transfer_func,
            units=self.n_hidden,
            name="encoded",
            kernel_regularizer=tf.keras.regularizers.l2(self.lambda_l2_hidden),
            kernel_initializer=tf.orthogonal_initializer())
        return encoded