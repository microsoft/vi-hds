# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

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

def ConditionalEncoder(params):
    n_filters = params["n_filters"]
    filter_size = params["filter_size"]
    pool_size = params["pool_size"]
    n_hidden = params["n_hidden"]
    data_format = params["data_format"]
    lambda_l2 = params["lambda_l2"]
    lambda_l2_hidden = params["lambda_l2_hidden"]
    transfer_func = params["transfer_func"]
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(n_filters, filter_size, 
            data_format=data_format, 
            kernel_regularizer=tf.keras.regularizers.l2(lambda_l2),
            kernel_initializer=tf.orthogonal_initializer()),
        tf.keras.layers.AveragePooling1D(pool_size=pool_size, strides=1, data_format=data_format),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            units=n_hidden,
            activation=transfer_func,
            kernel_initializer=tf.orthogonal_initializer(),
            kernel_regularizer=tf.keras.regularizers.l2(lambda_l2_hidden))
    ])