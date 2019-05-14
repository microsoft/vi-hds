# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np

def gaussian_log_prob(x, mu, vr, log_vr):
    lp = -0.5*np.sqrt(2.0*np.pi) - 0.5*log_vr -0.5*tf.square(x-mu)/vr
    return lp

def log_pdf_data(x_sample, x_obs, vr_data, log_vr_data):
    assert False, 'do not use'
    #return tf.reduce_sum(tf.distributions.Normal(loc=x_obs, scale=tf.sqrt(vr_data)).log_prob(x_sample), [2, 1])
    #return tf.reduce_sum(gaussian_log_prob(x_sample, x_obs, vr_data, log_vr_data), [1, 2])

def log_pdf_prior(log_theta, log_theta_mu, log_theta_log_var, log_theta_var):
    return tf.reduce_sum(tf.distributions.Normal(loc=log_theta_mu, scale=tf.sqrt(log_theta_var)).log_prob(log_theta), axis=-1)
    #return tf.reduce_sum(tf.distributions.Normal(loc=tf.expand_dims(log_theta_mu, 1), scale=tf.expand_dims(tf.sqrt(log_theta_var), 1)).log_prob(log_theta), axis=-1)
    #return tf.reduce_sum(gaussian_log_prob(log_theta, log_theta_mu, log_theta_log_var, log_theta_var), axis=1)

def log_pdf_q(log_theta, log_theta_mu, log_theta_log_var, log_theta_var):
    return tf.reduce_sum(tf.distributions.Normal(loc=tf.expand_dims(log_theta_mu, 1), scale=tf.expand_dims(tf.sqrt(log_theta_var), 1)).log_prob(log_theta), axis=-1)
    #return tf.reduce_sum(gaussian_log_prob(log_theta, log_theta_mu, log_theta_log_var, log_theta_var), axis=1)

def tf_diff(x):
    return x[:, :, 1:] - x[:, :, :-1]

def LogWeightsImportance(theta, log_theta, x_post_sample, x_obs, log_theta_mu, log_theta_log_var, log_theta_var, q_log_theta_mu, q_log_theta_log_var, q_log_theta_var, use_laplace):
    # use here means ones for log-likelihood, only precision I think
    use_log_theta = log_theta[:, :, -8:-4]
    use_theta = theta[:, :, -8:-4]

    # for overflow; not sure if still necessary
    dif = tf.clip_by_value(x_post_sample - tf.expand_dims(x_obs, 1), -100, 100)
    
    # as 1/sigma
    if use_laplace:
        log_p_x = tf.expand_dims(use_log_theta, 2) - tf.abs(dif*tf.expand_dims(use_theta, 2))
    else:
        log_p_x = tf.expand_dims(use_log_theta, 2) - 0.5*tf.square(dif*tf.expand_dims(use_theta, 2))

    log_p_x = tf.reduce_sum(log_p_x, [2, 3])
    log_p_z = log_pdf_prior(log_theta, log_theta_mu, log_theta_log_var, log_theta_var)
    log_q_z_x = log_pdf_q(log_theta, q_log_theta_mu, q_log_theta_log_var, q_log_theta_var)

    print("log_p_x = ", log_p_x)
    print("log_p_z = ", log_p_z)
    print("log_q_z_x = ", log_q_z_x)
    return log_p_x + log_p_z - log_q_z_x

def GeneralLogImportanceWeights(log_p_observations, log_p_theta, log_q_theta, beta):
    return log_p_observations +  beta*(log_p_theta - log_q_theta)  # log [p(x | theta) p(theta) / q (theta | x)]