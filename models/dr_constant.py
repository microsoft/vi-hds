# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

from models.base_model import BaseModel, log_prob_gaussian
from src.utils import default_get_value
import tensorflow as tf
import numpy as np
import pdb

class DR_Constant(BaseModel):

    def init_with_params(self, params, relevance, default_devices):
        super(DR_Constant, self).init_with_params(params, relevance, default_devices)
        # do the other inits now
        self.use_aRFP = default_get_value(params, "use_aRFP", False)
        self.species = ['OD', 'RFP', 'YFP', 'CFP', 'F510', 'F430', 'LuxR', 'LasR']

    def get_list_of_constants(self, constants):
        return [constants['init_x'],
                constants['init_rfp'],
                constants['init_yfp'],
                constants['init_cfp'],
                0.0,
                0.0,
                constants['init_luxR'],
                constants['init_lasR']]

    def gen_reaction_equations(self, theta, treatments, dev_1hot, condition_on_device=True):

        n_iwae = tf.shape(theta.r)[1]

        # tile treatments, one per iwae sample
        treatments_transformed = tf.clip_by_value(tf.exp(treatments) - 1.0, 0.0, 1e6)
        c6a, c12a = tf.unstack(treatments_transformed, axis=1)
        c6 = tf.tile(tf.expand_dims(c6a, axis=1), [1, n_iwae])
        c12 = tf.tile(tf.expand_dims(c12a, axis=1), [1, n_iwae])

        # need to clip these to avoid overflow
        r = tf.clip_by_value(theta.r, 0.0, 5.0)
        K = theta.K
        tlag = theta.tlag
        rc = theta.rc
        autoY = theta.autoY
        autoC = theta.autoC

        drfp = tf.clip_by_value(theta.drfp, 1e-12, 2.0)
        dyfp = tf.clip_by_value(theta.dyfp, 1e-12, 2.0)
        dcfp = tf.clip_by_value(theta.dcfp, 1e-12, 2.0)
        dR = tf.clip_by_value(theta.dR, 1e-12, 5.0)
        dS = tf.clip_by_value(theta.dS, 1e-12, 5.0)

        e76 = theta.e76
        e81 = theta.e81
        aCFP = theta.aCFP
        aYFP = theta.aYFP
        if self.use_aRFP:
            aRFP = theta.aRFP
        KGR_76 = theta.KGR_76
        KGS_76 = theta.KGS_76
        KGR_81 = theta.KGR_81
        KGS_81 = theta.KGS_81
        KR6 = theta.KR6
        KR12 = theta.KR12
        KS6 = theta.KS6
        KS12 = theta.KS12
        nR = tf.clip_by_value(theta.nR, 0.5, 3.0)
        nS = tf.clip_by_value(theta.nS, 0.5, 3.0)

        # condition on device information by mapping param_cond = f(param, d; \phi) where d is one-hot rep of device
        # currently, f is a one-layer MLP with NO activation function (e.g., offset and scale only)
        if condition_on_device:
            kinit = tf.keras.initializers.RandomNormal(mean=2.0, stddev=1.5)
            ones = tf.tile([[1.0]],tf.shape(theta.r))
            aR = self.device_conditioner(ones, 'aR', dev_1hot, kernel_initializer=kinit)
            aS = self.device_conditioner(ones, 'aS', dev_1hot, kernel_initializer=kinit)
        else:
            aR = theta.aR
            aS = theta.aS

        def reaction_equations(state, t):
            x, rfp, yfp, cfp, f510, f430, luxR, lasR = tf.unstack(state, axis=2)

            # Cells growing or not (not before lag-time)
            gr = r * tf.sigmoid(4.0 * (tf.cast(t, tf.float32) -  tlag))

            # Specific growth and dilution
            g = (1.0-x/K)
            gamma = gr * g

            # Promoter activity
            boundLuxR = luxR * luxR  *  ((KR6 * c6) ** nR + (KR12 * c12) ** nR) / ((1.0 + KR6 * c6 + KR12 * c12) ** nR)
            boundLasR = lasR * lasR  *  ((KS6 * c6) ** nS + (KS12 * c12) ** nS) / ((1.0 + KS6 * c6 + KS12 * c12) ** nS)
            P76 = (e76 + KGR_76 * boundLuxR + KGS_76 * boundLasR) / (1.0 + KGR_76 * boundLuxR + KGS_76 * boundLasR)
            P81 = (e81 + KGR_81 * boundLuxR + KGS_81 * boundLasR) / (1.0 + KGR_81 * boundLuxR + KGS_81 * boundLasR)

            # Check they are finite
            boundLuxR = tf.verify_tensor_all_finite(boundLuxR, "boundLuxR NOT finite")
            boundLasR = tf.verify_tensor_all_finite(boundLasR, "boundLasR NOT finite")

            # Right-hand sides
            d_x = gamma * x
            if self.use_aRFP is True:
                d_rfp = rc * aRFP - (gamma + drfp) * rfp
            else:
                d_rfp = rc - (gamma + drfp) * rfp
            d_yfp = rc * aYFP * P81 - (gamma + dyfp) * yfp
            d_cfp = rc * aCFP * P76 - (gamma + dcfp) * cfp
            d_f510 = rc * autoY - gamma * f510
            d_f430 = rc * autoC - gamma * f430
            d_luxR = rc * aR - (gamma + dR) * luxR
            d_lasR = rc * aS - (gamma + dS) * lasR

            X = tf.stack([d_x, d_rfp, d_yfp, d_cfp, d_f510, d_f430, d_luxR, d_lasR], axis=2)
            return X
        return reaction_equations, {'aR': aR, 'aS': aS}


class DR_ConstantStudentT(DR_Constant):

    def init_with_params(self, params):
        super(DR_ConstantStudentT, self).init_with_params(params)

        # use a fixed gamma prior over precisions
        self.alpha = params['precision_alpha']
        self.beta = params['precision_beta']

    def get_precision_list(self, theta):
        return self.precision_list

    def log_prob_observations(self, x_predict, x_obs, theta, x_sample):
        # expand x_obs for the iw samples in x_post_sample
        x_obs_ = tf.expand_dims(x_obs, 1)
        T = x_obs.shape[1].value

        # x_obs_.shape is [batch, 1, 86, 4] : batch, --, time, species
        # x_predict.shape is [batch, samples, time, species]
        alpha_star = self.alpha + 0.5 * T

        # sum along the time dimension
        errors = tf.reduce_sum(tf.square(x_obs_ - x_predict), 2)

        log_prob_constants = tf.lgamma(alpha_star) - tf.lgamma(self.alpha) - 0.5 * T * tf.log(2.0 * np.pi * self.beta)
        log_prob = log_prob_constants - alpha_star  *  tf.log(1.0 + (0.5 / self.beta)  *  errors)

        self.precision_modes = alpha_star / (self.beta + 0.5 * errors)
        self.precision_list = tf.unstack(self.precision_modes, axis=-1)
        # sum along the time and observed species axes
        log_prob = tf.reduce_sum(log_prob, 2)
        return log_prob
