# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

from models.base_model import BaseModel
from src.utils import default_get_value
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import numpy as np
import pdb

class DR_Blackbox( BaseModel ):
    
    def init_with_params( self, params, relevance, default_devices ):
        super(DR_Blackbox, self).init_with_params( params, relevance, default_devices )
        # do the other inits now
        self.n_z = params['n_z']
        self.n_hidden = params['n_hidden_decoder']
        self.n_latent_species = params['n_latent_species']        
        self.init_latent_species = default_get_value(params, 'init_latent_species', 0.001)
        self.latent_species_constants = [self.init_latent_species for i in range( self.n_latent_species)]

    def get_list_of_constants(self, constants):
        return [ constants['init_x'], constants['init_rfp'],\
                                      constants['init_yfp'],\
                                      constants['init_cfp'] ] + self.latent_species_constants
      
    def get_precision_list(self, theta ):
        return [theta.prec_x, theta.prec_rfp, theta.prec_yfp, theta.prec_cfp]

    def gen_reaction_equations( self, theta, treatments, dev_1hot, condition_on_device=True ):
        n_iwae = tf.shape( theta.z1 )[1]  
        n_batch = tf.shape( theta.z1 )[0]
        devices = tf.tile( dev_1hot, [n_iwae, 1] )
        treatments_rep = tf.tile( treatments, [n_iwae,1])

        Z = []
        for i in range(1,self.n_z+1):
            Z.append( getattr(theta,"z%d"%i))
        Z = tf.stack( Z, axis=2 )

        def reaction_equations( state, t ):
            n_states  = state.shape[-1].value
            n_z       = Z.shape[-1].value

            reshaped_state = tf.reshape( state, [n_batch*n_iwae, n_states])
            ZZ = tf.concat( [ reshaped_state, \
                        tf.reshape( Z, [n_batch*n_iwae, n_z]), \
                        treatments_rep,\
                        devices], axis=1 )

            n_hidden = self.n_hidden
            layer1 = tf.layers.dense( ZZ, units=n_hidden, activation = tf.nn.tanh, name="bb_hidden",reuse=tf.AUTO_REUSE )
            layer2 = tf.layers.dense( layer1, units=n_states, activation = tf.nn.sigmoid, name="bb_df_act",reuse=tf.AUTO_REUSE )
            layer3 = tf.layers.dense( layer1, units=n_states, activation = tf.nn.sigmoid, name="bb_df_deg",reuse=tf.AUTO_REUSE )

            return tf.reshape( layer2-layer3*reshaped_state,  [n_batch, n_iwae, n_states] )
        
        # Return equations and an empty dict, which can otherwise be populated with conditioned parameter values for TB visualization
        return reaction_equations, {}
        


    def observe( self, x_sample, theta, constants ):
        #x0 = [theta.x0, theta.rfp0, theta.yfp0, theta.cfp0]
        x_predict = [ x_sample[:,:,:,0], \
                x_sample[:,:,:,0]*x_sample[:,:,:,1], \
                x_sample[:,:,:,0]*x_sample[:,:,:,2], \
                x_sample[:,:,:,0]*x_sample[:,:,:,3]]
        x_predict = tf.stack( x_predict, axis=-1 )
        return x_predict

class DR_BlackboxStudentT( DR_Blackbox ):
    
    def init_with_params( self, params, relevance, default_devices ):
        super(DR_BlackboxStudentT, self).init_with_params( params, relevance, default_devices )
        
        # use a fixed gamma prior over precisions
        self.alpha = params['precision_alpha']
        self.beta = params['precision_beta']
      
    def get_precision_list(self, theta ):
        return self.precision_list
    
    def log_prob_observations( self, x_predict, x_obs, theta, x_sample ):
        #log_precisions, precisions     = self.expand_precisions( self.get_precision_list( theta ) )
        # expand x_obs for the iw samples in x_post_sample
        x_obs_ = tf.expand_dims( x_obs, 1 )
        T = x_obs.shape[1].value
        
        # x_obs_.shape is [batch, 1, 86, 4] : batch, --, time, species
        # x_predict.shape is [batch, samples, time, species]
        alpha_star = self.alpha + 0.5*T
        
        # sum along the time dimension
        errors = tf.reduce_sum( tf.square( x_obs_ - x_predict ), 2 )
        
        log_prob_constants = tf.lgamma(alpha_star) - tf.lgamma(self.alpha) -0.5*T*tf.log(2.0*np.pi*self.beta)
        log_prob = log_prob_constants - alpha_star * tf.log( 1.0 + (0.5/self.beta) * errors )
            
        self.precision_modes = alpha_star / (self.beta+0.5*errors)
        self.precision_list = tf.unstack( self.precision_modes, axis=-1 )

        # sum along the time and observed species axes
        log_prob   = tf.reduce_sum( log_prob, 2 )
        return log_prob
    
class DR_BlackboxPrecisions( DR_Blackbox ):
    def init_with_params( self, params, relevance, default_devices ):
        super(DR_BlackboxPrecisions, self).init_with_params( params, relevance, default_devices )
        self.init_prec = params['init_prec']
        self.prec_constants = [self.init_prec for i in range(4)]

    def get_list_of_constants(self, constants):
        return [ constants['init_x'], constants['init_rfp'],\
                                      constants['init_yfp'],\
                                      constants['init_cfp'] ] + self.latent_species_constants + self.prec_constants
    
    def expand_precisions_by_time( self, theta, x_predict, x_obs, x_sample ):
        var =  x_sample[:,:,:,-4:]
        prec = 1.0 / var
        log_prec = tf.log(prec)
        return log_prec, prec
        
    def gen_reaction_equations( self, theta, treatments, dev_1hot, condition_on_device=True ):
        n_iwae = tf.shape( theta.z1 )[1]  
        n_batch = tf.shape( theta.z1 )[0]
        devices = tf.tile( dev_1hot, [n_iwae, 1] )
        n_devices = tf.shape( dev_1hot )[1]
        treatments_rep = tf.tile( treatments, [n_iwae,1])
        n_treatments = tf.shape( treatments )[1]

        Z = []
        for i in range(1,self.n_z+1):
            Z.append( getattr(theta,"z%d"%i))
        Z = tf.stack( Z, axis=2 )

        def reaction_equations( state, t ):
            n_states  = state.shape[-1].value
            n_z       = Z.shape[-1].value
            all_reshaped_state = tf.reshape( state, [n_batch*n_iwae, n_states])

            # split for precisions and states
            reshaped_state = all_reshaped_state[:,:-4]
            reshaped_var_state = all_reshaped_state[:,-4:]
            
            ZZ_states = tf.concat( [ reshaped_state, \
                        tf.reshape( Z, [n_batch*n_iwae, n_z]), \
                        treatments_rep,\
                        devices], axis=1 )            
            states_inp = Dense(self.n_hidden, activation = tf.nn.relu, name="bb_hidden")   #activation = tf.nn.tanh
            states_act_layer = Dense(4+self.n_latent_species, activation = tf.nn.sigmoid, name="bb_df_act")
            states_deg_layer = Dense(4+self.n_latent_species, activation = tf.nn.sigmoid, name="bb_df_deg")
            states_act = Sequential([states_inp, states_act_layer])(ZZ_states)
            states_deg = Sequential([states_inp, states_deg_layer])(ZZ_states)            
            states    = states_act - states_deg*reshaped_state

            ZZ_vrs = tf.concat( [ all_reshaped_state, \
                        tf.reshape( Z, [n_batch*n_iwae, n_z]), \
                        treatments_rep,\
                        devices], axis=1 )
            vars_inp = Dense(25, activation = tf.nn.relu, name="bb_hidden_vrs")   #activation = tf.nn.tanh
            vars_act_layer = Dense(4, activation = tf.nn.sigmoid, name="bb_df_act_vrs")
            vars_deg_layer = Dense(4, activation = tf.nn.sigmoid, name="bb_df_deg_vrs")
            vars_act = Sequential([vars_inp, vars_act_layer])(ZZ_vrs)
            vars_deg = Sequential([vars_inp, vars_deg_layer])(ZZ_vrs)            
            vrs    = vars_act - vars_deg*reshaped_var_state
            
            return tf.reshape( tf.concat( [states,vrs],1),  [n_batch, n_iwae, n_states] )
        
        # Return equations and an empty dict, which can otherwise be populated with conditioned parameter values for TB visualization
        return reaction_equations, {}