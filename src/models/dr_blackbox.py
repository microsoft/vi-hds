# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from models.base_model import BaseModel
from utils import default_get_value
import tensorflow as tf
import numpy as np
import pdb

class DR_Blackbox( BaseModel ):
    
    def init_with_params( self, params ):
        super(DR_Blackbox, self).init_with_params( params )
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
    
    def init_with_params( self, params ):
        super(DR_BlackboxStudentT, self).init_with_params( params )
        
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
    def init_with_params( self, params ):
        super(DR_BlackboxPrecisions, self).init_with_params( params )
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

    # def initialize_state( self, theta, constants ):
    #     constants_tensors = tf.expand_dims( tf.constant( self.get_list_of_constants(constants), dtype=tf.float32 ), 0 )
    

    #     init_prec = 1.0 / tf.nn.softplus( tf.stack( [theta.init_prec_x, theta.init_prec_rfp, theta.init_prec_yfp, theta.init_prec_cfp], axis=2 ) )


    #     n_constants = constants_tensors.shape[1].value
    #     n_batch     = theta.get_n_batch() 
    #     n_iwae      = theta.get_n_samples() 
    
    #     init_constants = tf.reshape( tf.tile( constants_tensors, [n_batch*n_iwae,1] ), (n_batch,n_iwae,n_constants) )
        
    #     init_state = tf.concat( [init_constants, init_prec], axis=-1 )

    #     #pdb.set_trace()
    #     return init_state
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
            n_hidden = self.n_hidden
            all_reshaped_state = tf.reshape( state, [n_batch*n_iwae, n_states])

            # split for precisions and states
            reshaped_state = all_reshaped_state[:,:-4]
            reshaped_var_state = all_reshaped_state[:,-4:]
            ZZ_states = tf.concat( [ reshaped_state, \
                        tf.reshape( Z, [n_batch*n_iwae, n_z]), \
                        treatments_rep,\
                        devices], axis=1 )            
            layer1_states = tf.layers.dense( ZZ_states, units=n_hidden, activation = tf.nn.tanh, name="bb_hidden",reuse=tf.AUTO_REUSE )
            layer2_states = tf.layers.dense( layer1_states, units=4+self.n_latent_species, activation = tf.nn.sigmoid, name="bb_df_act",reuse=tf.AUTO_REUSE )
            layer3_states = tf.layers.dense( layer1_states, units=4+self.n_latent_species, activation = tf.nn.sigmoid, name="bb_df_deg",reuse=tf.AUTO_REUSE )

            ZZ_vrs = tf.concat( [ all_reshaped_state, \
                        tf.reshape( Z, [n_batch*n_iwae, n_z]), \
                        treatments_rep,\
                        devices], axis=1 )            
            layer1_vrs = tf.layers.dense( ZZ_vrs, units=25, activation = tf.nn.tanh, name="bb_hidden_vrs",reuse=tf.AUTO_REUSE )
            layer2_vrs = tf.layers.dense( layer1_vrs, units=4, activation = tf.nn.sigmoid, name="bb_df_act_vrs",reuse=tf.AUTO_REUSE )
            layer3_vrs = tf.layers.dense( layer1_vrs, units=4, activation = tf.nn.sigmoid, name="bb_df_deg_vrs",reuse=tf.AUTO_REUSE )

            states = layer2_states-layer3_states*reshaped_state
            vrs    = layer2_vrs-layer3_vrs*reshaped_var_state

            return tf.reshape( tf.concat( [states,vrs],1),  [n_batch, n_iwae, n_states] )
        
        # Return equations and an empty dict, which can otherwise be populated with conditioned parameter values for TB visualization
        return reaction_equations, {}