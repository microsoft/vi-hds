# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from models.base_model import BaseModel
from utils import default_get_value
import tensorflow as tf
import numpy as np
import pdb

class DR_HierarchicalBlackbox( BaseModel ):
    
    def init_with_params( self, params ):
        super(DR_HierarchicalBlackbox, self).init_with_params( params )
        # do the other inits now
        self.n_x = params['n_x']
        self.n_y = params['n_y']
        self.n_z = params['n_z']
        self.n_latent_species = params['n_latent_species']
        self.n_hidden_species = params['n_hidden_decoder_species']
        self.n_hidden_precisions = params['n_hidden_decoder_precisions']        
        self.init_latent_species = default_get_value(params, 'init_latent_species', 0.001)
        self.latent_species_constants = [self.init_latent_species for i in range( self.n_latent_species)]
        self.init_prec = default_get_value(params, 'init_prec', 0.00001)
        self.prec_constants = [self.init_prec for i in range(4)]
      
    def get_precision_list(self, theta ):
        return [theta.prec_x, theta.prec_rfp, theta.prec_yfp, theta.prec_cfp]

    def observe( self, x_sample, theta, constants ):
        #x0 = [theta.x0, theta.rfp0, theta.yfp0, theta.cfp0]
        x_predict = [ x_sample[:,:,:,0], \
                x_sample[:,:,:,0]*x_sample[:,:,:,1], \
                x_sample[:,:,:,0]*x_sample[:,:,:,2], \
                x_sample[:,:,:,0]*x_sample[:,:,:,3]]
        x_predict = tf.stack( x_predict, axis=-1 )
        return x_predict

    def get_list_of_constants(self, constants):
        return [ constants['init_x'], constants['init_rfp'],\
                                      constants['init_yfp'],\
                                      constants['init_cfp'] ] + self.latent_species_constants + self.prec_constants
    
    def expand_precisions_by_time( self, theta, x_predict, x_obs, x_sample ):
        var =  x_sample[:,:,:,-4:]
        prec = 1.0 / var
        log_prec = tf.log(prec)
        return log_prec, prec

    def device_conditioner( self, param, param_name, dev_1hot, use_bias=False, activation=None ):  # TODO: try e.g. activation=tf.nn.relu
        """Returns a 1D parameter conditioned on device
        ::NOTE:: condition_on_device is a closure over n_iwae, n_batch, dev_1hot_rep"""
        n_iwae = tf.shape( param )[1]  
        n_batch = tf.shape( param )[0]
        # tile devices, one per iwae sample
        dev_1hot_rep = tf.tile( dev_1hot, [n_iwae, 1] )
        param_flat = tf.reshape( param, [n_iwae*n_batch, 1] )
        #param_cond_inp = dev_1hot_rep #tf.concat( [param_flat, dev_1hot_rep], axis=1 )
        param_cond = tf.layers.dense( dev_1hot_rep, units = 1, use_bias = use_bias,
                        activation=activation,
                        #kernel_initializer=tf.initializers.orthogonal(gain=1.0), 
                        name = '%s_%s'%(param_name, 'decoder'))
                
        return tf.reshape( param_flat*tf.exp(param_cond), [n_batch, n_iwae] )

    def gen_reaction_equations( self, theta, treatments, dev_1hot, condition_on_device=True ):
        n_iwae = tf.shape( theta.z1 )[1]  
        n_batch = tf.shape( theta.z1 )[0]
        devices = tf.tile( dev_1hot, [n_iwae, 1] )

        treatments_rep = tf.tile( treatments, [n_iwae,1])

        # locals
        Z = []
        if self.n_z > 0:
            for i in range(1,self.n_z+1):
                Z.append( getattr(theta,"z%d"%i))
            Z = tf.stack( Z, axis=2 )

        # global conditionals
        Y = []
        if self.n_y > 0:
            for i in range(1,self.n_y+1):
                nm = "y%d"%i
                #Y.append( self.device_conditioner( getattr(theta, nm ), nm, dev_1hot )
                Y.append( getattr(theta, nm ) )
            Y = tf.stack( Y, axis=2 )
            Y_reshaped = tf.reshape( Y, [n_batch*n_iwae, self.n_y])
            Y_reshaped = Y_reshaped + tf.layers.dense( devices, units=self.n_y, activation = None, name="device_offsets",reuse=tf.AUTO_REUSE )
            Y = tf.reshape( Y_reshaped, [n_batch, n_iwae, self.n_y] )

        # globals
        X = []
        if self.n_x > 0:
            for i in range(1,self.n_x+1):
                X.append( getattr(theta,"x%d"%i))
            X = tf.stack( X, axis=2 )

        if self.n_z > 0 and self.n_y == 0 and self.n_x == 0:
            print("Black Box case: LOCALS only")
            latents = Z
        elif  self.n_z == 0 and self.n_y > 0 and self.n_x == 0:   
            print("Black Box case: GLOBAL CONDITIONS only")
            latents = Y
        elif  self.n_z == 0 and self.n_y == 0 and self.n_x > 0: 
            print("Black Box case: GLOBALS only")  
            latents = X
        elif self.n_z > 0 and self.n_y > 0 and self.n_x == 0:
            print("Black Box case: LOCALS and GLOBAL CONDITIONS only")  
            latents = tf.concat( [Y,Z], axis=-1 )
        elif self.n_z > 0 and self.n_y == 0 and self.n_x > 0:
            print("Black Box case: LOCALS and GLOBALS only")  
            latents = tf.concat( [X,Z], axis=-1 )
        elif self.n_z == 0 and self.n_y > 0 and self.n_x > 0:
            print("Black Box case: GLOBALS and GLOBAL CONDITIONS only")  
            latents = tf.concat( [X,Y], axis=-1 )
        elif self.n_z > 0 and self.n_y > 0 and self.n_x > 0:
            print("Black Box case: LOCALS & GLOBALS & GLOBAL CONDITIONS") 
            latents = tf.concat( [X,Y,Z], axis=-1 )
        else:
            raise Exception("must assign latents")

        def reaction_equations( state, t ):
            n_states  = state.shape[-1].value
            n_latents       = latents.shape[-1].value
            all_reshaped_state = tf.reshape( state, [n_batch*n_iwae, n_states])

            # split for precisions and states
            reshaped_state = all_reshaped_state[:,:-4]
            reshaped_var_state = all_reshaped_state[:,-4:]

            ZZ_states = tf.concat( [ reshaped_state, \
                        tf.reshape( latents, [n_batch*n_iwae, n_latents]), \
                        treatments_rep,\
                        devices], axis=1 )            
            layer1_states = tf.layers.dense( ZZ_states, units=self.n_hidden_species, activation = tf.nn.tanh, name="bb_hidden",reuse=tf.AUTO_REUSE )
            layer2_states = tf.layers.dense( layer1_states, units=4+self.n_latent_species, activation = tf.nn.sigmoid, name="bb_df_act",reuse=tf.AUTO_REUSE )
            layer3_states = tf.layers.dense( layer1_states, units=4+self.n_latent_species, activation = tf.nn.sigmoid, name="bb_df_deg",reuse=tf.AUTO_REUSE )
            
            ZZ_vrs = tf.concat( [ all_reshaped_state, \
                        tf.reshape( latents, [n_batch*n_iwae, n_latents]), \
                        treatments_rep,\
                        devices], axis=1 )            
            layer1_vrs = tf.layers.dense( ZZ_vrs, units=self.n_hidden_precisions, activation = tf.nn.tanh, name="bb_hidden_vrs",reuse=tf.AUTO_REUSE )
            layer2_vrs = tf.layers.dense( layer1_vrs, units=4, activation = tf.nn.sigmoid, name="bb_df_act_vrs",reuse=tf.AUTO_REUSE )
            layer3_vrs = tf.layers.dense( layer1_vrs, units=4, activation = tf.nn.sigmoid, name="bb_df_deg_vrs",reuse=tf.AUTO_REUSE )

            states = layer2_states-layer3_states*reshaped_state
            vrs    = layer2_vrs-layer3_vrs*reshaped_var_state

            return tf.reshape( tf.concat( [states,vrs],1),  [n_batch, n_iwae, n_states] )
        
        # Return equations and an empty dict, which can otherwise be populated with conditioned parameter values for TB visualization
        return reaction_equations, {}