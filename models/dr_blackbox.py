# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

from models.base_model import BaseModel, NeuralPrecisions
from src.utils import default_get_value, variable_summaries
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import numpy as np
import pdb

class DR_Blackbox( BaseModel ):
    
    def init_with_params( self, params, procdata ):
        super(DR_Blackbox, self).init_with_params( params, procdata )
        self.species = ['OD', 'RFP', 'YFP', 'CFP']
        self.nspecies = 4
        # do the other inits now
        self.n_z = params['n_z']
        self.n_hidden = params['n_hidden_decoder']
        self.n_latent_species = params['n_latent_species']        
        self.init_latent_species = default_get_value(params, 'init_latent_species', 0.001)

    def initialize_state(self, theta, _treatments):
        n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        x0 = tf.stack([theta.init_x, theta.init_rfp, theta.init_yfp, theta.init_cfp], axis=2)
        return x0
      
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
        return reaction_equations

    def observe( self, x_sample, theta ):
        #x0 = [theta.x0, theta.rfp0, theta.yfp0, theta.cfp0]
        x_predict = [ x_sample[:,:,:,0], \
                x_sample[:,:,:,0]*x_sample[:,:,:,1], \
                x_sample[:,:,:,0]*x_sample[:,:,:,2], \
                x_sample[:,:,:,0]*x_sample[:,:,:,3]]
        x_predict = tf.stack( x_predict, axis=-1 )
        return x_predict

class DR_BlackboxStudentT( DR_Blackbox ):
    
    def init_with_params( self, params, procdata ):
        super(DR_BlackboxStudentT, self).init_with_params( params, procdata )
        
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
        
        log_prob_constants = tf.lgamma(alpha_star) - tf.lgamma(self.alpha) -0.5*T*tf.math.log(2.0*np.pi*self.beta)
        log_prob = log_prob_constants - alpha_star * tf.math.log( 1.0 + (0.5/self.beta) * errors )
            
        self.precision_modes = alpha_star / (self.beta+0.5*errors)
        self.precision_list = tf.unstack( self.precision_modes, axis=-1 )

        # sum along the time and observed species axes
        log_prob   = tf.reduce_sum( log_prob, 2 )
        return log_prob
    
class DR_BlackboxPrecisions( DR_Blackbox ):
    def init_with_params( self, params, procdata ):
        super(DR_BlackboxPrecisions, self).init_with_params( params, procdata )
        self.init_prec = params['init_prec']
        self.n_hidden_precisions = params['n_hidden_decoder_precisions']
        self.n_states = 4 + self.n_latent_species + 4

    def initialize_state(self, theta, _treatments):
        n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        zero = tf.zeros([n_batch, n_iwae])
        x0 = tf.stack([theta.init_x, theta.init_rfp, theta.init_yfp, theta.init_cfp], axis=2)
        h0 = tf.fill([n_batch, n_iwae, self.n_latent_species], self.init_latent_species)
        prec0 = tf.fill([n_batch, n_iwae, 4], self.init_prec)
        return tf.concat([x0, h0, prec0], axis=2)

    def expand_precisions_by_time( self, theta, x_predict, x_obs, x_sample ):
        var =  x_sample[:,:,:,-4:]
        prec = 1.0 / var
        log_prec = tf.math.log(prec)
        return log_prec, prec

    def initialize_neural_states(self, n):
        '''Neural states'''
        inp = Dense(self.n_hidden, activation = tf.nn.relu, name="bb_hidden", input_shape=(n,))   #activation = tf.nn.tanh
        act_layer = Dense(4+self.n_latent_species, activation = tf.nn.sigmoid, name="bb_act")
        deg_layer = Dense(4+self.n_latent_species, activation = tf.nn.sigmoid, name="bb_deg")
        act = Sequential([inp, act_layer])
        deg = Sequential([inp, deg_layer])
        for layer in [inp, act_layer, deg_layer]:
            weights, bias = layer.weights
            variable_summaries(weights, layer.name + "_kernel", False)
            variable_summaries(bias, layer.name + "_bias", False)
        return act, deg

    def gen_reaction_equations( self, theta, treatments, dev_1hot, condition_on_device=True ):
        n_iwae = theta.get_n_samples()
        n_batch = theta.get_n_batch()
        devices = tf.tile( dev_1hot, [n_iwae, 1] )
        treatments_rep = tf.tile( treatments, [n_iwae,1])

        Z = []
        for i in range(1,self.n_z+1):
            Z.append( getattr(theta,"z%d"%i))
        Z = tf.stack( Z, axis=2 )

        n = 4 + self.n_latent_species + self.n_z + self.n_treatments + self.device_depth
        states_act, states_deg = self.initialize_neural_states(n)
        neural_precisions = NeuralPrecisions(self.nspecies, self.n_hidden_precisions, 
            inputs = self.n_states + self.n_z + self.n_treatments + self.device_depth, 
            hidden_activation = tf.nn.relu)

        def reaction_equations( state, t ):
            all_reshaped_state = tf.reshape( state, [n_batch*n_iwae, self.n_states])

            # split for precisions and states
            reshaped_state = all_reshaped_state[:,:-4]
            reshaped_var_state = all_reshaped_state[:,-4:]
            
            ZZ_states = tf.concat( [ reshaped_state, \
                        tf.reshape( Z, [n_batch*n_iwae, self.n_z]), \
                        treatments_rep,\
                        devices], axis=1 )            
            states    = states_act(ZZ_states) - states_deg(ZZ_states)*reshaped_state

            ZZ_vrs = tf.concat( [ all_reshaped_state, \
                        tf.reshape( Z, [n_batch*n_iwae, self.n_z]), \
                        treatments_rep,\
                        devices], axis=1 )
            vrs = neural_precisions.act(ZZ_vrs) - neural_precisions.deg(ZZ_vrs)*reshaped_var_state

            return tf.reshape( tf.concat( [states,vrs],1),  [n_batch, n_iwae, self.n_states] )
        return reaction_equations

class DR_HierarchicalBlackbox( DR_BlackboxPrecisions ):
    
    def init_with_params( self, params, procdata ):
        super(DR_HierarchicalBlackbox, self).init_with_params( params, procdata )
        # do the other inits now
        self.n_x = params['n_x']
        self.n_y = params['n_y']
        self.n_z = params['n_z']
        self.n_latent_species = params['n_latent_species']
        self.n_hidden_species = params['n_hidden_decoder']
        self.n_hidden_precisions = params['n_hidden_decoder_precisions']
        self.init_latent_species = default_get_value(params, 'init_latent_species', 0.001)
        self.init_prec = default_get_value(params, 'init_prec', 0.00001)
        
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
                Y.append( getattr(theta, nm ) )
            Y = tf.stack( Y, axis=2 )
            Y_reshaped = tf.reshape( Y, [n_batch*n_iwae, self.n_y])
            offset_layer = Dense(self.n_y, activation=None, name="device_offsets")
            Y_reshaped = Y_reshaped + offset_layer( devices )
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
        n_latents = self.n_x + self.n_y + self.n_z

        # Neural components initialization
        n = 4 + self.n_latent_species + n_latents + self.n_treatments + self.device_depth
        states_act, states_deg = self.initialize_neural_states(n)
        neural_precisions = NeuralPrecisions(self.nspecies, self.n_hidden_precisions, 
            inputs = self.n_states + self.n_x + self.n_y + self.n_z + self.n_treatments + self.device_depth, 
            hidden_activation = tf.nn.relu)

        def reaction_equations( state, t ):
            all_reshaped_state = tf.reshape( state, [n_batch*n_iwae, self.n_states])

            # States
            reshaped_state = all_reshaped_state[:,:-4]
            ZZ_states = tf.concat( [ reshaped_state, \
                        tf.reshape( latents, [n_batch*n_iwae, n_latents]), \
                        treatments_rep,\
                        devices], axis=1 )
            states = states_act(ZZ_states) - states_deg(ZZ_states)*reshaped_state

            # Precisions
            reshaped_var_state = all_reshaped_state[:,-4:]
            ZZ_vrs = tf.concat( [ all_reshaped_state, \
                        tf.reshape( latents, [n_batch*n_iwae, n_latents]), \
                        treatments_rep,\
                        devices], axis=1 )            
            vrs = neural_precisions.act(ZZ_vrs) - neural_precisions.deg(ZZ_vrs)*reshaped_var_state

            return tf.reshape( tf.concat( [states,vrs],1),  [n_batch, n_iwae, self.n_states] )
        return reaction_equations