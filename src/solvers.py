# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np

def modified_euler_integrate( d_states_d_t, init_state, times ):
    init_state = tf.verify_tensor_all_finite(init_state, "init_state NOT finite")
    x = [init_state]
    h = times[1]-times[0]
    F = [] 
    for t2,t1 in zip(times[1:],times[:-1]):
        
        f1 = d_states_d_t( x[-1], t1 )
        f2 = d_states_d_t( x[-1] + h*f1, t2 )

        # TODO: 
        x.append(  x[-1] + 0.5*h*(f1+f2) )
        F.append(0.5*h*(f1+f2))
    
    return tf.stack(x, axis = -1 ),tf.stack(F, axis = -1 )

import pdb
def gen_modified_euler_while_body(d_states_d_t, h, times ):
    def modified_euler_while_body( idx, x ):

        #pdb.set_trace()
        t1 = times[idx]
        t2 = times[idx+1]
        xi = x[idx]
        #pdb.set_trace()
        f1 = d_states_d_t( xi, t1 )
        f2 = d_states_d_t( xi + h*f1, t2 )
        y = tf.expand_dims( xi+0.5*h*(f1+f2), 0)
        
        return [tf.add(idx,1), tf.concat([x,y],axis=0)]
        #return [idx+1, 0.5*h*(f1+f2)]
    return modified_euler_while_body

#import pdb
def gen_modified_euler_while_conditions(T):
    def modified_euler_while_conditions( idx, x ):
        return tf.less( idx+1, T )
    return modified_euler_while_conditions

def modified_euler_integrate_while( d_states_d_t, init_state, times ):
    init_state = tf.verify_tensor_all_finite(init_state, "init_state NOT finite")
    x = tf.expand_dims( init_state, 0)  #[init_state]
    #x = init_state
    h = times[1]-times[0]
    T = len(times)
    F = [] 

    tf_times = tf.convert_to_tensor(times, np.float32)

    i0 = tf.Variable(0, trainable=False)
    n_species = x.shape[-1]
    shape_invariants = [tf.TensorShape(None), tf.TensorShape([None,None,None,n_species])] #tf.TensorShape([None,1,1,times.shape[0]])
    loop_vars = [i0, x]
    
    results = tf.while_loop( gen_modified_euler_while_conditions(T), \
                             gen_modified_euler_while_body(d_states_d_t, h, tf_times), \
                             loop_vars, \
                             shape_invariants=shape_invariants )

    x = results[-1]

    
    return x, None #tf.stack(x, axis = -1 ), None #tf.stack(F, axis = -1 )