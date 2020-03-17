import tensorflow as tf
from models.base_model import BaseModel

class Debug_Constant(BaseModel):
    def init_with_params(self, params, procdata):
        super(Debug_Constant, self).init_with_params(params, procdata)
        self.species = ['OD', 'RFP', 'YFP', 'CFP']
        self.n_species = len(self.species)

    def initialize_state(self, theta, _treatments):
        n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        x0 = tf.concat([tf.reshape(theta.init_x, [n_batch,n_iwae,1]), tf.zeros([n_batch, n_iwae, self.n_species-1])], 2)
        #x0 = tf.concat([tf.fill([n_batch, n_iwae, 1], 0.002), tf.zeros([n_batch, n_iwae, 7])], 2)        
        return x0
    
    def observe(cls, x_sample, _theta):
        x_predict = [
            x_sample[:, :, :, 0],
            x_sample[:, :, :, 0] * x_sample[:, :, :, 1],
            x_sample[:, :, :, 0] * x_sample[:, :, :, 2],
            x_sample[:, :, :, 0] * x_sample[:, :, :, 3]]
        x_predict = tf.stack(x_predict, axis=-1)
        return x_predict

    def gen_reaction_equations(self, theta, treatments, dev_1hot, condition_on_device=True):

        n_iwae = theta.get_n_samples()
        r = tf.clip_by_value(theta.r, 0.1, 2.0)
        
        def reaction_equations(state, t):
            state = tf.verify_tensor_all_finite(state, "state NOT finite")
            x, rfp, yfp, cfp = tf.unstack(state, axis=2)
            x = tf.verify_tensor_all_finite(x, "x NOT finite")
            rfp = tf.verify_tensor_all_finite(rfp, "rfp NOT finite")
            yfp = tf.verify_tensor_all_finite(yfp, "yfp NOT finite")
            cfp = tf.verify_tensor_all_finite(cfp, "cfp NOT finite")

            gamma = r * (1.0 - x)
            gamma = tf.verify_tensor_all_finite(gamma, "gamma NOT finite")
            # Right-hand sides
            d_x = x * gamma
            #d_x = tf.verify_tensor_all_finite(d_x, "d_x NOT finite")
            d_rfp = 1.0 - (gamma + 1.0) * rfp
            d_rfp = tf.verify_tensor_all_finite(d_rfp, "d_rfp NOT finite")
            d_yfp = 1.0 - (gamma + 1.0) * yfp
            d_yfp = tf.verify_tensor_all_finite(d_yfp, "d_yfp NOT finite")
            d_cfp = 1.0 - (gamma + 1.0) * cfp
            d_cfp = tf.verify_tensor_all_finite(d_cfp, "d_cfp NOT finite")

            X = tf.stack([d_x, d_rfp, d_yfp, d_cfp], axis=2)
            X = tf.verify_tensor_all_finite(X, "RHS NOT finite")
            return X
        return reaction_equations