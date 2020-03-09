# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

import datetime
import io
import os
import re
import shutil

import yaml
import tensorflow as tf

def get_data_directory():
    """ 
    Returns directory where observation datasets are stored (default: "data") 
    """
    data_dir = os.getenv('INFERENCE_DATA_DIR')
    if data_dir:
        return data_dir
    else: 
        return "data"
    
def get_results_directory():
    """ 
    Returns mount directory of remote machine on local, where inference results are to be stored (default: "results") 
    """
    results_dir = os.getenv('INFERENCE_RESULTS_DIR')
    if results_dir:
        return results_dir
    else:
        return "results"

def is_empty(a):
    if a:
        return False
    else:
        return True

def variable_summaries(var, name, plot_histograms=False):
    """ Attach summaries to a scalar node using Tensorboard """
    #print("- Attaching tensorboard summary for %s"%name)
    mean = tf.reduce_mean(var)
    tf.summary.scalar(name+'/mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(name+'/stddev', stddev)
    tf.summary.scalar(name+'/max', tf.reduce_max(var))
    tf.summary.scalar(name+'/min', tf.reduce_min(var))
    if plot_histograms: tf.summary.histogram(name+'/histogram', var)

def make_summary_image_op(fig, tag, scope, image_format='png'):
    buf = fig_to_byte_buffer(fig, image_format=image_format)
    summary_image = tf.Summary.Image(encoded_image_string=buf.getvalue())
    return tf.Summary.Value(tag='%s/%s'%(scope,tag), image=summary_image)

def fig_to_byte_buffer(fig, image_format='png'):
    buf = io.BytesIO()
    fig.savefig(buf, format=image_format, bbox_inches='tight')
    buf.seek(0)
    return buf

def load_config_file(filename):
    if filename is None:
        return None
    with open(filename, 'r') as stream:
        return yaml.unsafe_load(stream)

def default_get_value(dct, key, default_value, verbose=False):
    if key in dct:
        return dct[key]
    if verbose:
        print("%s using default %s" % (key, str(default_value)))
    return default_value

def apply_defaults(spec):
    params = {
        'solver': 'modeulerwhile',
        'use_laplace' : False,
        'n_filters' : 10,
        'filter_size' :  10,
        'pool_size' : 5,
        'lambda_l2' : 0.001,
        'lambda_l2_hidden' : 0.001,
        'n_hidden' : 50,
        'n_hidden_decoder' : 50,
        'n_batch' : 36,
        'data_format' : 'channels_last',
        'precision_type' : 'constant',
        'precision_alpha' : 1000.0,
        'precision_beta' : 1.0,
        'init_prec' : 0.00001,
        'init_latent_species' : 0.001,
        'transfer_func' : tf.nn.tanh,
        'n_hidden_decoder_precisions' : 20,
        'n_growth_layers' : 4,
        'tb_gradients' : False,
        'plot_histograms' : False
    }
    for k in spec:
        params[k] = spec[k]
    return params

class Trainer(object):
    """Collection functions and attributes for training a Model"""
    def __init__(self, args, add_timestamp=False):
        self.results_dir = get_results_directory()
        self.experiment = args.experiment
        self.yaml_file_name = args.yaml
        self.create_logging_dirs(add_timestamp)

    def _unique_dir_name(self, experiment, add_timestamp):
        now = datetime.datetime.now().isoformat()
        time_code = re.sub('[^A-Za-z0-9]+', '', now)  # current date and time concatenated into str for logging
        if add_timestamp is True:
            experiment += "_" + time_code
        return os.path.join(self.results_dir, experiment)

    def create_logging_dirs(self, add_timestamp=False):
        self.tb_log_dir = self._unique_dir_name(self.experiment, add_timestamp)
        os.makedirs(self.tb_log_dir, exist_ok=True)
        shutil.copyfile(self.yaml_file_name,
                        os.path.join(self.tb_log_dir, os.path.basename(self.yaml_file_name)))