# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

import os

from collections import OrderedDict
from functools import reduce
from typing import Any, Dict, List

import numpy as np
import pandas as pd

def extract_by_ids(dct, bool_query):
    d_extract = OrderedDict()
    d_extract['X'] = dct['X'][bool_query]
    d_extract['C'] = {'values': dct['C']['values'][bool_query],
                      'columns': dct['C']['columns']}
    d_extract['Observations'] = dct['Observations']
    d_extract['Time'] = dct['Time']
    return d_extract

def split_by_train_val_ids(data_dct, train_ids, val_ids):
    number_of_ids = len(train_ids) + len(val_ids)
    train_query = np.zeros(number_of_ids, dtype=bool)
    val_query = np.zeros(number_of_ids, dtype=bool)
    train_query[train_ids] = True
    val_query[val_ids] = True
    d_train = extract_by_ids(data_dct, train_query)
    d_val = extract_by_ids(data_dct, val_query)
    return d_train, d_val

def split_holdout_device(procdata, data_dct, holdout):
    c_panda = pd.DataFrame(data_dct['C']['values'],
                           columns=data_dct['C']['columns'])
    devices = c_panda['Device'].values.astype(int)
    holdout_device_id = procdata.device_map[holdout]
    val_query = devices == holdout_device_id
    train_query = devices != holdout_device_id
    print(data_dct.keys())
    d_val = extract_by_ids(data_dct, val_query)
    d_train = extract_by_ids(data_dct, train_query)
    return (d_train,
            d_val,
            np.arange(len(val_query))[train_query],
            np.arange(len(val_query))[val_query])

def process_condition(row, device):
    '''
    row: a string of the form 'a=b;c=d;...' where each RHS can be converted to float.
    Returns: a dictionary derived from row, plus the key 'Device' with value device.
    '''
    d = OrderedDict({'Device': device})
    if '=' not in row:
        return d
    conditions = row.split(';')
    for cond in conditions:
        els = cond.split('=')
        d[els[0]] = float(els[1])
    return d

def merge(dic1, dic2):
    '''
    Returns an OrderedDict whose keys are all those of dic1 and/or dic2,
    and whose values are those in dic2 or (if missing from dic2) dic1.
    '''
    return OrderedDict(dic1, **dic2)

def expand_conditions(treatments: List[OrderedDict], conditions):
    '''
    Given a list of "conds", returns a list of dicts, each of which
    is the corresponding member of "conds" expanded with "key: 0.0" members
    so that all the returned dicts have the same set of keys.
    '''
    # Establish all treatments
    zero = OrderedDict()
    for cond in conditions:
        zero[cond] = 0.0
        # Now fill each condition
    return np.array([merge(zero, tr) for tr in treatments])

def find_conditions(expanded, conditions):
    '''
    Returns the indices of expanded that only have zero values for unspecified conditions.
    '''
    treatments = list(expanded[0].keys())
    removes = list(set(treatments) - set(conditions))
    locs = [i for i,ex in enumerate(expanded) if all([ex[r]==0.0 for r in removes])]
    filtered = [OrderedDict((k, ex[k]) for k in conditions) for ex in expanded[locs]]
    return locs, filtered

def extract_signal(s):
    '''
    Returns the portion of s between the (first) pair of parentheses,
    or the whole of s if there are no parentheses.
    '''
    loc0 = s.find("(")
    if loc0 >= 0:
        loc1 = s.find(")")
        if loc1 >= 0:
            return s[loc0+1:s.find(")")]
    return s

########################################################
# Methods to enable the merging of datasets.
# Currently assumes equal timepoints, which is not ideal.
########################################################

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def merge_list(col1, col2):
    '''
    Given two sequences col1 and col2, returns a list of the
    items in them, in order, omitting any from col2 that are also
    in col1.
    '''
    cs = OrderedDict()
    for c in col1:
        cs[c] = 0.0
    for c in col2:
        cs[c] = 0.0
    return list(cs.keys())

def merge_conditions(c1, c2):
    col1 = np.array(c1['columns'])
    col2 = np.array(c2['columns'])
    v1 = c1['values']
    v2 = c2['values']
    #cs = np.array(list(set(col1).union(col2)))
    cs = merge_list(col1, col2)
    n1 = len(v1[:, 0])
    n2 = len(v2[:, 0])
    vs = np.empty((n1+n2, len(cs)))
    for i in range(len(cs)):
        if np.isin(cs[i], col1):
            vs[:, i] = np.concatenate((v1[:, np.argwhere(col1 == cs[i])[0][0]],
                                       v2[:, np.argwhere(col2 == cs[i])[0][0]]))
        else:
            vs[:, i] = np.zeros((n1 + n2, 1))
    return {'columns':cs, 'values':vs}

def merge_files(d1, d2):
    t1 = d1['Time']
    n1 = len(t1)
    t2 = d2['Time']
    n2 = len(t2)
    if n1 <= n2:
        ts = t1
        indices = list(map(lambda t: find_nearest(t2, t), t1))
        X1 = d1['X']
        X2 = list(map(lambda row: row[indices], d2['X']))
    else:
        ts = t2
        indices = list(map(lambda t: find_nearest(t1, t), t2))
        X1 = list(map(lambda row: row[indices], d1['X']))
        X2 = d2['X']

    ds = {'Time' : ts,
          'Observations' : d1['Observations'],
          'X' : np.concatenate((X1, X2)),
          'C' : merge_conditions(d1['C'], d2['C'])
         }
    return ds

def onehot(i,n):
    '''One-hot vector specifiying position i, with length n'''
    v = np.zeros((n))
    if i is not None:
        v[i] = 1
    return v

def depth(group_values):
    return len(set([g for g in group_values if g is not None]))

def apply_defaults(spec):
    ndevices = len(spec["devices"])
    params = {
        'groups': {'default': [0]*ndevices},
        'default_devices': dict(),
        'normalize': None,
        'subtract_background': True,
        'separate_conditions': False
    }
    for k in spec:
        params[k] = spec[k]
    return params
    
class ProcData:
    '''
    Class for data-handling methods that are specific to a particular type of experiment.
    Currently this is just the double-receiver experiment, but in future we may generalize
    it, either by subclassing or by handing over parameters to the constructor.
    '''
    def __init__(self, data_settings):
        # Measurable output signals: optical density and three fluorescent proteins.
        self.signals = data_settings["signals"]
        # The different devices we work with
        self.device_names = data_settings["devices"]
        self.pretty_devices = data_settings["pretty_devices"]
        self.default_devices = data_settings["default_devices"]
        # Conditions (inputs)
        self.conditions = data_settings["conditions"]
        # Files to be loaded
        self.files = data_settings["files"]
        # Group-level parameter assignments for each device
        groups_list = [ [k,v] for k, v in data_settings["groups"].items()]
        self.component_maps = OrderedDict()
        for k, group in groups_list:
            self.component_maps[k] = OrderedDict(zip(self.device_names, group)) 
        # Total number of group-level parameters        
        self.device_depth = sum([depth(cm.values()) for k, cm in self.component_maps.items()])
        # Relevance vectors for decoding multi-hot vector into multiple one-hot vectors
        self.relevance_vectors = OrderedDict()
        k1 = 0
        for k, group in groups_list:
            k2 = depth(group) + k1
            rv = np.zeros(self.device_depth)
            rv[k1:k2] = 1.0
            #print("Relevance for %s: "%k + str(rv))
            if k in self.default_devices:
                rv[k1 + self.default_devices[k]] = 0.0
            self.relevance_vectors[k] = rv.astype(np.float32)
            k1 = k2
        # Manually curated device list: map from device names to 0.0, 1.0, ...
        self.device_map = dict(zip(self.device_names, (float(v) for v in range(len(self.device_names)))))
        # Map from device indices (as ints) to device names
        self.device_idx_to_device_name = dict(enumerate(self.device_names))
        # Map from device indices (as floats) to device names
        self.device_lookup = {v: k for k, v in self.device_map.items()}
        # Separated conditions
        self.separate_conditions = data_settings["separate_conditions"]
        # Normalisation constants
        self.normalize = data_settings["normalize"]
        # Background subtraction
        self.subtract_background = data_settings["subtract_background"]
        
    def load_all(self, data_dir) -> Dict[str, Any]:
        '''
        Arguments:
          data_dir: a directory path
        Returns:
          a dict with these keys and values. When there are W conditions (wells), T time points
          and S outputs (signals), the keys and the shapes of the values are:
            'X':            numpy array of float, shape (W, T, S). Observed value for
                            every condition, time point and signal.
            'Observations': list of S signals, e.g. ['OD', 'mRFP1', 'EYFP', 'ECFP']
            'C':            dict: 'columns': list of column names, e.g. ['Device', 'C6', 'C12']
                                  'values':  numpy array of float, shape (W, 3). For col 0,
                                             Device, this is the device ID as a float (self.device_map);
                                             for cols 1 and 2 it's the level of C6 and C12. Together,
                                             these three values describe a condition.
            'Time'          numpy array of float, shape (T), hours into experiment.
        '''
        loaded = [self.load_multiple(data_dir, file, self.device_names, self.conditions) for file in self.files]
        filter_nonempty = [loaded[i] for i in range(len(loaded)) if loaded[i] is not None]
        data = reduce(merge_files, filter_nonempty)
        return data

    def get_cassettes(self, devices):
        '''
        devices: list of device indices (positions in self.device_names above)
        Returns a matrix of ones and zeros, where there are ones wherever
        the device (first index) contains the component (second index), with
        component indices taken from S components then R components.
        Each row of the matrix is a cassette.
        '''
        rows = []
        for d in devices:
            device_name = self.device_idx_to_device_name[d]
            vs = [onehot(cm[device_name], depth(cm.values())) for p, cm in self.component_maps.items()]
            rows.append(np.hstack(vs))
            #r_matrix[idx, r_value] = 1
        return np.array(rows)

    def process_row(self, row, headers):
        return [row.iloc[headers == signal].values for signal in self.signals]

    def load(self, data_dir, data_file, device, conditions):
        '''
        As for load_multiple, but with a single device
        '''
        return self.load_multiple(data_dir, data_file, [device], conditions)

    def load_multiple(self, data_dir, data_file, devices, conditions):
        '''
        data_dir, data_file: directory and basename which together point to a csv file,
          with header, consisting of:
            Content   Device, e.g. R33S32_Y81C76
            Colony    (blank)
            Well Col  int, 1 to 12
            Well Row  letter, A to H
            Content   Condition, e.g. C6=<float> or C12=<float> or EtOH=<float>
            and then for each of EYFP, ECFP, mRFP1 and OD, 100 readings at different times
          and then second line is "timesall": time of each col except for the first 5
        devices: collection of device names we're interested in
        Returns: data dictionary as described in docstring of load_all.
        '''
        data_full_path = os.path.join(data_dir, data_file)

        loaded = pd.read_csv(data_full_path, sep=',', na_filter=False)
        timesall = loaded.iloc[0, 5:] # times of the observations
        obs_rows = loaded.iloc[1:, :] # observation rows
        # Rows we want to keep are those whose first ("Content") value is in the "devices" list.
        rows = obs_rows.iloc[np.isin(obs_rows.iloc[:, 0], devices), :]

        # List of OrderedDicts, each with keys Device and either C6 or C12 (i.e. the two "content" columns above)
        # and float values.
        treatment_values = [process_condition(cond, self.device_map[dev])
                            for cond, dev in zip(rows.iloc[:, 4], rows.iloc[:, 0])]
        if len(treatment_values) is 0:
            return None  # flag value to indicate the dataset doesn't exist in this file

        # As treatment_values, but each OrderedDict additionally has the keys that the others have, with value 0.0.
        expanded = expand_conditions(treatment_values, conditions)

        # Filter out time-series that have nonzero values for unspecified conditions
        devices_conditions = ["Device"] + conditions
        locs,filtered = find_conditions(expanded, devices_conditions)
        values = np.array([list(cond.values()) for cond in filtered])

        observations = rows.iloc[locs, 5:]
        headers = np.array([v.split('.')[0] for v in observations.columns.values])
        header_signals = np.array([extract_signal(h) for h in headers])
        times = timesall.iloc[header_signals == 'OD'].values

        x_values = [self.process_row(row, header_signals) for idx, row in observations.iterrows()]

        d = {'C': {'columns': devices_conditions, 'values': values},
             'Observations': self.signals,
             # 'R': {'columns': ['Replicate'], 'values': np.expand_dims(rep, axis=1)}
             'Time': times,
             'X': np.transpose(np.array(x_values),(0, 2, 1))
            }
        return d

class Dataset(object):

    def __init__(self, data, D: OrderedDict, original_data_ids: List[int],
                 use_default_device: bool = False, default_device: str = 'Pcat_Y81C76'):
        '''
        :param D: OrderedDict with keys 'X', 'C', 'Observations' and 'Time'
        :param original_data_ids: TODO(dacart): document these
        :param conditions:
        :param use_default_device:
        :param default_device:
        '''
        self.procdata = ProcData(data)
        # Data dictionary
        self.D = D
        # List of conditions (strings)
        self.conditions = data["conditions"]
        # Number of conditions
        self.n_conditions = len(self.conditions)
        # Whether to use the default device, and what it is
        self.use_default_device = use_default_device
        self.default_device = default_device
        # Keep track of which ids from the original data this dataset comes from
        self.original_data_ids = original_data_ids
        # Numpy array of float, shape e.g. (234,86,4)
        self.X = D['X']
        c_panda = pd.DataFrame(D['C']['values'], columns=D['C']['columns'])
        # Numpy array of float, shape e.g. (234,86)
        self.C = D['C']['values'].astype(np.float32)
        # Numpy array of float, shape e.g. (234,2)
        self.treatments = c_panda[self.conditions].values.astype(np.float32)
        # print("Log + 1 transform conditions")
        self.treatments = np.log(1.0 + self.treatments)
        self.n_treatments = len(self.treatments)
        # Numpy array of float, shape e.g. (234,)
        self.devices = c_panda['Device'].values.astype(int)
        # more numpy data
        # Numpy array of float, shape e.g. (234,86,3)
        self.delta_X = np.diff(self.X, axis=-1)
        # Set up set.dev_1hot, numpy array of float, shape e.g. (234,11)
        self.dev_1hot = None
        self.prepare_devices(self.use_default_device)
        # Apply normalization or specified scales
        self.scale_data()

    #     ONE-HOT ENCODING OF DEVICE ID        #
    def prepare_devices(self, use_default_device):
        self.dev_1hot = np.zeros((self.size(), self.procdata.device_depth))
        self.dev_1hot = self.procdata.get_cassettes(self.devices)
        if use_default_device:
            print("Adding Default device: %s")
            self.dev_1hot = self.procdata.add_default_device(self.dev_1hot, self.default_device)

    def size(self):
        return len(self.X)

    def create_feed_dict(self, placeholders, u_value):
        return {placeholders.x_obs: self.X,
                placeholders.dev_1hot: self.dev_1hot,
                placeholders.conds_obs: self.treatments,
                placeholders.beta: 1.0,
                placeholders.u: u_value}

    def create_feed_dict_for_index(self, placeholders, index, beta_val, u_value):
        return {placeholders.x_obs: self.X[index],
                placeholders.dev_1hot: self.dev_1hot[index],
                placeholders.conds_obs: self.treatments[index],
                placeholders.beta: beta_val,
                placeholders.u: u_value}

    def scale_data(self):
        n_outputs = np.shape(self.X)[2]
        if self.procdata.normalize is None:
            self.scales = [np.max(self.X[:, :, i]).astype(np.float32) for i in range(n_outputs)]
        else:
            self.scales = self.procdata.normalize
        for i, scale in enumerate(self.scales):
            # First scale the data
            self.X[:, :, i] /= scale
            # mins = np.min(np.min(self.train.X[:, :, output_idx], axis=1))
            # Second, shift so smallest value for each time series is 0
            if self.procdata.subtract_background:
                self.X[:, :, i] -= np.min(self.X[:, :, i], axis=1)[:, np.newaxis]
            
class DatasetPair(object):
    '''A holder for a training and validation set and various associated parameters.'''
    # pylint: disable=too-many-instance-attributes,too-few-public-methods

    def __init__(self, train: Dataset, val: Dataset): # use_default_device=False):
        '''
        :param train: a Dataset containing the training data
        :param val: a Dataset containing the validation data
        '''
        # Dataset of the training data
        self.train = train
        # Dataset of the validation data
        self.val = val
        # Number of training instances (int)
        self.n_train = self.train.size()
        # Number of validation instances (int)
        self.n_val = self.val.size()
        # Number of species we're training on (int)
        self.n_species = self.train.X.shape[2]
        # Number of time points we're training on
        self.n_time = self.train.X.shape[1]
        # Number of group-level parameters (summed over all groups; int)
        self.depth = self.train.procdata.device_depth
        # Number of conditions we're training on
        self.n_conditions = self.train.n_conditions
        # Numpy array of time-point values (floats), length self.n_time
        self.times = self.train.D['Time'].astype(np.float32)
