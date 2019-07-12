# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg') # must be called before importing pylab
import pylab as pp # pylint: disable=wrong-import-position

class PandasData(object):

    def __init__(self, DATA):
        assert isinstance(DATA, pd.DataFrame)
        self.data = DATA

    def get_n_of_level(self, level_name):
        levels = self.get_levels_by_name(level_name)
        return len(levels)

    def get_levels_by_name(self, level_name):
        idx = pp.find(np.array(self.get_level_names()) == level_name)[0]
        levels = np.array(self.get_level(idx))
        return levels

    def has_level(self, level_name):
        return len(pp.find(np.array(self.get_level_names()) == level_name)) > 0

    def get_level_names(self):
        return self.data.columns.names

    def get_levels(self):
        return self.data.columns.levels

    def get_level(self, level_idx):
        return self.get_levels()[level_idx]
