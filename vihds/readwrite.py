# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np


def writeR(R_train, R_val, save_dir):
    np.savetxt(save_dir + "progress.txt", R_train)
    np.savetxt(save_dir + "progress_val.txt", R_val)


def loadR(save_dir):
    R = np.loadtxt("%s/progress.txt" % (save_dir))
    R_val = np.loadtxt("%s/progress_val.txt" % (save_dir))
    return [R, R_val]
