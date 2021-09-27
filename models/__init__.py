# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
# In this module, you must define the set of supported models. 
# The YAML loader enables you to reference each model by string.

from models import ( debug, dr_constant, dr_blackbox )

LOOKUP = {
    'debug_constant': debug.Debug_Constant,
    'dr_constant': dr_constant.DR_Constant,
    'dr_constant_v2': dr_constant.DR_Constant_V2,
    'dr_constant_precisions': dr_constant.DR_Constant_Precisions,
    'dr_constant_precisions_v2': dr_constant.DR_Constant_Precisions_V2,
    'dr_blackbox': dr_blackbox.DR_Blackbox,
}