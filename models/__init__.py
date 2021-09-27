# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
# In this module, you must define the set of supported models.
# The YAML loader enables you to reference each model by string.

from models import (
    debug,
    auto_constant,
    degrader_constant,
    dr_constant,
    dr_blackbox,
    inducer_constant,
    prpr_constant,
    relay_constant,
)

LOOKUP = {
    "debug_constant": debug.Debug_Constant,
    "auto_constant": auto_constant.Auto_Constant,
    "auto_constant_precisions": auto_constant.Auto_Constant_Precisions,
    "degrader_constant_precisions": degrader_constant.Degrader_Constant_Precisions,
    "dr_constant": dr_constant.DR_Constant,
    "dr_constant_v2": dr_constant.DR_Constant_V2,
    "dr_constant_precisions": dr_constant.DR_Constant_Precisions,
    "dr_constant_precisions_v2": dr_constant.DR_Constant_Precisions_V2,
    "dr_blackbox": dr_blackbox.DR_Blackbox,
    "inducer_constant": inducer_constant.Inducer_Constant,
    "inducer_constant_precisions": inducer_constant.Inducer_Constant_Precisions,
    "prpr_constant": prpr_constant.PRPR_Constant,
    "prpr_constant_precisions": prpr_constant.PRPR_Constant_Precisions,
    "relay_constant": relay_constant.Relay_Constant,
    "relay_constant_precisions": relay_constant.Relay_Constant_Precisions,
}
