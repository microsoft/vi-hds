# In this module, you must define the set of supported models. 
# The YAML loader enables you to reference each model by string.

from models import debug, dr_constant, dr_growthrate, dr_blackbox

LOOKUP = {
    'debug': debug.Debug_Constant,
    'dr_constant': dr_constant.DR_Constant,
    'dr_constant_precisions': dr_constant.DR_Constant_Precisions,
    'dr_growthrate': dr_growthrate.DR_Growth,
    'dr_blackbox': dr_blackbox.DR_Blackbox,
    'dr_blackbox_precisions': dr_blackbox.DR_BlackboxPrecisions,
    'dr_hierarchical_blackbox': dr_blackbox.DR_HierarchicalBlackbox
}