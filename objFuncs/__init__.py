__version__ = '1.0.2'
__version_descriptions__ = {
    '1.0.0':['2023-12-04',
             'Implementation of FLAME based Virtual Machine',  
             'Test lsq_linear MultiConditionalVar on FLAME VMs'],
    '1.0.1':['2023-12-14',
             'Add REA module', 
             'Detect if machine is REA by checking with caget("REA_EXP:ELMT")',
             'update print_nested_dict',
             'fix FLAME VM init x0 convert issue',],
    '1.0.2':['2023-12-19',
             'Add plot_multi_obj_history', 
             ],
}

print(f'objFuncs version: {__version__}. updated on {__version_descriptions__[__version__][0]}')



from . import construct_machineIO
_global_machineIO = construct_machineIO.construct_machineIO()

from .objFuncs import objFuncBase, objFuncGoals, objFuncMultiConditionalGoals
from . import residuals
from . import maximize_FC
from . import util

try:
    from . import flame_utils_kilean as flame_utils
except:
    print("flame_utils is not importable. ignoreing this feature...")
    
from . vritual_machine import VM
from . import REA