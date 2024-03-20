__version__ = '1.0.3'
__version_descriptions__ = {
    '1.0.0':['2023-12-04',
             'Implementation of FLAME based Virtual Machine',  
             'Test lsq_linear MultiConditionalVar on FLAME VMs'],
    '1.0.1':['2023-12-14',
             'Add REA module', 
             'Detect if machine is REA by checking with caget("REA_EXP:ELMT")',
             'update print_nested_dict',
             'fix FLAME VM init x0 convert issue',],
    '1.0.2':['2024-03-07',
             'Add plot_multi_obj_history', #2023-12-19
             'replace pd.append (deprecated from pandas 2.0) to pd.concat in virtual_machine.py', 
             'fix bug: machineIO -> time_span=None',   #2024-03-01
             'log into jason', 
             'update for REA ', 
             'machineIO verbose',
             ],
    '1.0.3':['2024-03-11',
             'remove global machineIO',
             'record machineIO history',
             'plot machineIO history',
             ],
}

print(f'objFuncs version: {__version__}. updated on {__version_descriptions__[__version__][0]}')

from .construct_machineIO import construct_machineIO
_global_machineIO = construct_machineIO()


from .objFuncs import objFuncBase, objFuncGoals, objFuncMultiConditionalGoals, objFuncMultiConditionalVar
from . import residuals
from . import maximize_FC
from . import util

try:
    from . import flame_utils_kilean as flame_utils
except:
    print("flame_utils is not importable. ignoreing this feature...")
    
from . vritual_machine import VM
from . import REA