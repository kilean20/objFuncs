# Overview

The objFuncs package is designed to provide users with an intuitive I/O interface to the EPICS system, specifically tailored for the Facility for Rare Isotope Beams (FRIB) driver LINAC and REA. It streamlines the construction of objective functions by scalarizing multi-objective optimization using heuristics that incorporate tolerances for each objective. Additionally, it includes wrapper functions for evaluating these objective functions on the machine.

## Key Features

1. **Easy I/O Interface**:
   - `objFuncs/construct_machineIO.py`
   - Simplifies interaction with the FRIB machine.
   - Provides streamlined methods for data input and output.

3. **Objective Function Construction**:
   - Facilitates quick and easy creation of objective functions for the EPICS system.
   - Uses scalarization techniques to handle multi-objective optimization based on simple heuristics regarding tolerance levels for different objectives.

4. **Wrapper Functions**:
   - Includes wrapper functions to facilitate the evaluation of objective functions directly on the EPICS system to be used in general optimization routine.

## Example Usage

```python

import objFuncs
from objFuncs import objFuncGoals

objFuncs._global_machineIO._test = False
objFuncs._global_machineIO._fetch_data_time_span = timespan_for_average
objFuncs._global_machineIO._ensure_set_timewait_after_ramp = additional_wait_after_powersupply_ramp

obj = objFuncGoals(
    decision_CSETs=['FE_LEBT:PSC2_D0001:I_CSET','FE_LEBT:PSC1_D001:I_CSET',
    decision_min = [-2,-2],
    decision_max = [ 2, 2],
    decision_tols = [0.1, 0.1],
    objective_goal   = {'FE_LEBT:FC_D0002:PKAVG_RD': {'more than': sourceFC}},
    objective_weight = {'FE_LEBT:FC_D0002:PKAVG_RD': 1},
    objective_norm   = {'FE_LEBT:FC_D0002:PKAVG_RD': 0.2*sourceFC},
)

# set corrector to zero and read objective
x = [0, 0]
obj_val = obj(x)
```
