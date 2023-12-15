import numpy as np
from typing import List, Dict, Union, Optional

from .objFuncs import objFuncGoals
from .util import warn


class maximize_FC(objFuncGoals):
    def __init__(self,
        decision_CSETs,
        decision_min,
        decision_max,
        objective_goal,
        objective_weight,
        objective_norm,
        objective_fill_none_by_init: Optional[bool] = False,
        objective_p_order:Optional[float] = 2,
        apply_bilog:Optional[bool] = False,
                 
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = np.inf,
        logging_tag: Optional[str] = None,
        logging_fname: Optional[str] = None,
        init_verbose: Optional[bool] = True,
        ):
        super().__init__(
            decision_CSETs=decision_CSETs,
            decision_min=decision_min,
            decision_max=decision_max,
            objective_goal=objective_goal, 
            objective_weight=objective_weight,
            objective_norm=objective_norm,
            objective_fill_none_by_init=objective_fill_none_by_init,
            objective_p_order=objective_p_order,
            apply_bilog=apply_bilog,
            decision_couplings=decision_couplings,
            decision_RDs=decision_RDs,
            decision_tols=decision_tols,
            history_buffer_size=history_buffer_size,
            logging_frequency=logging_frequency,
            logging_tag=logging_tag,
            logging_fname=logging_fname,
            init_verbose=init_verbose,
            )
    def _check_device_init(self):
        if self.machineIO._test:
            return
        if self._caget("REA_BTS34:FC_D1448:LMIN_RSTS") != 1:
            _RuntimeError("FC_D1448 is not in.")

    
class maximize_FC1058(maximize_FC):
    def __init__(self,
        decision_CSETs = ['REA_BTS19:DCVE_D0970:V_CSET',
                          'REA_BTS19:DCHE_D0979:V_CSET',
                          'REA_BTS19:DCHE_D0987:V_CSET',
                          'REA_BTS19:DCVE_D0987:V_CSET',
                          ],
        decision_min = -40,
        decision_max =  40,
        objective_goal   = {'REA_WK01:FC_D1058:BC_RD': {'more than': None}},
        objective_weight = {'REA_WK01:FC_D1058:BC_RD': 1},
        objective_norm   = {'REA_WK01:FC_D1058:BC_RD': 1e-12},
        objective_fill_none_by_init: Optional[bool] = False,
        objective_p_order:Optional[float] = 2,
        apply_bilog:Optional[bool] = False,
                 
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = np.inf,
        logging_tag: Optional[str] = None,
        logging_fname: Optional[str] = None,
        init_verbose: Optional[bool] = True,        
        ):
        super().__init__(
            decision_CSETs=decision_CSETs,
            decision_min=decision_min,
            decision_max=decision_max,
            objective_goal=objective_goal, 
            objective_weight=objective_weight,
            objective_norm=objective_norm,
            objective_fill_none_by_init=objective_fill_none_by_init,
            objective_p_order=objective_p_order,
            apply_bilog=apply_bilog,
            decision_couplings=decision_couplings,
            decision_RDs=decision_RDs,
            decision_tols=decision_tols,
            history_buffer_size=history_buffer_size,
            logging_frequency=logging_frequency,
            logging_tag=logging_tag,
            logging_fname=logging_fname,
            init_verbose=init_verbose,
            )
    def _check_device_init(self):
        super()._check_device_init()

            
            
class maximize_FC1164(maximize_FC):
    def __init__(self,
        decision_CSETs = ['REA_CM01:DCH_D1123:I_CSET',
                          'REA_CM01:DCV_D1123:I_CSET',
                          'REA_CM01:DCH_D1139:I_CSET',
                          'REA_CM01:DCV_D1139:I_CSET',
                          ],
        decision_min = -1.5,
        decision_max =  1.5,
        objective_goal   = {'REA_BTS24:FC_D1164:BC_RD': {'more than': None}},
        objective_weight = {'REA_BTS24:FC_D1164:BC_RD': 1},
        objective_norm   = {'REA_BTS24:FC_D1164:BC_RD': 1e-12},
        objective_fill_none_by_init: Optional[bool] = False,
        objective_p_order:Optional[float] = 2,
        apply_bilog:Optional[bool] = False,
                 
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = np.inf,
        logging_tag: Optional[str] = None,
        logging_fname: Optional[str] = None,
        init_verbose: Optional[bool] = True,        
        ):
        super().__init__(
            decision_CSETs=decision_CSETs,
            decision_min=decision_min,
            decision_max=decision_max,
            objective_goal=objective_goal, 
            objective_weight=objective_weight,
            objective_norm=objective_norm,
            objective_fill_none_by_init=objective_fill_none_by_init,
            objective_p_order=objective_p_order,
            apply_bilog=apply_bilog,
            decision_couplings=decision_couplings,
            decision_RDs=decision_RDs,
            decision_tols=decision_tols,
            history_buffer_size=history_buffer_size,
            logging_frequency=logging_frequency,
            logging_tag=logging_tag,
            logging_fname=logging_fname,
            init_verbose=init_verbose,
            )
    def _check_device_init(self):
        super()._check_device_init()
        
        
        
class maximize_FC1448(maximize_FC):
    def __init__(self,
        decision_CSETs = ['REA_CM01:DCH_D1123:I_CSET',
                          'REA_CM01:DCV_D1123:I_CSET',
                          'REA_CM01:DCH_D1139:I_CSET',
                          'REA_CM01:DCV_D1139:I_CSET',
                          ],
        decision_min = -1.5,
        decision_max =  1.5,
        objective_goal   = {'REA_BTS24:FC_D1164:BC_RD': {'more than': None}},
        objective_weight = {'REA_BTS24:FC_D1164:BC_RD': 1},
        objective_norm   = {'REA_BTS24:FC_D1164:BC_RD': 1e-12},
        objective_fill_none_by_init: Optional[bool] = False,
        objective_p_order:Optional[float] = 2,
        apply_bilog:Optional[bool] = False,
                 
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = np.inf,
        logging_tag: Optional[str] = None,
        logging_fname: Optional[str] = None,
        init_verbose: Optional[bool] = True,        
        ):
        super().__init__(
            decision_CSETs=decision_CSETs,
            decision_min=decision_min,
            decision_max=decision_max,
            objective_goal=objective_goal, 
            objective_weight=objective_weight,
            objective_norm=objective_norm,
            objective_fill_none_by_init=objective_fill_none_by_init,
            objective_p_order=objective_p_order,
            apply_bilog=apply_bilog,
            decision_couplings=decision_couplings,
            decision_RDs=decision_RDs,
            decision_tols=decision_tols,
            history_buffer_size=history_buffer_size,
            logging_frequency=logging_frequency,
            logging_tag=logging_tag,
            logging_fname=logging_fname,
            init_verbose=init_verbose,
            )
    def _check_device_init(self):
        super()._check_device_init()
        
