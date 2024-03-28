import numpy as np
from typing import List, Dict, Union, Optional

from .objFuncs import objFuncGoals
from .util import warn
from .construct_machineIO import construct_machineIO



class maximize_FC814(objFuncGoals):
    def __init__(self,
        decision_CSETs = ['FE_LEBT:PSC2_D0773:I_CSET',
                          'FE_LEBT:PSC1_D0773:I_CSET',
                          'FE_LEBT:PSC2_D0790:I_CSET',
                          'FE_LEBT:PSC1_D0790:I_CSET',
                          ],
        decision_min     = -4,
        decision_max     =  4,
        objective_goal   = {'FE_LEBT:FC_D0814:PKAVG_RD': {'more than': None}},
        objective_weight = {'FE_LEBT:FC_D0814:PKAVG_RD': 1},
        objective_norm   = {'FE_LEBT:FC_D0814:PKAVG_RD': None},
        objective_fill_none_by_init: Optional[bool] = False,
        objective_p_order:Optional[float] = 2,
        apply_bilog:Optional[bool] = False,
                 
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = np.inf,
        logging_tag: Optional[str] = "FC814",
        logging_fname: Optional[str] = None,
        init_verbose: Optional[bool] = True,
        machineIO: Optional[construct_machineIO] = None,
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
            machineIO = machineIO,
            )
    def _check_device_init(self):
        if self.machineIO._test:
            return
        if self.machineIO.caget("FE_LEBT:FC_D0814:LMNEG_RSTS_DRV") == 0:
            self._RuntimeError("FC_D0814 is not in.")
        if self.machineIO.caget('FE_LEBT:FC_D0814:RNG_CMD')==1:
            self._RuntimeError("FC_D0814 range is set to 1uA. Change to 1055uA.")
#             self.machineIO.caput('FE_LEBT:FC_D0814:RNG_CMD',0)
            
        if self.machineIO.caget("FE_LEBT:AP_D0796:LMIN_RSTS")==0 or self.machineIO.caget("FE_LEBT:AP_D0807:LMIN_RSTS")==0:
            self._RuntimeError("Apertures are not in")
        if self.machineIO.caget("ACS_DIAG:CHP:STATE_RD") != 3:
            self._RuntimeError("Chopper blocking.") 

    

    
class maximize_FC977(objFuncGoals):
    def __init__(self,
        decision_CSETs = ['FE_LEBT:PSC2_D0840:I_CSET',
                          'FE_LEBT:PSC1_D0840:I_CSET',
                          'FE_LEBT:PSC2_D0929:I_CSET',
                          'FE_LEBT:PSC1_D0929:I_CSET'],
        decision_min = -4,
        decision_max =  4,
        objective_goal   = {'FE_LEBT:FC_D0977:PKAVG_RD': {'more than': None}},
        objective_weight = {'FE_LEBT:FC_D0977:PKAVG_RD': 1},
        objective_norm   = {'FE_LEBT:FC_D0977:PKAVG_RD': None},
        objective_fill_none_by_init: Optional[bool] = False,
        objective_p_order:Optional[float] = 2,
        apply_bilog:Optional[bool] = False,
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = np.inf,
        logging_tag: Optional[str] = "FC977",
        logging_fname: Optional[str] = None,
        init_verbose: Optional[bool] = True,  
        machineIO: Optional[construct_machineIO] = None,
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
            machineIO = machineIO,
            )
    def _check_device_init(self):
        if self.machineIO._test:
            return
        if self.machineIO.caget("FE_LEBT:FC_D0814:LMNEG_RSTS_DRV") != 0:
            self._RuntimeError("FC_D0814 is in.")
        if self.machineIO.caget("FE_LEBT:FC_D0977:LMNEG_RSTS_DRV") == 0:
            self._RuntimeError("FC_D0977 is not in.")
        if self.machineIO.caget('FE_LEBT:FC_D0977:RNG_CMD')==1:
            warn("FC_D0977 range is set to 1uA. Changing to 1055uA.")
            self.machineIO.caput('FE_LEBT:FC_D0977:RNG_CMD',0)
        if self.machineIO.caget("FE_LEBT:ATT1_D0957:LMOUT_RSTS") == 0:
            warn("ATT1_D0957 is in.") 
        if self.machineIO.caget("FE_LEBT:ATT2_D0957:LMOUT_RSTS") == 0:
            warn("ATT2_D0957 is in.")  
        if self.machineIO.caget("FE_LEBT:ATT1_D0974:LMOUT_RSTS") == 0:
            warn("ATT1_D0974 is in.") 
        if self.machineIO.caget("FE_LEBT:ATT2_D0974:LMOUT_RSTS") == 0:
            warn("ATT2_D0974 is in.") 
        if self.machineIO.caget("ACS_DIAG:CHP:STATE_RD") != 3:
            warn("Chopper blocking.") 
            
  
            
            
class maximize_FC998(objFuncGoals):
    def __init__(self,
        decision_CSETs = [
                        'FE_LEBT:PSC2_D0821:I_CSET',
                        'FE_LEBT:PSC1_D0821:I_CSET',
                        'FE_LEBT:PSC2_D0948:I_CSET',
                        'FE_LEBT:PSC1_D0948:I_CSET',
                          ],
        decision_min = -4,
        decision_max =  4,
        objective_goal   = {'FE_LEBT:FC_D0998:PKAVG_RD': {'more than': None}},
        objective_weight = {'FE_LEBT:FC_D0998:PKAVG_RD': 1},
        objective_norm   = {'FE_LEBT:FC_D0998:PKAVG_RD': None},
        objective_fill_none_by_init: Optional[bool] = False,
        objective_p_order:Optional[float] = 2,
        apply_bilog:Optional[bool] = False,
                 
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = np.inf,
        logging_tag: Optional[str] = "FC998",    
        logging_fname: Optional[str] = None,
        init_verbose: Optional[bool] = True,
        machineIO: Optional[construct_machineIO] = None,
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
            logging_fname = logging_fname,
            init_verbose=init_verbose,
            machineIO = machineIO,
            )
    def _check_device_init(self):
        if self.machineIO._test:
            return
        if self.machineIO.caget("FE_LEBT:FC_D0814:LMNEG_RSTS_DRV") != 0:
            self._RuntimeError("FC_D0814 is in.")
        if self.machineIO.caget("FE_LEBT:FC_D0977:LMNEG_RSTS_DRV") != 0:
            self._RuntimeError("FC_D0977 is in.")

        if self.machineIO.caget("FE_LEBT:FC_D0998:LMIN_RSTS") == 0:
            self._RuntimeError("FC_D0998 is not in.")
        if self.machineIO.caget('FE_LEBT:FC_D0998:RNG_CMD')==1:
            warn("FC_D0998 range is set to 1uA. Changing to 1055uA.")
            self.machineIO.caput('FE_LEBT:FC_D0998:RNG_CMD',0)
        if self.machineIO.caget("DIAG-RIO01:FC_ENABLED1") != 1:
            self._RuntimeError("FC_D0998 is not enabled. Check bottom of FC overview in CSS")
        if self.machineIO.caget("DIAG-RIO01:PICO_ENABLED1") != 1:
            self._RuntimeError("FC_D0998 pico is not enabled. Check bottom of FC overview in CSS")

        if self.machineIO.caget("FE_LEBT:ATT1_D0957:LMOUT_RSTS") == 0:
            warn("ATT1_D0957 is in.") 
        if self.machineIO.caget("FE_LEBT:ATT2_D0957:LMOUT_RSTS") == 0:
            warn("ATT2_D0957 is in.")  
        if self.machineIO.caget("FE_LEBT:ATT1_D0974:LMOUT_RSTS") == 0:
            warn("ATT1_D0974 is in.") 
        if self.machineIO.caget("FE_LEBT:ATT2_D0974:LMOUT_RSTS") == 0:
            warn("ATT2_D0974 is in.") 
        if self.machineIO.caget("ACS_DIAG:CHP:STATE_RD") != 3:
            warn("Chopper blocking.") 
            


        
class maximize_FC1102(objFuncGoals):
    def __init__(self,
#         decision_CSETs = ['FE_LEBT:PSC2_D0979:I_CSET',
#                           'FE_LEBT:PSC1_D0979:I_CSET'],
        decision_CSETs=['FE_LEBT:PSC2_D0948:I_CSET', 'FE_LEBT:PSC1_D0948:I_CSET',
                        'FE_LEBT:PSC2_D0964:I_CSET', 'FE_LEBT:PSC1_D0964:I_CSET',
                        'FE_LEBT:PSC2_D0979:I_CSET', 'FE_LEBT:PSC1_D0979:I_CSET',],
        decision_min= [-3,-3, 0.01, 0.01,    -3,    -3],
        decision_max= [ 3, 3,    3,    3, -0.01, -0.01],
        objective_goal = { 
            'FE_MEBT:BPM_D1056:XPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1056:YPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1056:PHASE_RD': None,    #if goal is None, the value at initialization will be set to goal
            'FE_MEBT:BPM_D1056:MAG_RD'  : {'more than': None},
            'FE_MEBT:BPM_D1072:XPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1072:YPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1072:PHASE_RD': None,    #if goal is None, the value at initialization will be set to goal
            'FE_MEBT:BPM_D1072:MAG_RD'  : {'more than': None},
            'FE_MEBT:BPM_D1094:XPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1094:YPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1094:PHASE_RD':{'more than': None},
            'FE_MEBT:BPM_D1094:MAG_RD'  :{'more than': None},
            'FE_MEBT:BCM_D1055:AVGPK_RD/FE_LEBT:BCM_D0989:AVGPK_RD': {'more than': None},
            'FE_MEBT:FC_D1102:PKAVG_RD': {'more than': None},
                           },
        objective_weight = { 
            'FE_MEBT:BPM_D1056:XPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1056:YPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1056:PHASE_RD': 1., 
            'FE_MEBT:BPM_D1056:MAG_RD'  : 0., 
            'FE_MEBT:BPM_D1072:XPOS_RD' : 0.3,     
            'FE_MEBT:BPM_D1072:YPOS_RD' : 0.3,     
            'FE_MEBT:BPM_D1072:PHASE_RD': 0.8, 
            'FE_MEBT:BPM_D1072:MAG_RD'  : 0., 
            'FE_MEBT:BPM_D1094:XPOS_RD' : 0.1,     
            'FE_MEBT:BPM_D1094:YPOS_RD' : 0.1,     
            'FE_MEBT:BPM_D1094:PHASE_RD': 0.5,
            'FE_MEBT:BPM_D1094:MAG_RD'  : 0.,
            'FE_MEBT:BCM_D1055:AVGPK_RD/FE_LEBT:BCM_D0989:AVGPK_RD': 4,
            'FE_MEBT:FC_D1102:PKAVG_RD': 2,
            },
        objective_norm = { 
            'FE_MEBT:BPM_D1056:XPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1056:YPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1056:PHASE_RD': 1., 
            'FE_MEBT:BPM_D1056:MAG_RD'  : None, 
            'FE_MEBT:BPM_D1072:XPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1072:YPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1072:PHASE_RD': 1., 
            'FE_MEBT:BPM_D1072:MAG_RD'  : None, 
            'FE_MEBT:BPM_D1094:XPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1094:YPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1094:PHASE_RD': 1.,
            'FE_MEBT:BPM_D1094:MAG_RD'  : None,
            'FE_MEBT:BCM_D1055:AVGPK_RD/FE_LEBT:BCM_D0989:AVGPK_RD': 1,
            'FE_MEBT:FC_D1102:PKAVG_RD': None,
            },
        objective_fill_none_by_init: Optional[bool] = False,
        objective_p_order:Optional[float] = 2,
        apply_bilog:Optional[bool] = True,
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = np.inf,
        logging_tag: Optional[str] = "FC1102", 
        logging_fname: Optional[str] = None,
        init_verbose: Optional[bool] = True,
        machineIO: Optional[construct_machineIO] = None,
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
            machineIO=machineIO,
            )
        

    def _check_device_init(self):
#         if self.machineIO._test:
#             return
        '''
        check devices status,
        '''
#         if 'FE_LEBT:PSC2_D0992:I_CSET' not in self.decision_CSETs:
#             if self.machineIO.caget('FE_LEBT:PSC2_D0992:I_CSET')!=0:
#                 warn("FE_LEBT:PSC2_D0992:I_CSET is not zero")
#         if 'FE_LEBT:PSC1_D0992:I_CSET' not in self.decision_CSETs:
#             if self.machineIO.caget('FE_LEBT:PSC1_D0992:I_CSET')!=0:
#                 warn("FE_LEBT:PSC1_D0992:I_CSET is not zero")
        if hasattr(self.objective_weight,'FE_MEBT:FC_D1102:PKAVG_RD'):
            if self.machineIO.caget('FE_MEBT:FC_D1102:RNG_CMD')==1:
                self._RuntimeError("FC_D1102 range is 1uA. Change to 1055uA.")
#                 warn("FC_D1102 range is 1uA. Changing to 1055uA.")
#                 self.machineIO.caput('FE_MEBT:FC_D1102:RNG_CMD',0)
            if self.machineIO.caget("FE_MEBT:FC_D1102:LMIN_RSTS") == 0:
                self._RuntimeError("FC_D1102 is not in.")
            if self.machineIO.caget("DIAG-RIO01:FC_ENABLED2") != 1:
                self._RuntimeError("FC_D1102 is not enabled. Check bottom of FC overview in CSS")
            if self.machineIO.caget("DIAG-RIO01:PICO_ENABLED2") != 1:
                self._RuntimeError("FC_D1102 pico is not enabled. Check bottom of FC overview in CSS")
        if self.machineIO.caget("ACS_DIAG:CHP:STATE_RD") != 3:
            self._RuntimeError("Chopper blocking.") 
        if self.machineIO.caget("FE_LEBT:ATT1_D0957:LMOUT_RSTS") == 0:
            self._RuntimeError("ATT1_D0957 is in.") 
        if self.machineIO.caget("FE_LEBT:ATT2_D0957:LMOUT_RSTS") == 0:
            self._RuntimeError("ATT2_D0957 is in.")  
        if self.machineIO.caget("FE_LEBT:ATT1_D0974:LMOUT_RSTS") == 0:
            self._RuntimeError("ATT1_D0974 is in.") 
        if self.machineIO.caget("FE_LEBT:ATT2_D0974:LMOUT_RSTS") == 0:
            self._RuntimeError("ATT2_D0974 is in.")  
        if self.machineIO.caget("GTS_FTS:MSTR_N0001:PCUR_DFAC_RD") > 10.1:
            self._RuntimeError("Duty factor is more than 10%. Reduce it below 10% to avoid RFQ damage")
            


            

    
