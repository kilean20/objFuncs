import numpy as np
from typing import List, Union, Optional, Dict

from .objFuncs import objFuncMultiConditionalBPMvar, _test, warn


        
        
class multiQ_traj_FS1(objFuncMultiConditionalBPMvar):
    def __init__(self,
        decision_CSETs = [
            'FS1_BBS:PSC2_D2412:I_CSET',
            'FS1_BBS:PSC1_D2412:I_CSET',
            'FS1_BBS:PSC1_D2476:I_CSET',
            'FS1_BMS:PSC1_D2476:I_CSET',
#             'FS1_BBS:PSQ_D2424:I_CSET',
#             'FS1_BBS:PSS_D2419:I_CSET',                        
                               ],
        decision_min = [-14., 20., 20.,-16.],#, 100., 0.],
        decision_max = [ 14., 80., 80., 16.],#, 180., 4.],
        decision_couplings = None,
#         decision_couplings = {
#             'FS1_BBS:PSQ_D2424:I_CSET': {"FS1_BBS:PSQ_D2463:I_CSET":1}, 
#             'FS1_BBS:PSS_D2419:I_CSET': {"FS1_BBS:PSS_D2469:I_CSET":1}, 
#                              },
        objective_goal = { 
            'FS2_BMS:BPM_D4164:XPOS_RD' : 0.0,     #(mm)
            'FS2_BMS:BPM_D4164:YPOS_RD' : 0.0,     #(mm)
            'FS2_BMS:BPM_D4164:PHASE_RD': None,    #78.98, #(deg)
            'FS2_BMS:BPM_D4164:MAG_RD'  : np.inf,
            'FS2_BMS:BPM_D4177:XPOS_RD' : 0.0,     #(mm)
            'FS2_BMS:BPM_D4177:YPOS_RD' : 0.0,     #(mm)
            'FS2_BMS:BPM_D4177:PHASE_RD': None,    #-26.71, #(deg)
            'FS2_BMS:BPM_D4177:MAG_RD'  : np.inf,
            'FS1_BMS:BPM_D2587:XPOS_RD' : 0.0,     #(mm)
            'FS1_BMS:BPM_D2587:YPOS_RD' : 0.0,     #(mm)
            'FS1_BMS:BPM_D2587:PHASE_RD': None,    #-26.71, #(deg)
            'FS1_BMS:BPM_D2587:MAG_RD'  : np.inf,
            },
        objective_weight = { 
            'FS2_BMS:BPM_D4164:XPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4164:YPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4164:PHASE_RD': 1,    #78.98, #(deg)
            'FS2_BMS:BPM_D4164:MAG_RD'  : 1,
            'FS2_BMS:BPM_D4177:XPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4177:YPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4177:PHASE_RD': 1,    #-26.71, #(deg)
            'FS2_BMS:BPM_D4177:MAG_RD'  : 1,
            'FS1_BMS:BPM_D2587:XPOS_RD' : 1,     #(mm)
            'FS1_BMS:BPM_D2587:YPOS_RD' : 1,     #(mm)
            'FS1_BMS:BPM_D2587:PHASE_RD': 1,    #-26.71, #(deg)
            'FS1_BMS:BPM_D2587:MAG_RD'  : 1,
            },
        objective_norm = { 
            'FS2_BMS:BPM_D4164:XPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4164:YPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4164:PHASE_RD': 1,    
            'FS2_BMS:BPM_D4164:MAG_RD'  : None,
            'FS2_BMS:BPM_D4177:XPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4177:YPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4177:PHASE_RD': 1,    
            'FS2_BMS:BPM_D4177:MAG_RD'  : None,
            'FS1_BMS:BPM_D2587:XPOS_RD' : 1,     #(mm)
            'FS1_BMS:BPM_D2587:YPOS_RD' : 1,     #(mm)
            'FS1_BMS:BPM_D2587:PHASE_RD': 1,    
            'FS1_BMS:BPM_D2587:MAG_RD'  : None,
            },
        
        conditional_SETs = {'FS1_BBS:CSEL_D2405:CTR_MTR.VAL':[-17,0,17],
                           'FS1_BBS:CSEL_D2405:GAP_MTR.VAL':[10,10,10],},
        conditional_RDs: Optional[List[str]] = None,
        conditional_tols: Optional[List[float]] = [0.01,0.01],
        conditional_control_cost_more:[bool] = True,
                 
        objective_BPM_var_weight: Optional[Dict] = {'XY':1./6,'PHASE':2./6},
                 
        objective_p_order:Optional[float] = 2,
        objective_RD_avg_time:Optional[float] = 2,
        apply_bilog:Optional[bool] = False,
                 
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        ):
        super().__init__(
            decision_CSETs = decision_CSETs,
            decision_min = decision_min,
            decision_max = decision_max,
            objective_goal = objective_goal,
            objective_weight = objective_weight,
            objective_norm = objective_norm,

            conditional_SETs = conditional_SETs,
            conditional_RDs = conditional_RDs,
            conditional_tols = conditional_tols,
            conditional_control_cost_more = conditional_control_cost_more,

            objective_p_order = objective_p_order,
            objective_RD_avg_time = objective_RD_avg_time,
            apply_bilog = apply_bilog,

            decision_couplings = decision_couplings,
            decision_RDs = decision_RDs,
            decision_tols = decision_tols,
            history_buffer_size = history_buffer_size,
            )
    

    
    
class multiQ_traj_FS2(objFuncMultiConditionalBPMvar):
    def __init__(self,
        decision_CSETs = [
            'FS2_BTS:PSC2_D3962:I_CSET',
            'FS2_BBS:PSC1_D4010:I_CSET',
            'FS2_BBS:PSC1_D4096:I_CSET',
            'FS2_BMS:PSC2_D4146:I_CSET',
            'FS2_BBS:PSQ_D3996:I_CSET',
            'FS2_BBS:PSS_D4000:I_CSET',  
                           ],
        decision_min = [-14.,  0.,  0.,-10., 150., 0.],
        decision_max = [ 14., 10., 10., 10., 230., 3.5],
        decision_couplings = {
            'FS2_BBS:PSQ_D3996:I_CSET': {"FS2_BBS:PSQ_D4109:I_CSET":1}, 
            'FS2_BBS:PSS_D4000:I_CSET': {"FS2_BBS:PSS_D4106:I_CSET":1}, 
                             },
        objective_goal = { 
            'FS2_BMS:BPM_D4164:XPOS_RD' : 0.0,     #(mm)
            'FS2_BMS:BPM_D4164:YPOS_RD' : 0.0,     #(mm)
            'FS2_BMS:BPM_D4164:PHASE_RD': None,    #78.98, #(deg)
            'FS2_BMS:BPM_D4164:MAG_RD'  : np.inf,
            'FS2_BMS:BPM_D4177:XPOS_RD' : 0.0,     #(mm)
            'FS2_BMS:BPM_D4177:YPOS_RD' : 0.0,     #(mm)
            'FS2_BMS:BPM_D4177:PHASE_RD': None,    #-26.71, #(deg)
            'FS2_BMS:BPM_D4177:MAG_RD'  : np.inf,
            },
        objective_weight = { 
            'FS2_BMS:BPM_D4164:XPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4164:YPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4164:PHASE_RD': 1,    #78.98, #(deg)
            'FS2_BMS:BPM_D4164:MAG_RD'  : 1,
            'FS2_BMS:BPM_D4177:XPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4177:YPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4177:PHASE_RD': 1,    #-26.71, #(deg)
            'FS2_BMS:BPM_D4177:MAG_RD'  : 1,
            },
        objective_norm = { 
            'FS2_BMS:BPM_D4164:XPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4164:YPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4164:PHASE_RD': 1,    
            'FS2_BMS:BPM_D4164:MAG_RD'  : None,
            'FS2_BMS:BPM_D4177:XPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4177:YPOS_RD' : 1,     #(mm)
            'FS2_BMS:BPM_D4177:PHASE_RD': 1,    
            'FS2_BMS:BPM_D4177:MAG_RD'  : None,
            },
        
        conditional_SETs = {'FS1_BBS:CSEL_D2405:CTR_MTR.VAL':[-17,0,17],
                           'FS1_BBS:CSEL_D2405:GAP_MTR.VAL':[10,10,10],},
        conditional_RDs: Optional[List[str]] = None,
        conditional_tols: Optional[List[float]] = [0.01,0.01],
        conditional_control_cost_more:[bool] = True,
                 
        objective_BPM_var_weight: Optional[Dict] = {'XY':1./4,'PHASE':2./4},
                 
        objective_p_order:Optional[float] = 2,
        objective_RD_avg_time:Optional[float] = 2,
        apply_bilog:Optional[bool] = False,
                 
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        ):
        super().__init__(
            decision_CSETs = decision_CSETs,
            decision_min = decision_min,
            decision_max = decision_max,
            objective_goal = objective_goal,
            objective_weight = objective_weight,
            objective_norm = objective_norm,

            conditional_SETs = conditional_SETs,
            conditional_RDs = conditional_RDs,
            conditional_tols = conditional_tols,
            conditional_control_cost_more = conditional_control_cost_more,

            objective_p_order = objective_p_order,
            objective_RD_avg_time = objective_RD_avg_time,
            apply_bilog = apply_bilog,

            decision_couplings = decision_couplings,
            decision_RDs = decision_RDs,
            decision_tols = decision_tols,
            history_buffer_size = history_buffer_size,
            )