import numpy as np
from typing import List, Dict, Union, Optional, Callable

from .construct_machineIO import construct_machineIO
from . import _global_machineIO
from .util import warn, get_MEBT_objective_goal_from_BPMoverview

  
    
def get_preset(name: str,Q=None,A=None, machineIO=_global_machineIO, BPM_snapshot_fname=None, offline=False) -> Dict:
    prst = {}
    if Q is None or A is None:
        try:
            if offline:
                raise ValueError
            SCS = _global_machineIO.caget("ACS_DIAG:DEST:ACTIVE_ION_SOURCE")
            Q   = _global_machineIO.caget("FE_ISRC"+str(SCS)+":BEAM:Q_BOOK")
            A   = _global_machineIO.caget("FE_ISRC"+str(SCS)+":BEAM:A_BOOK")
            if SCS is None or Q is None or A is None:
                raise ValueError
        except:
            A = 2; Q = 1;
    AQ = A/Q
    
    
#========= [U-LEBT]FC814 =============
    if name == '[U-LEBT]FC814':
        # == decision info
        decision_CSETs = [
                  'FE_LEBT:PSC2_D0773:I_CSET',
                  'FE_LEBT:PSC1_D0773:I_CSET',
                  'FE_LEBT:PSC2_D0790:I_CSET',
                  'FE_LEBT:PSC1_D0790:I_CSET',
                  ]
        try:
            if offline:
                raise ValueError
            ave, _ = machineIO._fetch_data(decision_CSETs,0.01)
            decision_min = ave -0.7*AQ
            decision_max = ave +0.7*AQ
        except:
            decision_min = [-0.7*AQ]*len(decision_CSETs)
            decision_max = [ 0.7*AQ]*len(decision_CSETs)
        try:
            if offline:
                raise ValueError
            decision_tol = get_tolerance(decision_CSETs)
        except:
            decision_tol = [0.2]*len(decision_CSETs)
        # == objective info
        PV = 'FE_LEBT:FC_D0814:PKAVG_RD'
        try:
            if offline:
                raise ValueError
            goal = 5*round(machineIO._fetch_data([PV],2)[0][0])
        except:
            goal = 50
        objective_goal  = {PV: {'more than': goal}}
        objective_weight= {PV: 1}
        objective_norm  = {PV: round(0.2*goal)}
        
#========= [U-LEBT]FC814_with_chopper_off =============
    elif name == '[U-LEBT]FC814_with_chopper_off':
        # == decision info
        decision_CSETs = [
                  'FE_LEBT:PSC2_D0773:I_CSET',
                  'FE_LEBT:PSC1_D0773:I_CSET',
                  'FE_LEBT:PSC2_D0790:I_CSET',
                  'FE_LEBT:PSC1_D0790:I_CSET',
                  ]
        try:
            if offline:
                raise ValueError
            ave, _ = machineIO._fetch_data(decision_CSETs,0.01)
            decision_min = ave -0.7*AQ
            decision_max = ave +0.7*AQ
        except:
            decision_min = [-0.7*AQ]*len(decision_CSETs)
            decision_max = [ 0.7*AQ]*len(decision_CSETs)
        try:
            if offline:
                raise ValueError
            decision_tol = get_tolerance(decision_CSETs)
        except:
            decision_tol = [0.2]*len(decision_CSETs)
        # == objective info
        PV = 'FE_LEBT:FC_D0814:AVGNSE_RD'
        try:
            if offline:
                raise ValueError
            goal = 5*round(machineIO._fetch_data([PV],2)[0][0])
        except:
            goal = 50
        objective_goal  = {PV: {'more than': goal}}
        objective_weight= {PV: 1}
        objective_norm  = {PV: round(0.2*goal)}
        
#========= [MEBT]FC1102 =============
    elif name == '[MEBT]FC1102':
        # == decision info
        decision_CSETs = [
                 'FE_LEBT:PSC2_D0929:I_CSET', 'FE_LEBT:PSD1_D0936:V_CSET',  # one HCOR and one V-Edipole
                 'FE_LEBT:PSC2_D0948:I_CSET', 'FE_LEBT:PSC1_D0948:I_CSET',  # one COR pair in L-LEBT
                 'FE_LEBT:PSC2_D0979:I_CSET', 'FE_LEBT:PSC1_D0979:I_CSET',  # one COR pair in L-LEBT
                  ]
        try:
            if offline:
                raise ValueError
            ave, _ = machineIO._fetch_data(decision_CSETs,0.2)
            decision_min = []
            decision_max = []
            for PV in decision_CSETs:
                if 'PSC' in PV:
                    decision_min += [ -0.7*AQ]
                    decision_max += [ +0.7*AQ]
                elif 'PSD' in PV:
                    decision_min += [x0* 0.9995]
                    decision_max += [x0* 1.0005]
                elif 'PSOL' in PV:
                    decision_min += [x0* 0.95]
                    decision_max += [x0* 1.15]
                else:
                    warn(f'decision_min and decision_max for {PV} could not be automatically determined. Please adjust it manually ') 
                    decision_min += [np.nan]
                    decision_max += [np.nan]
        except:
            warn('decision_min and decision_max could not be automatically determined. Please adjust it manually ') 
            decision_min = [np.nan]*len(decision_CSETs)
            decision_max = [np.nan]*len(decision_CSETs)
        try:
            if offline:
                raise ValueError
            decision_tol = get_tolerance(decision_CSETs,machineIO=machineIO)
        except:
            warn('decision_tol could not be automatically determined. Please adjust it manually ') 
            decision_tol = [0.2]*len(decision_CSETs)
        # == objective info   
        if BPM_snapshot_fname:
            objective_goal = get_MEBT_objective_goal_from_BPMoverview(BPM_snapshot_fname)
        else:    
            objective_goal = {'FE_MEBT:BPM_D1056:XPOS_RD' : 0,
                              'FE_MEBT:BPM_D1056:YPOS_RD' : 0,
                              'FE_MEBT:BCM_D1055:AVGPK_RD': {'more than': None},
                              'FE_MEBT:FC_D1102:PKAVG_RD' : {'more than': None},
                             }
        objective_weight = {PV:1 for PV in objective_goal.keys()}
        objective_norm   = {PV:1 for PV in objective_goal.keys()}
        try:
            if offline:
                raise ValueError
            goal = 5*round(machineIO._fetch_data(['FE_MEBT:FC_D1102:PKAVG_RD'],2)[0][0])
        except:
            goal = 50
        objective_goal['FE_MEBT:BCM_D1055:AVGPK_RD'] = {'more than': goal}
        objective_goal['FE_MEBT:FC_D1102:PKAVG_RD' ] = {'more than': 0.8*goal}
        objective_norm['FE_MEBT:BCM_D1055:AVGPK_RD'] = 0.05*goal
        objective_norm['FE_MEBT:FC_D1102:PKAVG_RD' ] = 0.05*0.8*goal
    else:
        warn('Unknown preset. Returning simplest template..') 
        decision_CSETs = [
            'FE_LEBT:PSC2_D0773:I_CSET',
            'FE_LEBT:PSC1_D0773:I_CSET',
            ]
        decision_min = [-AQ]*len(decision_CSETs)
        decision_max = [ AQ]*len(decision_CSETs)
        decision_tol = [0.2]*len(decision_CSETs)
        PV = 'FE_LEBT:FC_D0814:PKAVG_RD'
        objective_goal  = {PV: {'more than': 50}}
        objective_weight= {PV: 1 }
        objective_norm  = {PV: 10}

    preset = {
        'decision_CSETs'  :decision_CSETs,
        'decision_min'    :decision_min,
        'decision_max'    :decision_max,
        'decision_tol'    :decision_tol,
        'objective_goal'  :objective_goal,
        'objective_weight':objective_weight,
        'objective_norm'  :objective_norm,
    }
    return preset
        
        
def get_tolerance(PV_CSETs: List[str], machineIO=_global_machineIO):
    '''
    Automatically define tolerance
    tol is defined by 10% of ramping rate: i.e.) tol = ramping distance in a 0.1 sec
    PV_CSETs: list of CSET-PVs 
    '''
    pv_ramp_rate = []
    for pv in PV_CSETs:
        if 'PSOL' in pv:
            pv_ramp_rate.append(pv[:pv.rfind(':')]+':RRTE_RSET')
        else:
            pv_ramp_rate.append(pv[:pv.rfind(':')]+':RSSV_RSET')
    try:
        ramp_rate,_ = machineIO._fetch_data(pv_ramp_rate,0.1)
        tol = 0.1*ramp_rate
    except:
        if machineIO._test:
            tol = None
        else:
            warn('decision_tol could not be automatically determined. Please adjust it manually ')
            tol = 0.2*len(PV_CSETs)
    return tol



def get_limits(PV_CSETs: List[str], machineIO=_global_machineIO):
    '''
    Automatically retrive limit for PV put
    PV_CSETs: list of CSET-PVs 
    '''
    lo_lim = []
    hi_lim = []
    for pv in PV_CSETs:
        if ':V_CSET' in pv:
#             tmp = [pv.replace(':V_CSET',':V_CSET.LOPR'), pv.replace(':V_CSET',':V_CSET.HOPR')]
            tmp = [pv.replace(':V_CSET',':V_CSET.DRVL'), pv.replace(':V_CSET',':V_CSET.DRVH')]
            tmp,_ =machineIO._fetch_data(tmp,0.1)
            lo_lim.append(tmp[0])
            hi_lim.append(tmp[1])
        elif ':I_CSET' in pv:
#             tmp = [pv.replace(':I_CSET',':I_CSET.LOPR'), pv.replace(':I_CSET',':I_CSET.HOPR')]
            tmp = [pv.replace(':I_CSET',':I_CSET.DRVL'), pv.replace(':I_CSET',':I_CSET.DRVH')]
            tmp,_ =machineIO._fetch_data(tmp,0.1)
            lo_lim.append(tmp[0])
            hi_lim.append(tmp[1])
        else:
            warn(f'failed to find operation limit for {pv}. Manually ensure the control limit')
            lo_lim.append(-np.inf)
            hi_lim.append( np.inf)
    lo_lim = np.array(lo_lim)
    hi_lim = np.array(hi_lim)
    assert np.all(lo_lim < hi_lim)
    return lo_lim, hi_lim


 
def get_RDs(PV_CSETs: List[str], machineIO=_global_machineIO):
    PV_RDs = []
    for pv in PV_CSETs:
        if '_CSET' in pv:
            PV_RDs.append(pv.replace('_CSET','_RD'))
        elif '_MTR.VAL' in pv:
            PV_RDs.append(pv.replace('_MTR.VAL','_MTR.RBV'))
        else:
            self._RuntimeError("Automatic decision of 'RD' for above 'PV_CSET' failed", machineIO)
    _,_ = machineIO._fetch_data(PV_RDs,1)
    return PV_RDs