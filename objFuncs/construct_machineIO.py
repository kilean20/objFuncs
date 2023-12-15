import time
import warnings
import numpy as np
import pandas as pd

from .gui import popup_handler
from .util import warn, cyclic_mean_var, suppress_outputs
    
        
popup_ramping_not_OK = popup_handler("Action required", "Ramping not OK. Manually adjust PV CSETs to jitter the power suppply before continue.")
_n_popup_ramping_not_OK = 0

_ensure_set_timeout = 30
_fetch_data_time_span = 2.05
_check_chopper_blocking = True


try:
    from phantasy import fetch_data as _phantasy_fetch_data
    from phantasy import ensure_set as _phantasy_ensure_set
    _phantasy_imported = True
except:
    warn("import phantasy failed")
    _phantasy_imported = False
    _check_chopper_blocking = False
    
try:
    from epics import caget as _epics_caget
    from epics import caput as _epics_caput
    _epics_imported = True
    
    with suppress_outputs():
        if _epics_caget("REA_EXP:ELMT") is not None:
            _check_chopper_blocking = False    # don't check FRIB chopper if machine is REA
    
    def _epics_fetch_data(pvlist,time_span = _fetch_data_time_span, 
                          abs_z = None, 
                          with_data=False,
                          verbose=False):
        data = {pv:[] for pv in pvlist}
        t0 = time.monotonic()
        while (time.monotonic()-t0 < time_span):
            for pv in pvlist:   
                data[pv].append(caget(pv))
            time.sleep(0.2)
        for pv in pvlist:
            mean = np.mean(data[pv])
            std  = np.std (data[pv])
            if abs_z is not None and std > 0:
                mask = np.logical_and(mean -abs_z*std < data[pv], data[pv] < mean +abs_z*std )
                mean = np.mean(np.array(data[pv])[mask])
                std  = np.std (np.array(data[pv])[mask])
            data[pv].append(len(data[pv]))
            data[pv].append(mean)
            data[pv].append(std )
        index = list(np.arange(len(data[pv])))
        index[-1] = 'std'
        index[-2] = 'mean'
        index[-3] = '#'
        data = pd.DataFrame(data,index=index).T

        return data['mean'].to_numpy(), data
    
    
    def _epics_ensure_set(setpoint_pv,readback_pv,goal,
                          tol=0.01,
                          timeout=_ensure_set_timeout,
                          verbose=False):
        t0 = time.monotonic()
        for pvname,value in zip(setpoint_pv,goal):
            if verbose:
                print('-- _epics_ensure_set')
                print('  {pvname}: {value}')
            _epics_caput(pvname,value)
        
        is_ramping = [True]*len(readback_pv)
        while(time.monotonic()-t0 < timeout or np.any(is_ramping)):
            i = 0
            for pvname,value,tolerance in zip(readback_pv,goal,tol):
                if np.abs(value-_epics_caget(pvname)) < tolerance:
                    is_ramping[i] = False
                i+=1
            time.sleep(0.2)            
except:
    warn("import epics failed")
    _epics_imported = False
    _check_chopper_blocking = False
    

def _dummy_fetch_data(pvlist,time_span = _fetch_data_time_span, 
                      abs_z = None, 
                      with_data=False,
                      verbose=False):

    time.sleep(time_span)
    data = {pv:[0]*10 for pv in pvlist} # 10 dummy meausre
    for pv in pvlist:
        mean = np.mean(data[pv])
        std  = np.std (data[pv])
        if abs_z is not None and std > 0:
            mask = np.logical_and(mean -abs_z*std < data[pv], data[pv] < mean +abs_z*std )
            mean = np.mean(np.array(data[pv])[mask])
            std  = np.std (np.array(data[pv])[mask])
        data[pv].append(len(data[pv]))
        data[pv].append(mean)
        data[pv].append(std )
    index = list(np.arange(len(data[pv])))
    index[-1] = 'std'
    index[-2] = 'mean'
    index[-3] = '#'
    data = pd.DataFrame(data,index=index).T

    return data['mean'].to_numpy(), data


    
class construct_machineIO:
    def __init__(self):
        self._test = False
#         self._machine = 'FRIB'
        self._ensure_set_timeout = _ensure_set_timeout
        self._ensure_set_timewait_after_ramp = 0.25
        self._fetch_data_time_span = _fetch_data_time_span
        self._return_obj_var = False
        self._check_chopper_blocking = _check_chopper_blocking
        #self.view()
        self._n_popup_ramping_not_OK = _n_popup_ramping_not_OK
        self._verbose = False
        
        
    def caget(self,pvname):
            if _epics_imported:
                return _epics_caget(pvname)
            else:
                if self._test:
                    warn("EPICS is not imported. caget will return fake zero")
                    return 0
                else:
                    raise ValueError("EPICS is not imported. cannot caget")
            
    def caput(self,pvname,value):
            if _epics_imported:
                _epics_caput(pvname,value)
            else:
                if self._test:
                    warn("EPICS is not imported. caput will do noting")
                else:
                    raise ValueError("EPICS is not imported. cannot caput")
      
    def ensure_set(self,
                   setpoint_pv,readback_pv,goal,
                   tol=0.01,
                   timeout=None,
                   verbose=False):
        
        verbose = verbose or self._verbose
        timeout = timeout or self._ensure_set_timeout
        t0 = time.monotonic()
        if self._test:
            pass
        elif _phantasy_imported:
            _phantasy_ensure_set(setpoint_pv,readback_pv,goal,tol,timeout,verbose)
        elif _epics_imported:
            _epics_ensure_set(setpoint_pv,readback_pv,goal,tol,timeout,verbose)
        else:
            raise ValueError("Cannot change SET: PHANTASY or EPICS is not imported.")
        # print("_ensure_set_timewait_after_ramp",self._ensure_set_timewait_after_ramp)
        time.sleep(self._ensure_set_timewait_after_ramp)
        
        if time.monotonic() - t0 > timeout+self._ensure_set_timewait_after_ramp: 
            warn("ramping_not_OK. trying again...")
            print("ramping_not_OK. trying again...")
            if self._n_popup_ramping_not_OK<5:
                popup_ramping_not_OK()
                self._n_popup_ramping_not_OK +=1
            else:
                warn("'ramping_not_OK' issued 5 times already. Ignoring 'ramping_not_OK' issue...")
            
            
            
    def fetch_data(self,pvlist,
                   time_span = _fetch_data_time_span, 
                   abs_z = None, 
                   with_data=False,
                   verbose=False,
                   check_chopper_blocking = None,
                   debug = False):
        time_span = time_span or self._fetch_data_time_span
        check_chopper_blocking = check_chopper_blocking or self._check_chopper_blocking
        
        if check_chopper_blocking and not self._test :
            pvlist = list(pvlist) + ["ACS_DIAG:CHP:STATE_RD"]
            
        
        # print("self._fetch_data_time_span",self._fetch_data_time_span)
            
        while(True):
            
            if debug:
                print('[debug][objFuncs][machineIO][construct_machineIO]fetch_data')
                print(  '_phantasy_imported, _epics_imported',_phantasy_imported, _epics_imported)
                print(  'pvlist', pvlist)
                
            if _phantasy_imported:
                ave,raw = _phantasy_fetch_data(pvlist,time_span,abs_z,with_data=True,verbose=verbose)
            elif _epics_imported:
                ave,raw =    _epics_fetch_data(pvlist,time_span,abs_z,with_data=True,verbose=verbose)
            elif self._test:
                warn("PHANTASY or EPICS is not imported. fetch_data will return zeros")
                ave,raw =    _dummy_fetch_data(pvlist,time_span,abs_z,with_data=True,verbose=verbose)
            else:
                raise ValueError("PHANTASY or EPICS is not imported and the machineIO is not in test mode.")
                
            if check_chopper_blocking and not self._test :
                if ave[-1] != 3:
                    warn("Chopper blocked during fetch_data. Re-try in 5 sec... ")
                    time.sleep(5)
                    continue
                else:
                    pvlist = pvlist[:-1]
                    ave  = ave[:-1]
                    raw.drop("ACS_DIAG:CHP:STATE_RD",inplace=True)
                    break
            else:
                break
                
            
        if np.any(pd.isna(raw[0])):
            raise ValueError("fetch_data 0th column have NaN. re-fetch")
        
        std = raw['std'].to_numpy()
        for i,pv in enumerate(pvlist):
            if 'PHASE' in pv:
                if 'BPM' in pv:
                    Lo = -90
                    Hi =  90
                else:
                    Lo = -180
                    Hi =  180
                nsample = raw.iloc[i,-3]    
                mean,var = cyclic_mean_var(raw.iloc[i,:nsample].dropna().values,Lo,Hi)
                ave[i] = mean
                std[i] = var**0.5

                
        if with_data:
            raw['mean'] = ave
            raw['std']  = std
            return ave,raw
        else:
            return ave,None
        
    def view(self):
        for k,v in vars(self).items():
            print("  ",k,":",v)

            

class construct_manual_fetch_data:
    def __init__(self,pv_for_manual_fetch):
        self.pv_for_manual_fetch = pv_for_manual_fetch
        self._fetch_data_time_span = _fetch_data_time_span
        
    def __call__(self,pvlist,
                 time_span=None, 
                 abs_z=None, 
                 with_data=False,
                 verbose=False):
        time_span = time_span or self._fetch_data_time_span
        
        if _phantasy_imported or _epics_imported:
            print("=== Manual Input. Leave blank for automatic data read. ===")
        else:
            print("=== Manual Input: ===")
        
        values = []
        pvlist_blank = []
        ipv_blank = []
        for i,pv in enumerate(pvlist):
            val = None
            if pv in self.pv_for_manual_fetch:
                try:
                    val = float(input(pv + ': '))
                except:
                    print("Input not accepatble format")
                    if _epics_imported:
                        print(f"trying caget {pv}...")
                        test = _epics_caget(pv)
                        if test is None:
                            while(val is None):
                                try:
                                    val = float(input(pv + ': '))
                                except:
                                    print("Input not accepatble format")
                                    pass
            if val is None:
                pvlist_blank.append(pv)
                ipv_blank.append(i)
            values.append(val)
            
        n_data = 2  # dummy numer of data samples
        if len(pvlist_blank) > 0:
            if _phantasy_imported:
                ave,raw = _phantasy_fetch_data(pvlist_blank,time_span,abs_z,with_data=True,verbose=verbose)
                n_data = raw.shape[1]-3
            elif _epics_imported:
                ave,raw =    _epics_fetch_data(pvlist_blank,time_span,abs_z,with_data=True,verbose=verbose)
                n_data = raw.shape[1]-3
            else:
                print("Automatic data read failed. please input manually:")
                for i,pv in zip(ipv_blank,pvlist_blank):
                    val = float(input(pv))
                    values[i] = val
                ipv_blank = []
                pvlist_blank = []
        
        data = {pv:[val]*n_data for pv,val in zip(pvlist,values)}
        for i,pv in enumerate(pvlist):
            if i in ipv_blank:
                data[pv].append(raw.loc[pv]['#'])
                data[pv].append(raw.loc[pv]['mean'])
                data[pv].append(raw.loc[pv]['std'] )
            else:
                mean = np.mean(data[pv])
                std  = np.std (data[pv])
                data[pv].append(n_data)
                data[pv].append(mean  )
                data[pv].append(std   )
        index = list(np.arange(len(data[pv])))
        index[-1] = 'std'
        index[-2] = 'mean'
        index[-3] = '#'
        data = pd.DataFrame(data,index=index).T

        return data['mean'].to_numpy(), data