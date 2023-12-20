import os
import time
import datetime
from copy import deepcopy as copy
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Callable
from collections import OrderedDict
from IPython.display import display

import numpy as np
import pandas as pd
import pickle

from .util import warn, elu, cyclic_distance, cyclic_mean, cyclic_mean_var, get_class_hierarchy, get_picklable_items_from_dict, print_nested_dict
from . import _global_machineIO

__all__ = ["objFuncBase","objFuncGoals","objFuncMultiConditionalGoals"]
_eps = 1e-15

        
def get_tolerance(PV_CSETs: List[str], machineIO=_global_machineIO):
    '''
    Automatically define tolerance
    tol is defined by 5% of ramping rate: i.e.) tol = ramping distance in a 0.05 sec
    PV_CSETs: list of CSET-PVs 
    '''
    pv_ramp_rate = []
    for pv in PV_CSETs:
        if 'PSOL' in pv:
            pv_ramp_rate.append(pv[:pv.rfind(':')]+':RRTE_RSET')
        else:
            pv_ramp_rate.append(pv[:pv.rfind(':')]+':RSSV_RSET')
    ramp_rate,_ = machineIO.fetch_data(pv_ramp_rate,1)
    return 0.2*ramp_rate

 
def get_RDs(PV_CSETs: List[str], machineIO=_global_machineIO):
    PV_RDs = []
    for pv in PV_CSETs:
        if '_CSET' in pv:
            PV_RDs.append(pv.replace('_CSET','_RD'))
        elif '_MTR.VAL' in pv:
            PV_RDs.append(pv.replace('_MTR.VAL','_MTR.RBV'))
        else:
            self._RuntimeError("Automatic decision of 'RD' for above 'PV_CSET' failed", machineIO)
    _,_ = machineIO.fetch_data(PV_RDs,1)
    return PV_RDs


def getPVs_from_objective_keys(objective_keys):
    PVs = []
    for key in objective_keys:
        iratio = key.find('/')
        if iratio!=-1:
            PVs.append(key[:iratio])
            PVs.append(key[iratio+1:])
        else:
            PVs.append(key)
    return PVs
    
class objFuncBase():#(ABC):
    def __init__(self,
        decision_CSETs: List[Union[str,List[str]]] = None,  
        decision_min: Union[float,List[float]] = None,  
        decision_max: Union[float,List[float]] = None,  
        decision_couplings: Optional[Dict] = None,  
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = np.inf,
        logging_tag: Optional[str] = "",
        logging_fname: Optional[str] = None,
        load_from_file: Optional[str] = None,
        init_verbose: Optional[bool] = True,
        called_by_child: Optional[bool] = False, 
        machineIO: Optional = None,
        ):
        '''
        decision_CSETs: [List of] List of CSET-PVs for control. 
        decision_min: Float or List of minimum of 'decision_CSETs', must be len(decision_min) == len(decision_CSETs) 
        decision_max: Float or List of maximum of 'decision_CSETs', must be len(decision_min) == len(decision_CSETs) 
        decision_couplings: CSETs that need to be coupled with one of decision_CSETs. 
            (e.g) decision_couplings = {"EQUAD1_D0000": {"EQUAD2_D0000":-1}, }    <- here "EQUAD1_D0000" should be in decision_CSETs and "EQUAD2_D0000" is coupled to "EQUAD1_D0000" with coupling coefficient -1.            
        decision_RDs: List of RD-PVs for control used to ensure ramping completion. If 'None' automatically decided by changing 'CSET' to 'RD' in PV name 
        decision_tols: List or numpy array of tolerance used to ensure ramping completion. If 'None' automatically decided based on ramping rate definition.objFuncs 
        '''
        if load_from_file is None:
            assert decision_CSETs is not None
            assert decision_min is not None
            assert decision_max is not None
        else:
            self.load(load_from_file)
            return
        
        
        self.init_time = datetime.datetime.now()
        self.input_parameters = get_picklable_items_from_dict(locals())        
        self.class_hierarchy = get_class_hierarchy(self)
        self.machineIO = _global_machineIO
        
        self.decision_CSETs = decision_CSETs
        if type(decision_min) is float or type(decision_min) is int:
            decision_min = [float(decision_min)]*len(decision_CSETs)
        assert len(decision_min) == len(decision_CSETs)
        self.decision_min = np.array(decision_min).astype(np.float64)
        if type(decision_max) is float or type(decision_max) is int:
            decision_max = [float(decision_max)]*len(decision_CSETs)
        assert len(decision_max) == len(decision_CSETs)
        self.decision_max = np.array(decision_max).astype(np.float64)
        self.decision_bounds = np.array([(d_min,d_max) for (d_min,d_max) in zip(decision_min, decision_max)])
         
        
        if decision_RDs is None:
            try:
                self.decision_RDs = get_RDs(decision_CSETs)
            except:
                if hasattr(self.machineIO,"_test"):
                    if self.machineIO._test:
                        warn("Automatic decision of 'RDs' for 'decision_CSETs' failed. Since self.machineIO is in test mode, we will simply use decision_CSETs as decision_RD")
                        self.decision_RDs = decision_CSETs
                    else:
                        self._RuntimeError("Automatic decision of 'RDs' for 'decision_CSETs' failed. Check 'decision_CSETs' or provide 'decision_RDs' manually.")
                else:
                    warn("Automatic decision of 'RDs' for 'decision_CSETs' failed. Since self.machineIO is in test mode, we will simply use decision_CSETs as decision_RD")
                    self.decision_RDs = decision_CSETs
        else:
            self.decision_RDs = decision_RDs
        if decision_tols is None:
            try:
                self.decision_tols = get_tolerance(decision_CSETs)                 
            except:
                if self.machineIO._test:
                    self.decision_tols = None
                else:
                    self._RuntimeError("Automatic decision of the 'decision_tols' failed. Check 'decision_CSETs' or provide 'decision_tols' manually.")
        else:
            self.decision_tols = np.array(decision_tols)
            assert np.all(self.decision_tols>0)
            
            
        self.decision_couplings = decision_couplings
        if self.decision_couplings is not None:
            self.decision_couplings = OrderedDict(decision_couplings)
            self._coupled_decision_info = {"CSETs":[], "index":[], "coeff":[]}
            for pv,couples in self.decision_couplings.items():
                assert pv in self.decision_CSETs
                icouple = list(self.decision_CSETs).index(pv)
                assert type(couples) is dict
                for cpv,coeff in couples.items():
                    assert cpv not in self._coupled_decision_info["CSETs"]
                    assert type(coeff) in [float,int]
                    self._coupled_decision_info["CSETs"].append(cpv)
                    self._coupled_decision_info["index"].append(icouple)
                    self._coupled_decision_info["coeff"].append(float(coeff))
            
            self._coupled_decision_info["coeff"] = np.array(self._coupled_decision_info["coeff"])
            try:
                self._coupled_decision_info["RDs"] = get_RDs(self._coupled_decision_info["CSETs"])
            except:
                self._RuntimeError("Automatic decision of 'RDs' for coupling 'PV_CSETs' failed. Contact Kilean Hwang.")
            try:
                self._coupled_decision_info["tols"] = get_tolerance(self._coupled_decision_info["CSETs"]) 
            except:
                print(self._coupled_decision_info["CSETs"])
                self._RuntimeError("Automatic decision of the 'decision_tols' of coupled CSETs (shown above) failed.")
            
        
        self.history_buffer_size = history_buffer_size
        self.logging_frequency = logging_frequency or 10
        self.logging_tag = logging_tag or ""
        self.logging_fname = logging_fname
        self.history = {'time':[],
                        'decision_CSETs':{'names':copy(self.decision_CSETs),
                                          'values':[]},
                        'decision_RDs':{'names':copy(self.decision_RDs),
                                        'values':[]},
                       }
        if self.decision_couplings is not None:
            self.history['coupled_decision_CSETs'] = {'names':copy(self._coupled_decision_info["CSETs"]),
                                                      'values':[]}

        self._get_xinit()
        
        if init_verbose and not called_by_child:
            self.print_class_info()
            
                
    def print_class_info(self, include_history = False):
        print("======== class info ========")
        dic = get_picklable_items_from_dict(self.__dict__)
        if 'input_parameters' in dic.keys():
            del dic['input_parameters']
        if not include_history:
            del dic['history']
        print_nested_dict(dic)
        print()
                
        
    def write_log(self, path="./log/",tag=None, fname=None):
        if fname is None and (
               self.logging_frequency == np.inf or \
               len(self.history["time"])%self.logging_frequency != self.logging_frequency-1):
            return
        if tag is not None:
            tag = self.logging_tag + "_" + tag
        else:
            tag = self.logging_tag
        if path[-1]!="/":
            path += "/"
        if not os.path.isdir(path):
            os.mkdir(path)
        now = str(self.init_time)
        now = "["+str(now)[:str(now).rfind(':')].replace(" ","_").replace(":","").replace("-","")+"]"
        
        if fname is None:
            if tag is None:
                fname = self.logging_fname or now+self.class_hierarchy[0]+".pkl"
            else:
                if tag !="":
                    if tag[0] != "_":
                        tag = "_"+tag
                fname = now+self.class_hierarchy[0]+tag+".pkl"
            fname = path + fname

        self.save(fname)
        

    def save(self,fname):
        dic = get_picklable_items_from_dict(self.__dict__)
        if  fname[-4:]!='.pkl':
            fname = fname +'.pkl'
            print(f'only .pkl file extension is accepted. saving to: {fname}')
        pickle.dump(dic, open(fname, "wb"))

        
    def load(self, fname):
        i = fname.rfind('.')
        if  fname[-4:]!='.pkl':
            fname = fname +'.pkl'
            print(f'only .pkl file extension is accepted. loading from: {fname}')
        warn("loading picklable data from file. load non-picklable data manually")    
        with open(fname, 'rb') as file:
            state_dict = pickle.load(file)
            dic = get_picklable_items_from_dict(state_dict)
            for key,val in dic.items():
                setattr(self, key, val)
        self.machineIO = _global_machineIO
        
        
    def _RuntimeError(s):
        if self.machineIO._test:
            warn(s)
        else:
            raise RuntimeError(s)


    def add_decision_CSETs(self,
        decision_CSETs: List[Union[str,List[str]]],
        decision_min: Union[float,List[float]],
        decision_max: Union[float,List[float]],
        previous_decision_CSETs: Optional[List[float]] = None,
        decision_couplings: Optional[Dict] = None,  
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        ):
        
        assert len(decision_CSETs)>0
        for pv in decision_CSETs:
            assert pv not in self.decision_CSETs
            
        if previous_decision_CSETs is None:
            previous_decision_CSETs,_ = self.machineIO.fetch_data(decision_CSETs,1)
        assert len(previous_decision_CSETs) == len(decision_CSETs)
  
        self.decision_CSETs = list(self.decision_CSETs) + decision_CSETs
        
        if type(decision_min) is float or type(decision_min) is int:
            decision_min = [float(decision_min)]*len(decision_CSETs)
        assert len(decision_min) == len(decision_CSETs)
        self.decision_min = np.concatenate((self.decision_min,decision_min))
        if type(decision_max) is float or type(decision_max) is int:
            decision_max = [float(decision_max)]*len(decision_CSETs)
        assert len(decision_max) == len(decision_CSETs)
        self.decision_max = np.concatenate((self.decision_max,decision_max))
        self.decision_bounds = np.array([(d_min,d_max) for (d_min,d_max) in zip(self.decision_min, self.decision_max)])
        
        if np.any(np.array(previous_decision_CSETs) > decision_max) or np.any(np.array(previous_decision_CSETs) < decision_min):
            warn('added decision_CSETs are out of decision bounds')
        
        if decision_RDs is None:
            try:
                decision_RDs = get_RDs(decision_CSETs)
            except:
                print(decision_CSETs)
                self._RuntimeError("Automatic decision of 'RDs' for above 'decision_CSETs' failed. Check 'decision_CSETs' or provide 'decision_RDs' manually.")
        else:
            assert len(decision_RDs) == len(decision_CSETs)
        self.decision_RDs = list(self.decision_RDs) + list(decision_RDs)
        
        if decision_tols is None:
            try:
                decision_tols = get_tolerance(decision_CSETs)
            except:
                self._RuntimeError("Automatic decision of the 'decision_tols' failed. Check 'decision_CSETs' or provide 'decision_tols' manually.")
        else:
            assert len(decision_tols) == len(decision_CSETs)
        self.decision_tols = list(self.decision_tols) + list(decision_tols)   
            
        if decision_couplings is not None:
            decision_couplings = OrderedDict(decision_couplings)
            if self.decision_couplings is None:
                self.decision_couplings = decision_couplings
            else:
                for key in decision_couplings.keys():
                    assert key not in self.decision_couplings
                self.decision_couplings.update(decision_couplings)

        if self.decision_couplings is not None:
            self.decision_couplings = OrderedDict(self.decision_couplings)
            self._coupled_decision_info = {"CSETs":[], "index":[], "coeff":[]}
            for pv,couples in self.decision_couplings.items():
                assert pv in self.decision_CSETs
                icouple = list(self.decision_CSETs).index(pv)
                assert type(couples) is dict
                for cpv,coeff in couples.items():
                    assert cpv not in self._coupled_decision_info["CSETs"]
                    assert type(coeff) in [float,int]
                    self._coupled_decision_info["CSETs"].append(cpv)
                    self._coupled_decision_info["index"].append(icouple)
                    self._coupled_decision_info["coeff"].append(float(coeff))
            
            self._coupled_decision_info["coeff"] = self._coupled_decision_info["coeff"]
            try:
                self._coupled_decision_info["RDs"] = get_RDs(self._coupled_decision_info["CSETs"])
            except:
                print(self._coupled_decision_info["CSETs"])
                self._RuntimeError("Automatic decision of 'RDs' for above coupling 'PV_CSETs' failed. Contact Kilean Hwang.")
            try:
                self._coupled_decision_info["tols"] = get_tolerance(self._coupled_decision_info["CSETs"]) 
            except:
                print(self._coupled_decision_info["CSETs"])
                self._RuntimeError("Automatic decision of the 'decision_tols' of coupled CSETs (shown above) failed.")
                
        self.history['decision_CSETs']['names']=copy(self.decision_CSETs)
        self.history['decision_RDs'  ]['names']=copy(self.decision_RDs)
        
        arr = np.array(self.history['decision_CSETs']['values'])
        b,d = arr.shape
        _ = np.zeros(( b,len(self.decision_CSETs) ))
        _[:,:d] = arr[:,:]
        _[:,d:] = np.array(previous_decision_CSETs)[None,:]
        self.history['decision_CSETs']['values'] = _.tolist()
        
        arr = np.array(self.history['decision_RDs']['values'])
        _ = np.zeros(( b,len(self.decision_CSETs) ))
        _[:,:d] = arr[:,:]
        _[:,d:] = np.array(previous_decision_CSETs)[None,:]
        self.history['decision_RDs']['values'] = _.tolist()
                
        
    def _check_device_init(self):
        '''
        check devices status before defining objective
        '''
        pass

            
    def _set_decision(self,x):
        x = np.array(x)
        if not x.ndim==1:
            x = x.flatten()
        assert len(x) == len(self.decision_CSETs)
        if self.decision_couplings is None:
            self.machineIO.ensure_set(self.decision_CSETs,self.decision_RDs,x,self.decision_tols)
        else:
            n = len(self.decision_CSETs) 
            x_ = np.zeros(n+len(self._coupled_decision_info["CSETs"]))
            x_[:n] = x[:]
            count = 0
            for i,coeff in zip(self._coupled_decision_info["index"],self._coupled_decision_info["coeff"]):
                x_[n+count] = x[i]*coeff
                count += 1
            self.machineIO.ensure_set( list(self.decision_CSETs)+ list(self._coupled_decision_info["CSETs"]),
                              list(self.decision_RDs  )+ list(self._coupled_decision_info["RDs"]  ),
                              x_,
                              list(self.decision_tols )+ list(self._coupled_decision_info["tols"] ))
        self.history['decision_CSETs']['values'].append(x)
        if self.decision_couplings is not None:
            self.history['coupled_decision_CSETs']['values'].append(x_[n:])
        self.history['time'].append(datetime.datetime.now())
            
        
    def _get_xinit(self):
        x0, _ = self.machineIO.fetch_data(self.decision_CSETs,1)
        self.x0 = x0
        
        if hasattr(self,'decision_min'):
            if np.any(self.x0 < self.decision_min) or np.any(self.x0 > self.decision_max):
                warn("Initial decision point is out of the decision bounds.")
                
            
            
class objFuncGoals(objFuncBase):
    def __init__(self,
        decision_CSETs: List[str] = None,
        decision_min: Union[float,List[float]] = None,
        decision_max: Union[float,List[float]] = None,
        objective_goal:  Dict = None, 
        objective_weight:  Dict = None,
        objective_norm: Optional[Dict] = None,
        objective_fill_none_by_init: Optional[bool] = False,
        objective_p_order:Optional[int] = 2,
        apply_bilog:Optional[bool] = False,
                 
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = np.inf,
        logging_tag: Optional[str] = "",
        logging_fname: Optional[str] = None,
        load_from_file: Optional[str] = None,
        init_verbose: Optional[bool] = True,
        called_by_child: Optional[bool] = False, 
                   
        ):
        '''
        objective_goal: a Dict specifing goal of key=PVname, val=goal. 
                            (e.g.) obj = 1 - |(value-goal)/(norm +_eps)|
                        if None and 'objective_fill_none_by_init' is True
                            the value at initialization will be set to goal 
                        if dict, must of the form 
                            {'less than': float or None}  or 
                            {'more than':float or None}
                        if the goal is {'more than':goal}
                            (e.g.) obj = -elu(-(value-goal)/(norm +_eps))
                        if the goal is {'less than':goal}
                            (e.g.) obj = -elu( (value-goal)/(norm +_eps))
                        where elu(x) = x (x>0)
                                     = e^x - 1 (x<=0)
                        if the goal is {'more than':None},  and 'objective_fill_none_by_init' is True
                            the value at initialization will be measured and used

        objective_weight: a Dict specifing weight of key=PVname, val=weight.  
                          if weight is 0 all corresponding objective will not be measured and calculated
        objective_norm: a Dict specifing normalization factor of key=PVname, val=norm. 
                        This value effectively serves as an tolerace of corresponding objective
        objective_p_order: integer for objective power. default=2
                           (e.g) obj = sign(obj)*|obj|^p_order
                           large p_order is useful to strongly penalize values far from goal

        apply_bilog: regularize aggregated objective value to suppress value away from zero.
                     -> obj_tot = np.sign(obj_tot)*np.log(1+np.abs(obj_tot))
                     It is useful when objective optimum is near zero and 
                     non-optimum objective value can become very large 
                     (which can happen when 'objective_p_order' is large)
                        
        e.g.)
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
        objective_norm_auto_helper = {}
        '''
        super().__init__(
            decision_CSETs=decision_CSETs,
            decision_min=decision_min,
            decision_max=decision_max,
            decision_couplings=decision_couplings,
            decision_RDs=decision_RDs,
            decision_tols=decision_tols,
            history_buffer_size = history_buffer_size,
            logging_frequency = logging_frequency,
            logging_tag = logging_tag,
            logging_fname = logging_fname,
            load_from_file = load_from_file,
            init_verbose = init_verbose,
            called_by_child =True,
            )        
        if load_from_file is None:
            assert decision_CSETs is not None
            assert decision_min is not None
            assert decision_max is not None
            assert objective_goal is not None
            assert objective_weight is not None
        else:
            return
        
        # objective_weight must be defined first
        self.input_parameters = get_picklable_items_from_dict(locals())
        self.objective_weight = OrderedDict([(key,val) for key,val in objective_weight.items() if val > 0.])
        isBPMphase = False
        for key in self.objective_weight.keys():
            if 'PHASE' in key:
                if not apply_bilog:
                    warn('PHASE in in objective. turning on "apply_bilog" is strongly recommended: It can be done by obj.apply_bilog = True')

        self.objective_goal   = OrderedDict([(key,val) for key,val in objective_goal.items() if key in self.objective_weight.keys()])
        assert self.objective_goal.keys() == self.objective_weight.keys()
        
        objective_fill_none_by_init = objective_fill_none_by_init or False
        isNone = False
        for key,goal in self.objective_goal.items():
#             print('key,goal',key,goal)
            if type(goal) is dict:
                assert len(goal) == 1
                for k,v in goal.items():
                    assert k in ["more than", "less than"]
                    if v is None:
                        isNone = True
#                         print("isNone, key,goal",key,goal)
            elif goal is None:
#                 print("isNone, key,goal",key,goal)
                if not objective_fill_none_by_init:
                    self._RuntimeError("objective_goal for "+key+" is must be provided unless 'objective_fill_none_by_init' is 'True'")
                isNone = True
            else:
                assert type(goal) in [int,float]
                self.objective_goal[key]=float(goal)
                

        self.objective_norm = OrderedDict({})
        for key in self.objective_weight.keys():
            if type(objective_norm) is dict:
                if key in objective_norm.keys():
                    self.objective_norm[key] = objective_norm[key]
                elif objective_fill_none_by_init:
                    self.objective_norm[key] = None
                    isNone = True
                else:
                    self._RuntimeError("objective_norm for "+key+" is must be provided unless 'objective_fill_none_by_init' is 'True'")
            else:
                if objective_fill_none_by_init:
                    self.objective_norm[key] = None
                    isNone = True
                else:
                    self._RuntimeError("objective_norm for "+key+" is must be provided unless 'objective_fill_none_by_init' is 'True'")
 
        self._check_device_init() 
        self._get_xinit()
        self.objective_RDs = getPVs_from_objective_keys(list(self.objective_goal.keys()))
        
        if objective_fill_none_by_init and isNone:
            warn('Some objective goal or norm is None. Setting from current values but we srongly recommed to manually set values')
            self._fill_obj_none()
        
        wtot = np.sum(list(self.objective_weight.values()))
        for key in self.objective_weight.keys():
            self.objective_weight[key] /= wtot
        
        self.history['objectives'] = {'names': list(self.objective_weight.keys()),
                                      'values':[]}
        self.history['objectives']['total'] = []
        self.objective_RDs = getPVs_from_objective_keys(list(self.objective_goal.keys()))               
        self.history["objective_RDs"] = {'names': self.objective_RDs,
                                         'values':[]} 

        self.objective_p_order = objective_p_order or 2+np.clip(np.log(len(self.objective_goal.keys())),a_min=0,a_max=4)
        self.apply_bilog = apply_bilog

        if init_verbose and not called_by_child:   
#             print("== objective_goal ==")
#             display(self.objective_goal)
#             print("== objective_norm ==")
#             display(self.objective_norm)
#             print("== objective_weight ==")
#             display(self.objective_weight)
#             print("== etc ==")
#             print("objective_p_order: ", self.objective_p_order)
#             print("apply_bilog: ", apply_bilog)
            self.print_class_info()
            
#         self(self.x0)
            
            
    def _fill_obj_none(self):
        RDs,_ = self.machineIO.fetch_data(self.objective_RDs,5,abs_z=3)
        RDs = {key:val for key,val in zip(self.objective_RDs,RDs)}
#         print("RDs",RDs)
        
        for key, goal in self.objective_goal.items():
#             print("key,goal",key,goal)            
            if type(goal) is float:
                continue
            if goal is None:
                if "FC_" in key and "PKAVG" in key:
                    self.objective_goal[key] = np.clip(RDs[key],a_min=2,a_max=None)
                elif "BCM_" in key and "AVGPK_RD" in key:
                    self.objective_goal[key] = np.clip(RDs[key],a_min=2,a_max=None)
                elif "BPM" in key and "MAG_RD" in key:
                    self.objective_goal[key] = np.clip(RDs[key],a_min=5e-3,a_max=None)
                else:
                    self.objective_goal[key] = RDs[key]
#                 print("self.objective_goal[key]",self.objective_goal[key])
                continue
            assert len(goal)==1
            for k,v in goal.items():
#                 print("k,v",k,v)
                if v is None:
                    iratio = key.find('/')
                    if iratio!=-1:
                        self.objective_goal[key][k] = RDs[key[:iratio]]/( RDs[key[iratio+1:]] + _eps )
                    else:
                        if "FC_" in key and "PKAVG" in key:
                            self.objective_goal[key][k] = np.clip(RDs[key],a_min=2,a_max=None)
                        elif "BCM_" in key and "AVGPK_RD" in key:
                            self.objective_goal[key][k] = np.clip(RDs[key],a_min=2,a_max=None)
                        elif "BPM" in key and "MAG_RD" in key:
                            self.objective_goal[key][k] = np.clip(RDs[key],a_min=5e-3,a_max=None)
                        else:
                            self.objective_goal[key][k] = RDs[key]
                    
        for key, norm in self.objective_norm.items():
            if norm is None:
                iratio = key.find('/')
                if iratio!=-1:
                    self.objective_norm[key] = 0.1*np.abs( RDs[key[:iratio]]/( RDs[key[iratio+1:]] + _eps ) )
                elif "POS" in key or "PHASE" in key:
                    self.objective_norm[key] = 1
                elif "FC_" in key and "PKAVG" in key:
                    self.objective_norm[key] = 0.1*np.clip(RDs[key],a_min=2,a_max=None)
                elif "BCM_" in key and "AVGPK_RD" in key:
                    self.objective_norm[key] = 0.1*np.clip(RDs[key],a_min=2,a_max=None)
                elif "BPM" in key and "MAG_RD" in key:
                    self.objective_norm[key] = 0.1*np.clip(RDs[key],a_min=5e-3,a_max=None)
                else:
                    self._RuntimeError("Could not decide normalization factor for "+key+
                                     " automatically. Please provide normalization factor manually")
        
        
    def _calculate_objectives(self,
                              RD_data,
                             ):
        
        objective_goal = self.objective_goal
        objective_weight = self.objective_weight
        objective_norm = self.objective_norm
        p_order = self.objective_p_order
        
        objs = []
        obj_tot = 0
        i = 0
        for key,goal in objective_goal.items():
#             print("key,goal",key,goal)
#             print(objective_norm[key])
            iratio = key.find('/')
            if iratio==-1:
                value = RD_data[i]
                i+=1
            else:
                value = RD_data[i]/(RD_data[i+1] +_eps)
                i+=2 
            if type(goal) is float:
                if 'BPM' in key and 'PHASE' in key:
                    obj = 2 -(np.abs(cyclic_distance(value,goal,-90,90)/(objective_norm[key] +_eps)))**p_order
                else:
                    obj = 2 -(np.abs((value-goal)/(objective_norm[key] +_eps)))**p_order
            elif 'more than' in goal:
                obj = -elu(-(value-goal['more than'])/(objective_norm[key] +_eps))
#                 obj = np.sign(obj)*np.abs(obj)**p_order
            elif 'less than' in goal:    
                obj = -elu( (value-goal['less than'])/(objective_norm[key] +_eps))
#                 obj = np.sign(obj)*np.abs(obj)**p_order
            else:
                self._RuntimeError("goal is not recognized. It must be float or {'more than':float} or {'less than':float}")
            objs.append(obj)
            obj_tot += objective_weight[key]*obj    
                
        if self.apply_bilog:
            obj_tot = np.sign(obj_tot)*np.log(1+np.abs(obj_tot))
            
        return obj_tot, objs
    
    
    def _get_object(self,time_span=None,abs_z=None):
        
        ave_data, _ = self.machineIO.fetch_data(
                            list(self.decision_RDs) + list(self.objective_RDs),
                            time_span=time_span,
                            abs_z=abs_z)
        
        #regularize
        obj_tot,objs = self._calculate_objectives(ave_data[len(self.decision_RDs):]) #\
#               +self.regularize(x)
                        
        self.history['decision_RDs' ]['values'].append(ave_data[:len(self.decision_RDs) ])
        self.history['objective_RDs']['values'].append(ave_data[ len(self.decision_RDs):])
        self.history['objectives'   ]['values'].append(objs)
        self.history['objectives']['total'].append(obj_tot)  
        
        super().write_log()
        
        return obj_tot
                        
        
    def update_objective(self,
        objective_goal: Optional[Dict] = None,
        objective_weight: Optional[Dict] = None,
        objective_norm: Optional[Dict] = None,
        objective_p_order: Optional[int] = None,
        ):
        
        if objective_goal is not None:
            assert type(objective_goal) == dict
            for key,val in objective_goal.items():
                assert key in self.objective_goal
                if type(val) == dict:
                    for key1,val1 in val.items():
                        self.objective_goal[key][key1] = val1
                else:
                    self.objective_goal[key] = val
        if objective_weight is not None:
            assert type(objective_weight) == dict
            for key,val in objective_weight.items():
                if val > 0:
                    assert key in self.objective_weight
                else:
                    continue
                if type(val) == dict:
                    for key1,val1 in val.items():
                        self.objective_weight[key][key1] = val1
                else:
                    self.objective_weight[key] = val
        if objective_norm is not None:
            assert type(objective_norm) == dict
            for key,val in objective_norm.items():
                assert key in self.objective_norm
                if type(val) == dict:
                    for key1,val1 in val.items():
                        self.objective_norm[key][key1] = val1
                else:
                    self.objective_norm[key] = val

        if objective_p_order is not None:
            self.objective_p_order = objective_p_order
            
                        
        assert len(self.history['decision_RDs']['values']) == len(self.history['objective_RDs']['values']) == len(self.history['objectives']['values']) == len(self.history['objectives']['total'])
        
        for i,RD in enumerate(self.history['objective_RDs']['values']):
            obj_tot,objs = self._calculate_objectives(RD) #\
            self.history['objectives']['values'][i] = objs
            self.history['objectives']['total'][i] = obj_tot
    
            
    def __call__(self,x,time_span=None,abs_z=None, callbacks=None):
        self._set_decision(x)
        objs = self._get_object(time_span=time_span,
                                abs_z=abs_z)
        if callbacks is not None:
            for f in callbacks:
                f()
        return objs 
        
    
    
class objFuncMultiConditionalGoals(objFuncBase):
    def __init__(self,
        decision_CSETs: List[str],
        decision_min: Union[float,List[float]],
        decision_max: Union[float,List[float]],
        objective_goal:  Dict, 
        objective_weight:  Dict,
        objective_norm: Dict,  
        conditional_SETs: Dict[str,List[float]],
        conditional_RDs: Optional[List[str]] = None,
        conditional_tols: Optional[List[float]] = None,
        conditional_control_cost_more: bool = True,
        each_condition_objective_weights: Optional[List[float]] = None,
        
        objective_fill_none_by_init: Optional[bool] = False,         
        objective_p_order:Optional[float] = 2,
        apply_bilog:Optional[bool] = False,
                 
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = np.inf,
        logging_tag: Optional[str] = "",
        logging_fname: Optional[str] = None,
        init_verbose: Optional[bool] = True,
        called_by_child: Optional[bool] = False, 
        ):
        '''
        conditions_SETs: 
            a OrderedDict specifing fixed self.machineIOs for defining conditions (e.g. charge state) 
            for aggregated objective definition. 
                (e.g.) for different chage coditions (i.e. charge selector), 
                  conditional_SETs = {
                      'FS1_BBS:CSEL_D2405:CTR_MTR.VAL':[-17,17],
                      'FS1_BBS:CSEL_D240array5:GAP_MTR.VAL':[-17,17],
                      }
                      
        conditional_RDs:
            readback PVs for conditions_SETs.keys()  !!important: order must be consistent with conditions_SETs
        
        objective_goal: a Dict specifing goal of key=PVname, val=list of goals. 
                        The list must be length of the conditions. 
                        Otherwise the goal will be duplicated to form 
                        a list of len(conditions_SETs.keys())
        
        objective_weight: a Dict specifing weight of key=PVname, val=list of weights. 
                          The list must be length of the conditions. 
                          Otherwise the weight will be duplicated to form 
                          a list of len(conditions_SETs.keys())
                          
        objective_norm: a Dict specifing normalization factor of key=PVname, val=list of norm. 
                        The list must be length of the conditions. 
                        Otherwise the norm will be duplicated to form 
                        a list of len(conditions_SETs.keys())
            (e.g.)
                objective_goal = { 
                    'FS2_BMS:BPM_D4142:XPOS_RD' : [-0.5, 0.5]   <-- two different goals for two conditions
                    'FS2_BMS:BPM_D4142:YPOS_RD' : None,
                    'FS2_BMS:BPM_D4142:PHASE_RD': [None, None]
                    'FS2_BMS:BPM_D4142:MAG_RD'  : [{'more than': None},{'more than': None},],
                    'FS2_BBS:BCM_D4169:AVGPK_RD': {'more than': None},   <-- one goal for two conditions
                                   },
                objective_weight = { 
                    'FS2_BMS:BPM_D4142:XPOS_RD' : [1,1],    <-- two different weights for two conditions
                    'FS2_BMS:BPM_D4142:YPOS_RD' : 1,        <-- one weights for two conditions
                    'FS2_BMS:BPM_D4142:PHASE_RD': 1,
                    'FS2_BMS:BPM_D4142:MAG_RD'  : 1,,
                    'FS2_BBS:BCM_D4169:AVGPK_RD': 1,
                    },
                objective_norm = { 
                    'FS2_BMS:BPM_D4142:XPOS_RD' : 1,
                    'FS2_BMS:BPM_D4142:YPOS_RD' : 1,
                    'FS2_BMS:BPM_D4142:PHASE_RD': 1,
                    'FS2_BMS:BPM_D4142:MAG_RD'  : None,
                    'FS2_BBS:BCM_D4169:AVGPK_RD': None
                    },
        '''
        super().__init__(
            decision_CSETs=decision_CSETs,
            decision_min=decision_min,
            decision_max=decision_max,
            decision_couplings=decision_couplings,
            decision_RDs=decision_RDs,
            decision_tols=decision_tols,
            history_buffer_size = history_buffer_size,
            logging_frequency = logging_frequency,
            logging_tag = logging_tag,
            logging_fname = logging_fname,
            init_verbose = init_verbose,
            called_by_child = True,
            )      
        
        
        i=0
        for key,val in conditional_SETs.items():
            if i==0:
                self.n_condition = len(val)
            else:
                assert self.n_condition == len(val)
            i+=1 
        self.conditional_SETs = OrderedDict(conditional_SETs)
        self.conditional_control_cost_more = conditional_control_cost_more
        
        if logging_tag is None:
            logging_tag = ""
        else:
            assert type(logging_tag) == str
        if len(logging_tag)>0:
            if logging_tag[-1] != "_":
                logging_tag += "_"
        self.condition_controller = objFuncBase(
            decision_CSETs = list(self.conditional_SETs.keys()),
            decision_min = np.min(list(self.conditional_SETs.values()),axis=1),
            decision_max = np.max(list(self.conditional_SETs.values()),axis=1),
            decision_couplings = None,  
            decision_RDs = conditional_RDs,
            decision_tols = conditional_tols,
            history_buffer_size = history_buffer_size,
            logging_frequency = np.inf,
            logging_tag = logging_tag + "condition_controler",
            init_verbose = init_verbose,
            )
          
            
        def _get_ith_val_from_list_in_dict(i,dict_):
            out = {}
            for key,val in dict_.items():
                if type(val) is list:
                    out[key] = val[i]
                else:
                    out[key] = val
            return out
        
        if each_condition_objective_weights is None:
            self.each_condition_objective_weights = np.ones(self.n_condition)
        else:
            assert len(each_condition_objective_weights) == n_condition
            self.each_condition_objective_weights = each_condition_objective_weights
        self.each_condition_objective_weights /= self.each_condition_objective_weights.sum()
        self.objFuncGoals = []
        for i in range(self.n_condition):
            _objective_goal = _get_ith_val_from_list_in_dict(i,objective_goal)
            _objective_weight = _get_ith_val_from_list_in_dict(i,objective_weight)
            _objective_norm = _get_ith_val_from_list_in_dict(i,objective_norm)       
            self.condition_controller._set_decision(np.array(list(self.conditional_SETs.values()))[:,i])
            self.objFuncGoals.append(
                objFuncGoals
                    (
                    decision_CSETs= decision_CSETs,
                    decision_min  = decision_min,
                    decision_max  = decision_max,
                    objective_goal   = _objective_goal,
                    objective_weight = _objective_weight,
                    objective_norm   = _objective_norm,
                    objective_fill_none_by_init = objective_fill_none_by_init,
                    objective_p_order= objective_p_order,
                    apply_bilog = apply_bilog,
                    decision_couplings = decision_couplings,
                    decision_RDs = decision_RDs,
                    decision_tols = decision_tols,
                    history_buffer_size = history_buffer_size,
                    logging_frequency = np.inf,
                    logging_tag = logging_tag + "objFuncGoal_condition"+str(i),
                    init_verbose = init_verbose,    
                    )
                )
        

        for i,o in enumerate(self.objFuncGoals):
            self.history["condition"+str(i)] = o.history
        self.history["condition controller"] = self.condition_controller.history
        self.history["decision_CSETs"] = self.history["condition0"]["decision_CSETs"]
        self.history["decision_RDs"]   = self.history["condition0"]["decision_RDs"]
        self.history["time"] = self.history["condition0"]["time"]
        
        
        self.objective_RDs = getPVs_from_objective_keys(list(o.objective_weight.keys()))
        self.history['objectives'] = {'names' : list(o.objective_weight.keys()),
                                      'values':[],
                                      'total' :[] }
                                      
        if init_verbose and not called_by_child:
            self.print_class_info()
        
    def __call__(self,decision_vals,time_span=None,abs_z=None, callbacks=None):
        
        decision_vals = np.atleast_2d(decision_vals)
        batch_size, dim = decision_vals.shape
        
        obj = np.zeros((batch_size,self.n_condition))
        
        if self.conditional_control_cost_more:
            try:
                proximal_index = np.argsort([o.history['time'][-1] for o in self.objFuncGoals])[::-1]
            except:
                proximal_index = range(self.n_condition)
            for i in proximal_index:
                self.condition_controller._set_decision(np.array(list(self.conditional_SETs.values()))[:,i])
                for b in range(batch_size):
                    obj[b,i] = self.objFuncGoals[i](decision_vals[b,:],time_span=time_span,abs_z=abs_z)
        else:
            for b in range(batch_size):
                try:
                    proximal_index = np.argsort([o.history['time'][-1] for o in self.objFuncGoals])[::-1]
                except:
                    proximal_index = range(self.n_condition)
                for i in proximal_index:
                    self.condition_controller._set_decision(np.array(list(self.conditional_SETs.values()))[:,i])
                    obj[b,i] = self.objFuncGoals[i](decision_vals[b,:],time_span=time_span,abs_z=abs_z)
                    
        
        for b in range(batch_size):
            obj_each_cond = [o.history['objectives']['values'][-batch_size+b] for o in self.objFuncGoals]
            self.history['objectives']['values'].append( 
                np.sum(self.each_condition_objective_weights[:,None]*np.array(obj_each_cond),axis=0) )
        
        obj_batches = np.sum(obj*self.each_condition_objective_weights[None,:],axis=1)
        self.history['objectives']['total'] += obj_batches.tolist()
        
        if callbacks is not None:
            for f in callbacks:
                f()
        return obj_batches
        
    
    def add_decision_CSETs(self,
        decision_CSETs: List[Union[str,List[str]]],
        decision_min: Union[float,List[float]],
        decision_max: Union[float,List[float]],
        previous_decision_CSETs: Optional[List[float]] = None,
        decision_couplings: Optional[Dict] = None,  
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        ):
#         super().add_decision_CSETs(
#             decision_CSETs=decision_CSETs,
#             decision_min=decision_min,
#             decision_max=decision_max,
#             previous_decision_CSETs=previous_decision_CSETs,
#             decision_couplings=decision_couplings,
#             decision_RDs=decision_RDs,
#             decision_tols=decision_tols,
#             )
        
        for o in self.objFuncGoals:
            o.add_decision_CSETs(
                decision_CSETs=decision_CSETs,
                decision_min=decision_min,
                decision_max=decision_max,
                previous_decision_CSETs=previous_decision_CSETs,
                decision_couplings=decision_couplings,
                decision_RDs=decision_RDs,
                decision_tols=decision_tols,
                )
        o = self.objFuncGoals[0]
        self.decision_CSETs = o.decision_CSETs
        self.decision_min = o.decision_min 
        self.decision_max = o.decision_max 
        self.decision_bounds = o.decision_bounds
        self.decision_couplings = o.decision_couplings
        self.decision_RDs = o.decision_RDs
        self.decision_tols = o.decision_tols
        for i,o in enumerate(self.objFuncGoals):
            self.history["condition"+str(i)] = o.history
        self.history["decision_CSETs"] = o.history["decision_CSETs"]
        self.history["decision_RDs"]   = o.history["decision_RDs"]
        self.history["time"] = o.history["time"]
        
            
    def update_objective(self,
        objective_goal: Optional[Union[List[Dict],Dict]] = None,
        objective_weight: Optional[Union[List[Dict],Dict]] = None,
        objective_norm: Optional[Union[List[Dict],Dict]] = None,
        objective_p_order: Optional[int] = None,
        each_condition_objective_weights: Optional[List[float]] = None,
        time_span:Optional[float] = None,
        ):
        if type(objective_goal) == dict or objective_goal is None:
            objective_goal = [objective_goal]*self.n_condition
        if type(objective_weight) == dict or objective_weight is None:
            objective_weight = [objective_weight]*self.n_condition
        if type(objective_norm) == dict or objective_norm is None:
            objective_norm = [objective_norm]*self.n_condition
        if each_condition_objective_weights is not None:
            assert len(each_condition_objective_weights) == self.n_condition
            self.each_condition_objective_weights = each_condition_objective_weights
        for i,o in enumerate(self.objFuncGoals):
            o.update_objective(
                objective_goal = objective_goal[i],
                objective_weight = objective_weight[i],
                objective_norm = objective_norm[i],
                objective_p_order = objective_p_order,
                time_span = time_span,
                )
        
        self.history['objectives']['values'] = []
        n = len(o.history['objectives']['values'])
        for i in range(n):
            objs = [o.history['objectives']['values'][i] for o in self.objFuncGoals]
            self.history['objectives']['values'].append( 
                np.sum(self.each_condition_objective_weights[:,None]*np.array(objs),axis=0) )
            
        objs = [o.history['objectives']['total'] for o in self.objFuncGoals]
        self.history['objectives']['total'] = np.sum(self.each_condition_objective_weights[:,None]*np.array(objs),axis=0)
        
        
        
class objFuncMultiConditionalVar(objFuncMultiConditionalGoals):
    def __init__(self,
        decision_CSETs: List[str],
        decision_min: Union[float,List[float]],
        decision_max: Union[float,List[float]],
        objective_goal:  Dict, 
        objective_weight:  Dict,
        objective_norm: Dict,
        objective_var_weight: Dict,
        
        conditional_SETs: Dict,
        conditional_RDs: Optional[List[str]] = None,
        conditional_tols: Optional[List[float]] = None,
        conditional_control_cost_more:[bool] = True,
                 
        each_condition_objective_weights: Optional[List[float]] = None,         
#         objective_BPM_var_weight: Optional[Dict] = {'XY':2./3,'PHASE':1./3},
        var_obj_weight_ratio: Optional[float] = 1.,
                 
        objective_p_order:Optional[float] = 2,
        apply_bilog:Optional[bool] = False,
                 
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        init_verbose: Optional[bool] = True,
                         
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
            each_condition_objective_weights = each_condition_objective_weights,

            objective_p_order = objective_p_order,
            apply_bilog = apply_bilog,

            decision_couplings = decision_couplings,
            decision_RDs = decision_RDs,
            decision_tols = decision_tols,
            history_buffer_size = history_buffer_size,
            init_verbose = init_verbose
            )
        
        self.objective_var_weight = OrderedDict([(key,val) for key,val in objective_var_weight.items() if val > 0.])
        wtot = np.sum(list(self.objective_var_weight.values()))
        for key in self.objective_var_weight.keys():
            assert key in self.history['objectives']['names']
            self.objective_var_weight[key] /= wtot
        self.history['objectives_var'] = {'names': list(objective_var_weight.keys()),
                                          'description':'variation of objectives(residuals) over conditions',
                                          'values':[]}
        self.objective_var_ipv = {key:self.history['objectives']['names'].index(key) for key in self.objective_var_weight.keys()}
        self.objective_var_norm = OrderedDict()
        for key in self.objective_var_weight.keys():
            self.objective_var_norm[key] = np.mean([self.objFuncGoals[i].objective_norm[key] for i in range(self.n_condition)])
        self.var_obj_weight_ratio = var_obj_weight_ratio
        
        if init_verbose:
            self.print_class_info()
        
        
    def __call__(self,decision_vals,time_span=None,abs_z=None,callbacks=None):
        decision_vals = np.atleast_2d(decision_vals)
        batch_size, dim = decision_vals.shape
        obj_batches = super().__call__(decision_vals,time_span=time_span,abs_z=abs_z)
        obj_vars = np.zeros(batch_size,len(self.objective_var_weight),self.n_condition)
        for icon in range(self.n_condition):
            objective_RDs = self.history['condition'+str(icon)]['objective_RDs']
            for j,key in enumerate(self.objective_var_weight.keys()):
                ipv = self.objective_var_ipv[key]
                obj_vars[:,j,icon] = objective_RDs['values'][-batch_size:,ipv]/self.objective_var_norm[key]
        obj_vars = -np.var(obj_vars,axis=2) + 0.5
        self.history['objectives_var'] += obj_vars.tolist()
        
        obj_batches *= (1.-var_obj_weight_ratio)
        obj_batches += var_obj_weight_ratio*np.sum(np.array(objective_var_weight.values())[None,:]*obj_vars,axis=1)
        self.history['objectives']['total'] += obj_batches.tolist()

        if callbacks is not None:
            for f in callbacks:
                f()
        return obj_batches
        
        
    def update_objective(self,
        objective_goal: Optional[Union[Dict]] = None,
        objective_weight: Optional[Union[Dict]] = None,
        objective_norm: Optional[Union[List[Dict],Dict]] = None, 
        
        objective_var_weight: Optional[Union[Dict]] = None,                 
        each_condition_objective_weights: Optional[List[float]] = None,         
        var_obj_weight_ratio: Optional[float] = None,                

        objective_p_order: Optional[int] = None,
        time_span:Optional[float] = None,
        ):
        super().update_objective(
            objective_goal=objective_goal,
            objective_weight=objective_weight,
            objective_norm=objective_norm,
            objective_p_order=objective_p_order,
            time_span=time_span,
            )
   
        if objective_var_weight is not None:
            self.objective_var_weight = OrderedDict([(key,val) for key,val in objective_var_weight.items() if val > 0.])
            wtot = np.sum(list(self.objective_var_weight.values()))
            for key in self.objective_var_weight.keys():
                assert key in self.history['objectives']['names']
                self.objective_var_weight[key] /= wtot
            self.history['objectives_var'] = {'names': list(objective_var_weight.keys()),
                                              'description':'variation of objectives(residuals) over conditions',
                                              'values':[]}
            self.objective_var_ipv = {key:self.history['objectives']['names'].index(key) for key in self.objective_var_weight.keys()}
            self.objective_var_norm = OrderedDict()
            for key in self.objective_var_weight.keys():
                self.objective_var_norm[key] = np.mean([self.objFuncGoals[i].objective_norm[key] for i in range(self.n_condition)])
                
        self.var_obj_weight_ratio = var_obj_weight_ratio or self.var_obj_weight_ratio

        obj_batches = self.history['objectives']['total']
        obj_vars = np.zeros(len(obj_batches),len(self.objective_var_weight),self.n_condition)
        for icon in range(self.n_condition):
            objective_RDs = self.history['condition'+str(icon)]['objective_RDs']
            for j,key in enumerate(self.objective_var_weight.keys()):
                ipv = self.objective_var_ipv[key]
                obj_vars[:,j,icon] = objective_RDs['values'][:,ipv]/self.objective_var_norm[key]
        obj_vars = -np.var(obj_vars,axis=2) + 0.5
        self.history['objectives_var'] += obj_vars.tolist()
        
        obj_batches *= (1.-var_obj_weight_ratio)
        obj_batches += var_obj_weight_ratio*np.sum(np.array(objective_var_weight.values())[None,:]*obj_vars,axis=1)
        self.history['objectives']['total'] = obj_batches.tolist()
