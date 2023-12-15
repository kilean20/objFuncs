import numpy as np
from typing import List, Dict, Union, Optional
import pandas as pd

from .objFuncs import objFuncBase
from .util import warn
from .machineIO import machineIO


def _RuntimeError(s):
    if machineIO._test:
        warn(s)
    else:
        raise RuntimeError(s)
        
        
        
class ScanBase(objFuncBase):
    def __init__(self,
        decision_CSETs: List[str],
        decision_min: Union[float,List[float]],
        decision_max: Union[float,List[float]],
        n_scan: int,
        observe_RDs:  List[str],
        RD_avg_time:Optional[float] = 2,       
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = 10,
        logging_tag: Optional[str] = "",
        logging_fname: Optional[str] = None,
        init_verbose: Optional[bool] = True,
        ):
        '''
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
            )        
        
        # objective_weight must be defined first
        self.n_scan = n_scan
        self.observe_RDs = observe_RDs
        self.RD_avg_time = RD_avg_time
        self.history['observe_RDs'] = {'names':observe_RDs,
                                       'values':[]}
        
    def __call__(self,x_batch=None,
                 decision_min=None, decision_max=None, n_scan=None,
                 RD_avg_time=None, 
                 abs_z=None, 
                 callbacks=None):
        
        
        if decision_min is None:
            decision_min = self.decision_min
        if decision_max is None:
            decision_max = self.decision_max
        if n_scan is None:
            n_scan = self.n_scan
        if x_batch is None:
            x_batch = np.zeros((n_scan,len(decision_max)))
            for i in range(len(decision_max)):
                x_batch[:,i] = np.linspace(decision_min[i],decision_max[i],n_scan)
        x_batch = np.atleast_2d(x_batch)
        RD_avg_time = RD_avg_time or self.RD_avg_time
        for x in x_batch:
            self._set_decision(x)
            ave_data, _ = machineIO.fetch_data(
                                list(self.decision_RDs) + list(self.observe_RDs),
                                RD_avg_time,
                                abs_z=abs_z)
            self.history['decision_RDs' ]['values'].append(ave_data[:len(self.decision_RDs) ])
            self.history['observe_RDs']['values'].append(ave_data[ len(self.decision_RDs):])
        
        if callbacks is not None:
            for f in callbacks:
                f()
                
        return pd.DataFrame(np.hstack((self.history['decision_RDs']['values'],self.history['observe_RDs']['values'])), columns=self.history['decision_RDs']['names']+self.history['observe_RDs']['names'])