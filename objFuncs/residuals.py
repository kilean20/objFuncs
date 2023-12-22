import time
import numpy as np
from typing import List, Dict, Union, Optional
from collections import OrderedDict

from scipy.optimize import lsq_linear
from copy import deepcopy as copy

from .objFuncs import objFuncBase, _eps
from .util import warn
# from . import _global_machineIO as machineIO

       
class residualObj(objFuncBase):
    def __init__(self,
        decision_CSETs: List[str],
        decision_min: Union[float,List[float]],
        decision_max: Union[float,List[float]],
        objective_goal: Dict,
        objective_norm: Dict,
        objective_weight: Optional[Dict] = None,
                 
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = np.inf,
        logging_tag: Optional[str] = "",
        init_verbose: Optional[bool] = True, 
        called_by_child: Optional[bool] = False, 
        ):
        '''
        objective_goal: a Dict specifing goal of key=PVname, val=goal. 
                        (e.g.) objective = (value-goal)/(norm +_eps)
        objective_norm: a Dict specifing normalization factor of key=PVname, val=norm. 
                        This value effectively serves as an tolerace of corresponding objective
                        
        e.g.)
        objective_goal = { 
            'FE_MEBT:BPM_D1056:XPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1056:YPOS_RD' : 0.0,     #(mm)
                           },
        objective_norm = { 
            'FE_MEBT:BPM_D1056:XPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1056:YPOS_RD' : 1.,     
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
            init_verbose = init_verbose,
            called_by_child = True,
            )        
        
        self.objective_goal   = OrderedDict({key:val for key,val in objective_goal.items()})
        for key,goal in self.objective_goal.items():
            assert type(goal) in [int,float]
            self.objective_goal[key]=float(goal)   
        self.objective_norm = OrderedDict({key:objective_norm[key] for key in objective_goal.keys()})
        if objective_weight is None:
            self.objective_weight = OrderedDict({key:1. for key in objective_goal.keys()})
        else:
            self.objective_weight = OrderedDict({key:objective_weight[key] for key in objective_goal.keys()})
        self._objective_weight_arr = np.array(list(self.objective_weight.values()))
        if len(self._objective_weight_arr) > 1:
            self._objective_weight_arr /= self._objective_weight_arr.mean()
                
        assert self.objective_norm.keys() == self.objective_goal.keys() == self.objective_weight.keys()
 
        self._check_device_init() 
        self._get_xinit()
        self.objective_RDs = list(self.objective_goal.keys())

        self.history['objectives'] = {'names': list(self.objective_goal.keys()),
                                     'values':[]}
        self.objective_RDs = list(self.objective_goal.keys())
        self.history["objective_RDs"] = {'names': self.objective_RDs,
                                        'values':[]} 
        jac_names = []
        for i,objname in enumerate(self.objective_goal.keys()):
#             jac_names.append([])
            for j,decname in enumerate(self.decision_CSETs):
#                 jac_names[-1].append('d '+objname + '/d '+decname)
                jac_names.append('d '+objname + '/d '+decname)
#             print(jac_names[-1])
                
        self.history['jacobian'] = {'names': jac_names,
                                    'values':[]}

        if init_verbose and not called_by_child:
            self.print_class_info()
            
            
    def _calculate_objectives(self,
                              RD_data,
                             ):
        objective_goal = self.objective_goal
        objective_norm = self.objective_norm       
        objs = []
        i = 0
        for key,goal in objective_goal.items():
            objs.append((RD_data[i]-goal)/(objective_norm[key] +_eps))
            i+=1
        return np.array(objs)
    
    
    def _get_objective(self,time_span=None,abs_z=None):        
        ave_data, _ = self.machineIO.fetch_data(
                            list(self.decision_RDs) + list(self.objective_RDs),
                            time_span=time_span,
                            abs_z=abs_z)
        objs = self._calculate_objectives(ave_data[len(self.decision_RDs):]) 
        self.history['decision_RDs' ]['values'].append(ave_data[:len(self.decision_RDs) ])
        self.history['objective_RDs']['values'].append(ave_data[ len(self.decision_RDs):])
        self.history['objectives'   ]['values'].append(objs)
        super().write_log()
        return objs
                         
    def __call__(self,x,time_span=None,abs_z=None,callbacks=None):
        self._set_decision(x)
        obj = np.array(self._get_objective(time_span=time_span, abs_z=abs_z))
        if callbacks is not None:
            for f in callbacks:
                f()
        return obj*self._objective_weight_arr

    def normalize_decision(self,x):
        return 2*(x - self.decision_min)/(self.decision_max - self.decision_min) - 1
    def unnormalize_decision(self,xn):
        return 0.5*(xn + 1)*(self.decision_max - self.decision_min) + self.decision_min
    def _normalized_call(self,xn,time_span=None,abs_z=None,callbacks=None):
        return self.__call__(self.unnormalize_decision(xn),time_span=time_span,abs_z=abs_z,callbacks=callbacks)
    
    def _eval_normalized_jacobian(self,xn,use3points=True,dxn=None,callbacks=None):
        assert len(xn) == len(self.decision_CSETs)
        jac = np.zeros((len(self.objective_goal),len(xn)))
        dxn = dxn or np.ones(len(xn))*0.1
        if type(dxn) is float:
            dxn = dxn*np.ones(len(xn))   
        if len(self.history['objectives']['values']) >=1:
             ref = np.array(self.history['objectives']['values'][-1])*self._objective_weight_arr
        else:
            ref = self._normalized_call(xn,callbacks=callbacks)

        for i in range(len(xn)):
            _ = copy(xn)
            _[i] += dxn[i]
            jac[:,i] = (self._normalized_call(_,callbacks=callbacks) - ref)/dxn[i]
        if use3points:
            jac2 = np.zeros((len(self.objective_goal),len(xn)))
            for i in range(len(xn)):
                _ = copy(xn)
                _[i] -= dxn[i]
                jac2[:,i] = -(self._normalized_call(_,callbacks=callbacks) - ref)/dxn[i]
            jac = 0.5*(jac+jac2)
        
        unnormalized_jac = 2*jac/(self.decision_max[None,:] - self.decision_min[None,:])
        self.history['jacobian']['values'].append(unnormalized_jac.flatten())
        return jac, ref
    
    
    def eval_jacobian(self,x,use3points=True,dx=None,callbacks=None):
        jac, ref = self._eval_normalized_jacobian(self.normalize_decision(x0).flatten(),callbacks=callbacks)
#         unnormalized_jac = jac/(self.decision_max[None,:] - self.decision_min[None,:])
        return copy(history['jacobian']['values'][-1])
    
    
    def lsq_linear(self,
                   jac_use3points=True,
                   jac_dx = None,
                   method='trf', 
                   tol=5e-2,
                   lsq_solver=None, 
                   lsmr_tol=None, 
                   max_iter=None, 
                   verbose=0, 
                   callbacks=None, 
                  ):
        '''
        method : ‘trf’ or ‘bvls’, optional
            Method to perform minimization.
            ‘trf’ : Trust Region Reflective algorithm adapted for a linear least-squares problem. This is an 
                    interior-point-like method and the required number of iterations is weakly correlated with 
                    the number of variables.
            ‘bvls’ : Bounded-variable least-squares algorithm. This is an active set method, which requires 
                     the number of iterations comparable to the number of variables. Can’t be used when A is 
                     sparse or LinearOperator.

        tol : float, optional
            Tolerance parameter. The algorithm terminates if a relative change of the cost function is less 
            than tol on the last iteration. Additionally, the first-order optimality measure is considered:
                method='trf' terminates if the uniform norm of the gradient, scaled to account for the presence 
                             of the bounds, is less than tol.
                method='bvls' terminates if Karush-Kuhn-Tucker conditions are satisfied within tol tolerance.

        lsq_solver : {None, ‘exact’, ‘lsmr’}, optional
            Method of solving unbounded least-squares problems throughout iterations:
            ‘exact’ : Use dense QR or SVD decomposition approach. 
            ‘lsmr’ : Use scipy.sparse.linalg.lsmr iterative procedure which requires only matrix-vector product 
                     evaluations. Can’t be used with method='bvls'.
            If None (default), the solver is chosen based on type of Jacobian.

        lsmr_tol : None, float or ‘auto’, optional
            Tolerance parameters ‘atol’ and ‘btol’ for scipy.sparse.linalg.lsmr. If None (default), it is set 
            to 1e-2 * tol. If ‘auto’, the tolerance will be adjusted based on the optimality of the current 
            iterate, which can speed up the optimization process, but is not always reliable.

        verbose : {0, 1, 2}, optional
            0 : work silently (default).
            1 : display a termination report.
            2 : display progress during iterations.
        '''
        if len(self.history['decision_CSETs']['values']) > 1:
            x0 = self.history['decision_CSETs']['values'][-1]
        else:
            x0 = self.x0
        xn = self.normalize_decision(x0).flatten()
        if jac_dx is None:
            jac_dxn = None
        else:
            jac_dxn = 2*jac_dx/(self.decision_max - self.decision_min)
        jac, ref = self._eval_normalized_jacobian(xn=xn,use3points=jac_use3points,dxn=jac_dxn,callbacks=callbacks)
        bounds = [-np.ones(len(xn))-xn,np.ones(len(xn))-xn]
        result = lsq_linear(jac, -ref,
                            bounds = bounds,
                            method=method, 
                            tol=tol,
                            lsq_solver=lsq_solver, 
                            lsmr_tol=lsmr_tol, 
                            max_iter=max_iter, 
                            verbose=verbose, 
                           )
        self._normalized_call(xn+result.x,callbacks=callbacks)
        result.x = self.unnormalize_decision(xn+result.x).flatten()
        return result


    
class residualObjMultiConditional(objFuncBase):
    def __init__(self,
        decision_CSETs: List[str],
        decision_min: Union[float,List[float]],
        decision_max: Union[float,List[float]],
        objective_goal: Dict, 
        objective_norm: Dict,  
        conditional_SETs: Dict[str,List[float]],
        conditional_RDs: Optional[List[str]] = None,
        conditional_tols: Optional[List[float]] = None,
        conditional_control_cost_more: bool = True,
                 
        objective_weight: Optional[Dict] = None,
        each_condition_objective_weights: Optional[List[float]] = None,
                 
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = np.inf,
        logging_tag: Optional[str] = "",
        init_verbose: Optional[bool] = True,
        called_by_child: Optional[bool] = False, 
        ):
        '''
        conditions_SETs: 
            a OrderedDict specifing fixed self.machineIOs for defining conditions (e.g. charge state, quad scan) 
            for aggregated objective definition. 
                (e.g.) for different chage coditions (i.e. charge selector), 
                  conditional_SETs = {
                      'FS1_BBS:CSEL_D2405:CTR_MTR.VAL':[-17,17],
                      'FS1_BBS:CSEL_D2405:GAP_MTR.VAL':[10,10],
                      }
                      
        conditional_RDs:
            readback PVs for conditions_SETs.keys()  !!important: order must be consistent with conditions_SETs
        
        objective_goal: a Dict specifing goal of key=PVname, val=list of goals. 
                        The list must be length of the conditions. 
                        Otherwise the goal will be duplicated to form 
                        a list of len(conditions_SETs.keys())
                                
        objective_norm: a Dict specifing normalization factor of key=PVname, val=list of norm. 
                        The list must be length of the conditions. 
                        Otherwise the norm will be duplicated to form 
                        a list of len(conditions_SETs.keys())
            (e.g.)
                objective_goal = { 
                    'FS2_BMS:BPM_D4142:XPOS_RD' : [-0.5, 0.5]   <-- two different goals for two conditions
                    'FS2_BMS:BPM_D4142:YPOS_RD' : 0, <-- one goals for two conditions
                                   },
                objective_norm = { 
                    'FS2_BMS:BPM_D4142:XPOS_RD' : 1,
                    'FS2_BMS:BPM_D4142:YPOS_RD' : 1,
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
        
        if logging_tag is None:
            logging_tag = ""
        else:
            assert type(logging_tag) == str
        if len(logging_tag)>0:
            if logging_tag[-1] != "_":
                logging_tag += "_"
                
        self.conditional_control_cost_more = conditional_control_cost_more
        print(f"[condition_controller] ",end='')
        self.condition_controller = objFuncBase(
            decision_CSETs = list(self.conditional_SETs.keys()),
            decision_min = np.min(list(self.conditional_SETs.values()),axis=1),
            decision_max = np.max(list(self.conditional_SETs.values()),axis=1),
            decision_couplings = None,  
            decision_RDs = conditional_RDs,
            decision_tols = conditional_tols,
            history_buffer_size = history_buffer_size,
            logging_frequency = logging_frequency,
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
        
        self.n_object = len(objective_goal)
        if objective_weight is None:
            objective_weight = OrderedDict({key:1. for key in objective_goal.keys()})  
        else:
            objective_weight = OrderedDict({key:objective_weight[key] for key in objective_goal.keys()})
        if each_condition_objective_weights is None:
            self.each_condition_objective_weights = np.ones(self.n_condition)
        else:
            assert len(each_condition_objective_weights) == n_condition
            for v in each_condition_objective_weights:
                assert type(v) in [float, int]
            self.each_condition_objective_weights = each_condition_objective_weights
        self.each_condition_objective_weights /= self.each_condition_objective_weights.sum()    
        self.residualObj = []
        for i in range(self.n_condition):
            _objective_goal = _get_ith_val_from_list_in_dict(i,objective_goal)
            _objective_norm = _get_ith_val_from_list_in_dict(i,objective_norm)
            _objective_weight = _get_ith_val_from_list_in_dict(i,objective_weight)
#             self.condition_controller._set_decision(np.array(list(self.conditional_SETs.values()))[:,i])
#             time.sleep(5)
            print(f"[condition {i}] ",end='')
            self.residualObj.append(
                residualObj(
                    decision_CSETs = decision_CSETs,
                    decision_min   = decision_min,
                    decision_max   = decision_max,
                    objective_goal = _objective_goal,
                    objective_norm = _objective_norm,
                    objective_weight = _objective_weight,
                    decision_couplings = decision_couplings,
                    decision_RDs = decision_RDs,
                    decision_tols = decision_tols,
                    history_buffer_size = history_buffer_size,
                    logging_frequency = np.inf,
                    logging_tag = logging_tag + ".residual_condition"+str(i),
                    init_verbose = init_verbose,
                    )
                )
        for i,o in enumerate(self.residualObj):
            self.history["condition"+str(i)] = o.history
        self.history["condition controller"] = self.condition_controller.history
        self.history["decision_CSETs"] = self.history["condition0"]["decision_CSETs"]
        self.history["decision_RDs"]   = self.history["condition0"]["decision_RDs"]
        self.history["time"] = []
#         self.history["time"] = self.history["condition0"]["time"]

        self.objective_RDs = list(o.objective_goal.keys())
        self.history['objectives'] = {'names' : list(o.objective_goal.keys()),
                                      'values': [],
                                      }
        
        assert len(self.residualObj[0].history['objectives']['values']) == 0
        # for b in range(batch_size):
            # obj_each_cond = [np.array(o.history['objectives']['values'][-batch_size+b])*o._objective_weight_arr[None,:] for o in self.residualObj]
            # self.history['objectives']['values'].append( 
                # np.sum(self.each_condition_objective_weights[:,None,None]*np.array(obj_each_cond),axis=0) )
            # self.history["time"].append(np.min([self.history["condition"+str(i)]["time"][-batch_size+b] for i in range(self.n_condition)]))       

        if init_verbose and not called_by_child:
            self.print_class_info()
        

    def __call__(self,decision_vals,time_span=None,abs_z=None, callbacks=None, multi_batch=False):
        decision_vals = np.atleast_2d(decision_vals)
        batch_size, dim = decision_vals.shape
        nobj = len(self.residualObj[0].objective_goal)
        obj_batches = np.zeros((batch_size,self.n_condition*nobj))
        
        if self.conditional_control_cost_more:
            try:
                proximal_index = np.argsort([o.history['time'][-1] for o in self.residualObj])[::-1]
            except:
                proximal_index = range(self.n_condition)
            for i in proximal_index:
                self.condition_controller._set_decision(np.array(list(self.conditional_SETs.values()))[:,i])
                for b in range(batch_size):
                    obj_batches[b,i*nobj:(i+1)*nobj] = self.residualObj[i](decision_vals[b,:],time_span=time_span,abs_z=abs_z)
                                   
        else:
            for b in range(batch_size):
                try:
                    proximal_index = np.argsort([o.history['time'][-1] for o in self.residualObj])[::-1]
                except:
                    proximal_index = range(self.n_condition)
                for i in proximal_index:
                    self.condition_controller._set_decision(np.array(list(self.conditional_SETs.values()))[:,i])
                    obj_batches[b,i*nobj:(i+1)*nobj] = self.residualObj[i](decision_vals[b,:],time_span=time_span,abs_z=abs_z)
                        
                    
        for b in range(batch_size):
            obj_each_cond = [np.array(o.history['objectives']['values'][-batch_size+b])*o._objective_weight_arr[None,:] for o in self.residualObj]
            self.history['objectives']['values'] += list(
                np.sum(self.each_condition_objective_weights[:,None,None]*np.array(obj_each_cond),axis=0) 
            )
            self.history["time"].append(np.min([self.history["condition"+str(i)]["time"][-batch_size+b] for i in range(self.n_condition)]))
        
        if callbacks is not None:
            for f in callbacks:
                f()
        
        if batch_size == 1 and not multi_batch:
            return obj_batches[0,:]
        else:
            return obj_batches

    def normalize_decision(self,x):
        return 2*(np.atleast_2d(x) - self.decision_min[None,:])/(self.decision_max[None,:] - self.decision_min[None,:]) - 1
    def unnormalize_decision(self,xn):
        return 0.5*(np.atleast_2d(xn) + 1)*(self.decision_max[None,:] - self.decision_min[None,:]) + self.decision_min[None,:]
    def _normalized_call(self,xn, time_span=None ,abs_z=None, multi_batch=False, callbacks=None):
        return self.__call__(self.unnormalize_decision(xn),time_span=time_span,abs_z=abs_z,callbacks=callbacks, multi_batch=multi_batch)
   
    def _eval_normalized_jacobian(self,xn,use3points=True,dxn=None,callbacks=None):
        assert len(xn) == len(self.decision_CSETs)
        dxn = dxn or np.ones(len(xn))*0.1
        if type(dxn) is float:
            dxn = dxn*np.ones(len(xn))

        ref = []
        if len(self.history["condition0"]['objectives']['values']) > 0:
            if np.all(np.abs(self.normalize_decision(self.history["condition0"]["decision_CSETs"]['values'][-1]) - xn) < 0.01 ):
                for o in self.residualObj:
                    ref += list(o.history['objectives']['values'][-1]*o._objective_weight_arr)
                ref = np.array(ref)
        
        jac = np.zeros((self.n_condition*self.n_object,len(xn)))    
        if use3points:
            if len(self.history["condition0"]['objectives']['values']) > 0:
                xn_simplex = np.zeros((2*len(self.decision_CSETs),len(self.decision_CSETs)))
                ioff = 0
            else:
                xn_simplex = np.zeros((2*len(self.decision_CSETs)+1,len(self.decision_CSETs)))
                xn_simplex[0,:] = xn
                ioff = 1
            for i in range(len(self.decision_CSETs)):
                xn_simplex[ioff+i,:] = xn
                xn_simplex[ioff+i,i] -= dxn[i]
                xn_simplex[ioff+len(self.decision_CSETs)+i,:] = xn
                xn_simplex[ioff+len(self.decision_CSETs)+i,i] += dxn[i]
            yn = self._normalized_call(xn_simplex,multi_batch=True,callbacks=callbacks)
            for i in range(len(xn)): 
                jac[:,i] = (yn[ioff+len(self.decision_CSETs)+i,:] - yn[ioff+i,:])/(2*dxn[i])
                
            if len(ref) == 0:
                ref = yn[0,:]
        else:
            if len(ref) > 0:
                xn_simplex = np.zeros((len(self.decision_CSETs),len(self.decision_CSETs)))
                ioff = 0
            else:
                xn_simplex = np.zeros((len(self.decision_CSETs)+1,len(self.decision_CSETs)))
                xn_simplex[0,:] = xn
                ioff = 1
            for i in range(len(self.decision_CSETs)):
                xn_simplex[ioff+i,:] = xn
                xn_simplex[ioff+i,i] += dxn[i]              
            yn = self._normalized_call(xn_simplex,multi_batch=True,callbacks=callbacks)
            if len(ref) == 0:
                ref = yn[0,:]
            for i in range(len(xn)): 
                jac[:,i] = (yn[ioff+i,:] - ref)/dxn[i]

        unnormalized_jac = 2*jac/(self.decision_max[None,:] - self.decision_min[None,:])
        for i in range(self.n_condition):
            self.history["condition"+str(i)]['jacobian']['values'].append(unnormalized_jac[i*self.n_object:(i+1)*self.n_object].flatten())
        return jac, ref
    
    
    def eval_jacobian(self,x,use3points=True,dx=None,callbacks=None):
        jac, ref = self._eval_normalized_jacobian(self.normalize_decision(x0).flatten(),callbacks=callbacks)
        unnormalized_jac = 2*jac/(self.decision_max[None,:] - self.decision_min[None,:])
        return unnormalized_jac
    
    
    def lsq_linear(self,
                   jac_use3points=True,
                   jac_dx = None,
                   method='trf', 
                   tol=5e-2,
                   lsq_solver=None, 
                   lsmr_tol=None, 
                   max_iter=None, 
                   verbose=0, 
                   callbacks=None, 
                  ):
        '''
        method : ‘trf’ or ‘bvls’, optional
            Method to perform minimization.
            ‘trf’ : Trust Region Reflective algorithm adapted for a linear least-squares problem. This is an 
                    interior-point-like method and the required number of iterations is weakly correlated with 
                    the number of variables.
            ‘bvls’ : Bounded-variable least-squares algorithm. This is an active set method, which requires 
                     the number of iterations comparable to the number of variables. Can’t be used when A is 
                     sparse or LinearOperator.

        tol : float, optional
            Tolerance parameter. The algorithm terminates if a relative change of the cost function is less 
            than tol on the last iteration. Additionally, the first-order optimality measure is considered:
                method='trf' terminates if the uniform norm of the gradient, scaled to account for the presence 
                             of the bounds, is less than tol.
                method='bvls' terminates if Karush-Kuhn-Tucker conditions are satisfied within tol tolerance.

        lsq_solver : {None, ‘exact’, ‘lsmr’}, optional
            Method of solving unbounded least-squares problems throughout iterations:
            ‘exact’ : Use dense QR or SVD decomposition approach. 
            ‘lsmr’ : Use scipy.sparse.linalg.lsmr iterative procedure which requires only matrix-vector product 
                     evaluations. Can’t be used with method='bvls'.
            If None (default), the solver is chosen based on type of Jacobian.

        lsmr_tol : None, float or ‘auto’, optional
            Tolerance parameters ‘atol’ and ‘btol’ for scipy.sparse.linalg.lsmr. If None (default), it is set 
            to 1e-2 * tol. If ‘auto’, the tolerance will be adjusted based on the optimality of the current 
            iterate, which can speed up the optimization process, but is not always reliable.

        verbose : {0, 1, 2}, optional
            0 : work silently (default).
            1 : display a termination report.
            2 : display progress during iterations.
        '''
        if len(self.history['decision_CSETs']['values']) > 1:
            x0 = self.history['decision_CSETs']['values'][-1]
        else:
            x0 = self.x0
        xn = self.normalize_decision(x0).flatten()
        if jac_dx is None:
            jac_dxn = None
        else:
            jac_dxn = 2*jac_dx/(self.decision_max - self.decision_min)
        jac, ref = self._eval_normalized_jacobian(xn=xn,use3points=jac_use3points,dxn=jac_dxn,callbacks=callbacks)
        bounds = [-np.ones(len(xn))-xn,np.ones(len(xn))-xn]
        result = lsq_linear(jac, -ref,
                            bounds = bounds,
                            method=method, 
                            tol=tol,
                            lsq_solver=lsq_solver, 
                            lsmr_tol=lsmr_tol, 
                            max_iter=max_iter, 
                            verbose=verbose, 
                           )
        self._normalized_call(xn+result.x,callbacks=callbacks)
        result.x = self.unnormalize_decision(xn+result.x).flatten()
        return result
    

class residualObjMultiConditionalVar(residualObjMultiConditional):
    def __init__(self,
        decision_CSETs: List[str],
        decision_min: Union[float,List[float]],
        decision_max: Union[float,List[float]],
        objective_goal: Dict, 
        objective_norm: Dict,  
        conditional_SETs: Dict[str,List[float]],
        conditional_RDs: Optional[List[str]] = None,
        conditional_tols: Optional[List[float]] = None,
        conditional_control_cost_more: bool = True,

        objective_weight: Optional[Dict] = None,
        each_condition_objective_weights: Optional[List[float]] = None,         
        var_obj_weight_fraction: Optional[float] = 1.,
                 
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        history_buffer_size: Optional[int] = None,
        logging_frequency: Optional[int] = np.inf,
        logging_tag: Optional[str] = "",
        init_verbose: Optional[bool] = True,
        called_by_child: Optional[bool] = False, 
        ):
        '''
        conditions_SETs: 
            a OrderedDict specifing fixed self.machineIOs for defining conditions (e.g. charge state, quad scan) 
            for aggregated objective definition. 
                (e.g.) for different chage coditions (i.e. charge selector), 
                  conditional_SETs = {
                      'FS1_BBS:CSEL_D2405:CTR_MTR.VAL':[-17,17],
                      'FS1_BBS:CSEL_D2405:GAP_MTR.VAL':[10,10],
                      }
                      
        conditional_RDs:
            readback PVs for conditions_SETs.keys()  !!important: order must be consistent with conditions_SETs
        var_obj_weight_fraction:
            float.  1(default): consider only variation minimzation over different conditions
                    0:          no variation is considered. residualObjMultiConditionalBPMVar 
                                becomes same as residualObjMultiConditional
        
        objective_goal: a Dict specifing goal of key=PVname, val=list of goals. 
                        The list must be length of the conditions. 
                        Otherwise the goal will be duplicated to form 
                        a list of len(conditions_SETs)
                                
        objective_norm: a Dict specifing normalization factor of key=PVname, val=list of norm. 
                        The list must be length of the conditions. 
                        Otherwise the norm will be duplicated to form 
                        a list of len(conditions_SETs)
            (e.g.)
                objective_goal = { 
                    'FS2_BMS:BPM_D4142:XPOS_RD' : [-0.5, 0.5]   <-- two different goals for two conditions
                    'FS2_BMS:BPM_D4142:YPOS_RD' : 0, <-- one goals for two conditions
                                   },
                objective_norm = { 
                    'FS2_BMS:BPM_D4142:XPOS_RD' : 1,
                    'FS2_BMS:BPM_D4142:YPOS_RD' : 1,
                    },
        '''
        super().__init__(
            decision_CSETs=decision_CSETs,
            decision_min=decision_min,
            decision_max=decision_max,
            objective_goal=objective_goal,
            objective_norm=objective_norm,
            objective_weight=objective_weight,
            conditional_SETs=conditional_SETs,
            conditional_RDs=conditional_RDs,
            conditional_tols=conditional_tols,
            conditional_control_cost_more=conditional_control_cost_more,
            decision_couplings=decision_couplings,
            decision_RDs=decision_RDs,
            decision_tols=decision_tols,
            history_buffer_size=history_buffer_size,
            logging_frequency=logging_frequency,
            logging_tag=logging_tag,
            init_verbose=init_verbose, 
            called_by_child = True, 
            )      
        self.var_obj_weight_fraction = var_obj_weight_fraction or 1.
        assert 0< var_obj_weight_fraction <= 1
        names = []
        for i in range(self.n_condition-1):
            for name in objective_goal.keys():
                names.append('D'+str(i+1)+'0_'+name)
        self.history['objectives_var'] = {
                                          'names': names,
                                          'description':'variation of objectives(residuals) relative to the 0th condition. values[0].shape=(n_object*(n_condition-1))',
                                          'values':[]}
        names = []
        for i in range(self.n_condition-1):
            for name in objective_goal.keys():
                for cset in self.decision_CSETs:
                    names.append('d (D'+str(i+1)+'0 '+name + ')/d '+cset)
        self.history['jacobian_var'] = {'names': names,
                                        'description':'jacobian of variation of objectives(residuals). values[0].shape=(n_object*(n_condition-1)*n_decision)',
                                        'values':[]}
                                        
        if init_verbose and not called_by_child:
            self.print_class_info()

    def calculate_objs_var(self,objs):
        objs = np.array(objs)
        batch_size, dim = objs.shape
        ref = objs[:,:self.n_object]
        objs_var = objs[:,self.n_object:].reshape(batch_size, self.n_condition-1, self.n_object)-ref.reshape(batch_size, 1, self.n_object)
        for b in range(batch_size):
            self.history['objectives_var']['values'].append(objs_var[b,:,:].flatten())
        
        return objs_var

    def __call__(self,decision_vals,time_span=None,abs_z=None, callbacks=None, multi_batch=False):
        objs = super().__call__(decision_vals,time_span=time_span,abs_z=abs_z, multi_batch=True)
     
        objs_var = self.calculate_objs_var(objs)
        if self.var_obj_weight_fraction < 1:
            objs = np.hstack((objs*(1-self.var_obj_weight_fraction), self.var_obj_weight_fraction*objs_var.reshape(len(objs_var),-1))) 
        else:
            objs = objs_var.reshape(len(objs_var),-1)
           
        if callbacks is not None:
            for f in callbacks:
                f() 

        if len(objs_var) == 1 and not multi_batch:
            return objs[0,:]
        else:
            return objs


    def normalize_decision(self,x):
        return 2*(np.atleast_2d(x) - self.decision_min[None,:])/(self.decision_max[None,:] - self.decision_min[None,:]) - 1
    def unnormalize_decision(self,xn):
        return 0.5*(np.atleast_2d(xn) + 1)*(self.decision_max[None,:] - self.decision_min[None,:]) + self.decision_min[None,:]
    def _normalized_call(self,xn,time_span=None,abs_z=None, callbacks=None, multi_batch=False):
        return self.__call__(self.unnormalize_decision(xn),time_span=time_span,abs_z=abs_z,callbacks=callbacks,multi_batch=multi_batch)
    
    def _eval_normalized_jacobian(self,xn,use3points=True,dxn=None,callbacks=None):
        assert len(xn) == len(self.decision_CSETs)
        dxn = dxn or np.ones(len(xn))*0.1
        if type(dxn) is float:
            dxn = dxn*np.ones(len(xn))
        
        ref = []
        if len(self.history["condition0"]['objectives']['values']) > 0:
            if self.var_obj_weight_fraction < 1:
                for i,res in enumerate(self.residualObj):
                    ref += list(res.history['objectives']['values'][-1]*res._objective_weight_arr*(1-self.var_obj_weight_fraction))
            ref += list(self.history['objectives_var']['values'][-1].flatten()*self.var_obj_weight_fraction)
            ref = np.array(ref)
        
        if self.var_obj_weight_fraction < 1:
            jac = np.zeros(((2*self.n_condition-1)*self.n_object,len(xn)))
        else:
            jac = np.zeros(((  self.n_condition-1)*self.n_object,len(xn)))
            
        if use3points:
            if len(self.history["condition0"]['objectives']['values']) > 0:
                xn_simplex = np.zeros((2*len(self.decision_CSETs),len(self.decision_CSETs)))
                ioff = 0
            else:
                xn_simplex = np.zeros((2*len(self.decision_CSETs)+1,len(self.decision_CSETs)))
                xn_simplex[0,:] = xn
                ioff = 1
            for i in range(len(self.decision_CSETs)):
                xn_simplex[ioff+i,:] = xn
                xn_simplex[ioff+i,i] -= dxn[i]
                xn_simplex[ioff+len(self.decision_CSETs)+i,:] = xn
                xn_simplex[ioff+len(self.decision_CSETs)+i,i] += dxn[i]
            yn = self._normalized_call(xn_simplex, multi_batch=True,callbacks=callbacks)
            for i in range(len(xn)): 
                jac[:,i] = (yn[ioff+len(self.decision_CSETs)+i,:] - yn[ioff+i,:])/(2*dxn[i])
                
            if len(ref) == 0:
                ref = yn[0,:]            
        else:
            if len(ref) > 0:
                xn_simplex = np.zeros((len(self.decision_CSETs),len(self.decision_CSETs)))
                ioff = 0
            else:
                xn_simplex = np.zeros((len(self.decision_CSETs)+1,len(self.decision_CSETs)))
                xn_simplex[0,:] = xn
                ioff = 1
            for i in range(len(self.decision_CSETs)):
                xn_simplex[ioff+i,:] = xn
                xn_simplex[ioff+i,i] += dxn[i]              
            yn = self._normalized_call(xn_simplex, multi_batch=True,callbacks=callbacks)
            if len(ref) == 0:
                ref = yn[0,:]
            for i in range(len(xn)): 
                jac[:,i] = (yn[ioff+i,:] - ref)/dxn[i]

        unnormalized_jac = 2*jac/(self.decision_max[None,:] - self.decision_min[None,:])
        if self.var_obj_weight_fraction < 1:
            for i in range(self.n_condition):
                self.history["condition"+str(i)]['jacobian']['values'].append(unnormalized_jac[i*self.n_object:(i+1)*self.n_object].flatten())
        self.history['jacobian_var']['values'].append(unnormalized_jac[-self.n_object*(self.n_condition-1):,:].flatten())
            
        return jac, ref