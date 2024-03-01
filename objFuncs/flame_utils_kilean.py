import os
import time
import datetime

import re
import io
import warnings
import contextlib

from copy import deepcopy as copy
from typing import List, Tuple, Optional, Dict, Any, Union

import math
import numpy as np
import pandas as pd
from scipy import optimize

import matplotlib.pyplot as plt
import matplotlib.lines as lin
import matplotlib.patches as ptc

from flame_utils import ModelFlame as _ModelFlame
from collections import OrderedDict
from phantasy import MachinePortal, disable_warnings

@contextlib.contextmanager    
def suppress_outputs():
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield

disable_warnings()
_mp = MachinePortal(machine='FRIB',segment='LINAC',verbose=False);


class VM():
    def __init__(self,
                 latfile,
                 decision_CSETs,
                 objective_RDs,
                 conditional_SETs = None,
                 fm_moment0 = None,
                 fm_moment1 = None,
                 Dnum_from = 1001,
                 Dnum_to = 9999,
                 ):
        
        self.decision_CSETs = decision_CSETs
        self.objective_RDs  = objective_RDs
        self.conditional_SETs = conditional_SETs

        if not os.path.exists(latfile):
            print(f"'{latfile}' does not exist in current working path.")
            module_path = os.path.dirname(os.path.abspath(__file__))
            latfile = os.path.join(module_path,'FLAME_lat',os.path.basename(latfile))
            if os.path.exists(latfile):
                print(f"FLAME lattice file found from lat repo: \n    {latfile}")
            else:
                raise ValueError(f"{latfile} does not exist in lat repo neither....")
        
        self.fm = ModelFlame(latfile)
        self._update_flame(self.fm)
        
    def _update_flame(self,
        fm,
        fm_moment0 = None,
        fm_moment1 = None,
        ):
        
        self.fm = fm
        if fm_moment0:
            self.fm.bmstate.moment0 = fm_moment0
        if fm_moment1:
            self.fm.bmstate.moment1 = fm_moment1
        self.decision_CSETs_fm_elems = get_FLAMEelem_from_PVs(self.decision_CSETs,self.fm)
        self.decision_CSETs_fm_names = [elem['properties']['name'] for elem in self.decision_CSETs_fm_elems]
        self.decision_CSETs_mp_elems = get_MPelem_from_PVs(self.decision_CSETs)
        self.decision_CSETs_MP_FLAME_fields = get_MP_FLAME_fields_from_PVs(self.decision_CSETs, self.fm)
        self.x0 = []
        self.x0_fm = []
        for fm_elem, mp_elem, field in zip(self.decision_CSETs_fm_elems,
                                           self.decision_CSETs_mp_elems,
                                           self.decision_CSETs_MP_FLAME_fields):
            
            x0_fm = fm_elem['properties'][field[1]]
            self.x0.append(wrap_mp_elem_convert(mp_elem, x0_fm, from_field=field[1], to_field=field[0]))
            self.x0_fm.append(x0_fm)
        
        self.objective_RDs_fm_elems = get_FLAMEelem_from_PVs(self.objective_RDs,self.fm)
        self.objective_RDs_fm_names = [elem['properties']['name'] for elem in self.objective_RDs_fm_elems]
        
        self.objective_RDs_fm_collect_data_attributes = get_FLAME_collect_data_attributes(self.objective_RDs, self.fm)
        self.last_elem_index = max([fm_elem['index'] for fm_elem in self.objective_RDs_fm_elems])
        
        if self.conditional_SETs is not None:
            fm_elems = get_FLAMEelem_from_PVs(self.conditional_SETs.keys(),self.fm)
            fields = get_MP_FLAME_fields_from_PVs(self.conditional_SETs.keys(), self.fm)
            mp_elems = get_MPelem_from_PVs(self.conditional_SETs.keys())
            for value, fm_elem, mp_elem, field in zip(self.conditional_SETs.values(),
                                                   fm_elems,
                                                   mp_elems,
                                                   fields):
                self.fm.reconfigure(fm_elem['index'], {field[1]: wrap_mp_elem_convert(mp_elem, value ,from_field=field[0],to_field=field[1])})
            
        
    def __call__(self,x):
        
        for x_, fm_elem, mp_elem, field in zip(x,
                                               self.decision_CSETs_fm_elems,
                                               self.decision_CSETs_mp_elems,
                                               self.decision_CSETs_MP_FLAME_fields):
            self.fm.reconfigure(fm_elem['index'], {field[1]: wrap_mp_elem_convert(mp_elem, x_,from_field=field[0],to_field=field[1])})
        r,s = self.fm.run(monitor='all',to_element=self.last_elem_index)
        r = self.fm.collect_data(r,'pos',*set(self.objective_RDs_fm_collect_data_attributes))
        output = []
        for fm_elem,attr in zip(self.objective_RDs_fm_elems, self.objective_RDs_fm_collect_data_attributes):
            output.append(r[attr][fm_elem['index']])
        
#         for x0_, fm_elem, mp_elem, field in zip(self.x0_fm,
#                                                 self.decision_CSETs_fm_elems,
#                                                 self.decision_CSETs_mp_elems,
#                                                 self.decision_CSETs_MP_FLAME_fields):
#             self.fm.reconfigure(fm_elem['index'], {field[1]: x0_})
            
        return output
            
        
class ModelFlame(_ModelFlame):
    def __init__(self,lat_file=None, **kws):
        super().__init__(lat_file=lat_file,**kws)
        i=0
        z=0
        while(True):
            try:
                elem = self.get_element(index=i)[0]
                elem_prop = elem['properties']
            except:
                break
            if 'L' in elem_prop:
                L = elem_prop['L']
            else:
                L = 0
#             elem_prop['s0']=s
#             elem_prop['s1']=s+L
            self.reconfigure(i,{'z':z,
                                'L':L})
            z += L
            i += 1
        
        self.types = self.get_all_types()
        self.names = self.get_all_names()
        self.bpm_indices = self.find(type='bpm')
        self.quad_indices = self.find(type='quadrupole')
        self.sbend_indices = self.find(type='sbend')
            
    def get_df_by_type(self,type):
        ind = self.get_index_by_type(type)[type]
        elems = self.get_element(index = ind)
        elem_props = []
        for i,elem in enumerate(elems):
            elem_props.append(elem['properties'])
            elem_props[-1]['index'] = ind[i]
            
        return pd.DataFrame(copy(elem_props)).set_index('index')
    
    def collect_data(self, result, *arg, **kwargs):
        if 'BPM_misalign_effect' in kwargs:
            BPM_misalign_effect = kwargs.pop('BPM_misalign_effect')
        else:
            BPM_misalign_effect = False
            
        data = super().collect_data(result,*arg, **kwargs)
        if BPM_misalign_effect:
            for i,r in enumerate(result):
                if r[0] in self.bpm_indices:
                    bpm = self.get_element(index=r[0])[0]['properties']
                    try:
                        data['xcen'][i] -= bpm['dx']
                    except:
                        pass
                    try:
                        data['ycen'][i] -= bpm['dy']
                    except:
                        pass
                    try:
                        data['xcen_all'][i] -= bpm['dx']
                    except:
                        pass
                    try:
                        data['ycen_all'][i] -= bpm['dy']
                    except:
                        pass
        return data
    
    def zero_orbtrim(self):
        iorbtrim = self.find(type='orbtrim')
        for i in iorbtrim:
            elem = self.get_element(index=i)[0]['properties']
            if "tm_xkick" in elem:
                self.reconfigure(i  ,{'tm_xkick':0.})
            if "tm_ykick" in elem:
                self.reconfigure(i  ,{'tm_ykick':0.})
    
    def plot_lattice(self,starting_offset=0,start=None, end=None, ymin=0.0, ymax=1.0, 
                 legend=True, ax = None):
        tmp = _plot_lattice(self.machine)
        tmp(starting_offset=starting_offset,
            start=start, 
            end=end, 
            ymin=ymin, 
            ymax=ymax, 
            legend=legend, 
            ax = ax)


class _plot_lattice():
    def __init__(self, M):
        self.M = M
        self.types = {'rfcavity':   {'flag':True, 'name':'rfcavity', 'color':'orange', 'scale':0.0},
                      'solenoid':   {'flag':True, 'name':'solenoid', 'color':'red',    'scale':0.0},
                      'quadrupole': {'flag':True, 'name':'quad',     'color':'purple', 'scale':0.0},
                      'sextupole':  {'flag':True, 'name':'sext',     'color':'navy',   'scale':0.0},
                      'sbend':      {'flag':True, 'name':'bend',     'color':'green',  'scale':0.0},
                      'equad':      {'flag':True, 'name':'e-quad',   'color':'blue',   'scale':0.0},
                      'edipole':    {'flag':True, 'name':'e-dipole', 'color':'lime',   'scale':0.0},
                      'bpm':        {'flag':True, 'name':'bpm',      'color':'m',      'scale':0.0},
                      'orbtrim':    {'flag':True, 'name':'corr',     'color':'black',  'scale':0.0},
                      'stripper':   {'flag':True, 'name':'stripper', 'color':'y',      'scale':0.0},
                      'marker':     {'flag':True, 'name':'pm',       'color':'c',      'scale':0.0}
                      }
                      
    def _get_scl(self, elem):
        """Get arbital strength of the optical element.
        """
        try:
            if elem['type'] == 'rfcavity':
                scl = elem['scl_fac']*np.cos(2.0*np.pi*elem['phi']/360.0)
            elif elem['type'] == 'solenoid':
                scl = elem['B']
            elif elem['type'] == 'quadrupole':
                scl = elem['B2'] if 'B2' in elem else 1.0
            elif elem['type'] == 'sextupole':
                scl = elem['B3']
            elif elem['type'] == 'sbend':
                scl = elem['phi']
            elif elem['type'] == 'edipole':
                scl = elem['phi']
            elif elem['type'] == 'equad':
                if hasattr(elem,'V'):
                    scl = elem['V']/elem['radius']**2.0
                elif hasattr(elem,'scl_fac0'):
                    scl = elem['scl_fac0']/elem['radius']**2.0
                else:
                    scl = 1.0
        except:
            scl = 0.0

        return scl
    
    def __call__(self,starting_offset=0,start=None, end=None, ymin=0.0, ymax=1.0, 
                 legend=True, legend_ncol=2, ax = None):
        
        if ax is None:
            fog,ax = plt.subplots(figsize=(10,3))
        
        ydif = ymax - ymin
        yscl = ydif
        if ydif == 0.0:
            ydif = ymax*0.1 if ymax != 0.0 else 0.1
            yscl = ydif*0.2
        ycen=ymin-0.2*ydif
        yscl=0.1*yscl
        
        pos = starting_offset
        bp = ycen
        indexes = range(len(self.M))[start:end]
        foundelm = []

        for i in indexes:
            elem = self.M.conf(i)
            try:
                dL = elem['L']
            except:
                dL = 0.0

            if elem['type'] in self.types.keys():
                info = self.types[elem['type']]

                if foundelm.count(elem['type']) == 0:
                    foundelm.append(elem['type'])
                    if legend and info['flag']:
                        ax.fill_between([0,0],[0,0],[0,0], color=info['color'], label=info['name'])

                if info['flag']:
                    if dL != 0.0:
                        bpp = bp
                        if info['scale'] != 0.0:
                            ht = yscl*self._get_scl(elem)/info['scale'] + 0.05
                        else:
                            ht = yscl*np.sign(self._get_scl(elem))

                        if elem['type'] == 'rfcavity' or elem['type'] == 'solenoid':
                            bpp = bp-yscl*0.7
                            ht = yscl*2.0*0.7

                        ax.add_patch(ptc.Rectangle((pos, bpp), dL, ht,
                                                           edgecolor='none',facecolor=info['color']))
                    else:
                        ax.add_line(lin.Line2D([pos,pos],[-yscl*0.3+bp, yscl*0.3+bp],color=info['color']))

            pos += dL

        ax.add_line(lin.Line2D([0.0, pos], [bp,bp], color='gray', zorder=-5))

        if legend:
            ax.legend(ncol=legend_ncol, loc='upper left', bbox_to_anchor=(1.01,0.99))
            
            
            
def get_MPelem_from_PVs(PVs: list, mp=_mp) -> list or None:
    """
    Retrieves MachinePortal elements from a list of PVs.

    Args:
        PVs (list): List of PV strings.
        mp: MachinePortal instance.

    Returns:
        list or None: List of MachinePortal elements corresponding to the PVs.
    """
    names = [split_name_key_from_PV(PV)[0] for PV in PVs]
    replaces = (('PSQ', 'Q'),
                 ('PSQ', 'QV'),
                 ('PSQ', 'QH'),
                 ('PSC2', 'DCH'),
                 ('PSC1', 'DCV'),)
    if mp is None:
        mp = MachinePortal(machine='FRIB', segment='LINAC')
    mp_names = mp.get_all_names()
    mp_dnums = [get_Dnum_from_pv(mp_name) for mp_name in mp_names]
    elems = []
    for name in names:
        with suppress_outputs():
            elem = mp.get_elements(name=name)
        if len(elem) == 0:
            # try replaces
            for orig, new in replaces:
                with suppress_outputs():
                    elem = mp.get_elements(name=name.replace(orig, new))
                if len(elem) > 0:
                    break
            # if still not found, get elem from matching dnum
            if len(elem) == 0:
                i = mp_dnums.index(get_Dnum_from_pv(name))
                if i >= 0:
                    with suppress_outputs():
                        elem = mp.get_elements(name=mp_names[i])
        if len(elem) == 0:
            elems.append(None)
            print(f"MachinePortal element is not found for PV: {name}")
        else:
            elems.append(elem[0])
    return elems
            
    
    
def get_FLAMEelem_from_PVs(PVs: list, fm) -> list or None:
    """
    Retrieves FLAME elements from a list of PVs.

    Args:
        PVs (list): List of PV strings.
        fm: FLAME instance.

    Returns:
        list or None: List of FLAME elements corresponding to the PVs.
    """
    names = [split_name_key_from_PV(PV)[0] for PV in PVs]
    replaces = (('PSQ', 'Q'),
                ('PSQ', 'QV'),
                ('PSQ', 'QH'),
                ('PSC2', 'DCH'),
                ('PSC1', 'DCV'),
                )

    fm_names = fm.get_all_names()
    fm_dnums = [get_Dnum_from_pv(fm_name) for fm_name in fm_names]
    elems = []
    for name in names:
        if name in fm_names:
            elem = fm.get_element(name=name)
        else:
            elem = []
        if len(elem) == 0:
            # try replaces
            for orig, new in replaces:
                name_ = name.replace(orig, new)
                if name_ in fm_names:
                    elem = fm.get_element(name=name_)
                else:
                    elem = []
                if len(elem) > 0:
                    break
            # if still not found, get elem from matching dnum
            if len(elem) == 0:
                i = fm_dnums.index(get_Dnum_from_pv(name))
                if i > 0:
                    elem = fm.get_element(name=fm_names[i])
                    print(f"FLAME element finding from name {name} was not successful. The FLAME element found based on D-number is: {elem[0]['properties']['name']}")
        if len(elem) == 0:
            elems.append(None)
            print(f"FLAME element is not found for PV: {name}")
        else:
            elems.append(elem[0])
    return elems


def get_MP_FLAME_fields_from_PVs(PVs: list, fm) -> list:
    """
    Retrieves fields from MachinePortal or FLAME based on PVs.

    Args:
        PVs (list): List of PV strings.
        fm: MachinePortal or FLAME instance.

    Returns:
        list: List of extracted fields.
    """
    names = [split_name_key_from_PV(PV)[0] for PV in PVs]
    info = {':PSC1': ('I', 'tm_ykick'),
            ':PSC2': ('I', 'tm_xkick'),
            ':PSQ' : ('I', 'B2')}
    fields = []
    for PV in PVs:
        f = None
        for key, val in info.items():
            if key in PV:
                f = val
                break
        fields.append(f)
    return fields
    
    
def wrap_mp_elem_convert(mp_elem, value, from_field, to_field):
    sign = 1
    if to_field == 'tm_xkick':
        to_field = 'TM'
        sign = -1
        assert from_field == 'I'
    elif to_field == 'tm_ykick':
        to_field = 'TM'
        assert from_field == 'I'
    if from_field == 'tm_xkick':
        from_field = 'TM'
        sign = -1
        assert to_field == 'I'
    elif from_field == 'tm_ykick':
        from_field = 'TM'
        assert to_field == 'I'
    
    output = sign*mp_elem.convert(value,from_field,to_field)
    if output is None:
        raise ValueError(f"MachinePortal convert failed to convert with from_field: {from_field} and to_field: {to_field}")
    return output
    
    
    
def get_FLAME_collect_data_attributes(PVs: list, fm) -> list:
    attributes = []
    for pv in PVs:
        if ':XPOS_RD' in pv:
            attributes.append('xcen')
        elif ':YPOS_RD' in pv:
            attributes.append('ycen')
        else: 
            raise NotImplementedError(f'cannot infer FLAME result attribute from {pv}')
    return attributes
            
    
def split_name_key_from_PV(PV: str) -> tuple:
    """
    Splits the PV into name and key components.

    Args:
        PV (str): The PV string.

    Returns:
        tuple: A tuple containing the name and key components.
    """
    # Find the index of the first colon
    first_colon_index = PV.find(':')

    if first_colon_index == 1:
        print(f"Name of PV: {PV} is not found")
        return None

    second_colon_index = PV.find(':', first_colon_index + 1)
    if second_colon_index != -1:
        return PV[:second_colon_index], PV[second_colon_index + 1:]
    else:
        return PV, None
        
        
def get_Dnum_from_pv(pv: str) -> int or None:
    """
    Extracts the D number from a PV string.

    Args:
        pv (str): The PV string.

    Returns:
        int or None: The extracted D number or None if not found.
    """
    try:
        return int(re.search(r"_D(\d{4})", pv).group(1))
    except:
        return None


def get_Dnums_from_FLAME_pv(pv: str) -> list:
    """
    Extracts D numbers from a FLAME PV string.

    Args:
        pv (str): The FLAME PV string.

    Returns:
        list: A list of extracted D numbers.
    """
    pattern = re.compile(r'_D(\d{4})')
    pattern2 = re.compile(r'_(\d{4})')
    matches = pattern.finditer(pv)
    Dnums = []
    for match in matches:
        _next = True
        Dnums.append(int(match.group()[2:]))
        start_position = match.end()
        while _next:
            match = pattern2.fullmatch(pv[start_position:start_position + 5])
            _next = bool(match)
            if _next:
                Dnums.append(int(match.group()[1:]))
                start_position += match.end()
    return Dnums
    

def sort_by_Dnum(strings):
    """
    Sort a list of PVs by dnum.
    """
    # Define a regular expression pattern to extract the 4-digit number at the end of each string
    pattern = re.compile(r'\D(\d{4})$')

    # Define a custom sorting key function that extracts the 4-digit number using the regex pattern
    def sorting_key(s):
        match = pattern.search(s)
        if match:
            return int(match.group(1))
        return 0  # Default value if no match is found

    # Sort the strings based on the custom sorting key
    sorted_strings = sorted(strings, key=sorting_key)

    return sorted_strings