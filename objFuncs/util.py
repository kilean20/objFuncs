import os
import time
import datetime
from copy import deepcopy as copy
from typing import List, Dict, Union, Optional, Callable
import re
from IPython import display
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker, HPacker

import io
import contextlib
import warnings

@contextlib.contextmanager    
def suppress_outputs():
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield

# from warnings import warn as _warn
def _warn(message, *args, **kwargs):
    return 'warning: ' +str(message) + '\n'
#     return _warn(x,stacklevel=2)  

warnings.formatwarning = _warn
def warn(x):
    return warnings.warn(x)


def elu(x):
    if x>0:
        return x
    else:
        return np.exp(x) -1


def cyclic_distance(x,y,Lo,Hi):
    x_ang = 2*np.pi*(x-Lo)/(Hi-Lo)
    y_ang = 2*np.pi*(y-Lo)/(Hi-Lo)
    return np.arccos(np.cos(y_ang-x_ang))/np.pi*0.5*(Hi-Lo)

def cyclic_mean(x,Lo,Hi):
    x_ = np.array(x)
    if x_.ndim==1 and len(x_)<1:
        return x_
    mean = np.mod(np.angle(np.mean(np.exp(1j*2*np.pi*(x_-Lo)/(Hi-Lo)),axis=0)),2*np.pi)/(2*np.pi)*(Hi-Lo)+Lo
    return mean
    
def cyclic_mean_var(x,Lo,Hi):
    x_ = np.array(x)
    if x_.ndim==1 and len(x_)<1:
        return x_, np.zeros(x_.shape)
    x_ang = 2*np.pi*(x-Lo)/(Hi-Lo)
    mean = np.mod(np.angle(np.mean(np.exp(1j*2*np.pi*(x_-Lo)/(Hi-Lo)),axis=0)),2*np.pi)/(2*np.pi)*(Hi-Lo)+Lo
    return mean, np.mean(cyclic_distance(x,mean,Lo,Hi)**2)

def cyclic_difference(x,y,Lo,Hi):
#     warn("Orientation of cyclic_distance is not well defined")
    x_ang = 2*np.pi*(x-Lo)/(Hi-Lo)
    y_ang = 2*np.pi*(y-Lo)/(Hi-Lo)
    distance = cyclic_distance(x,y,Lo,Hi)        
    return distance*np.sign(np.sin(y_ang-x_ang))

def is_picklable(obj):
    try:
        pickle.dumps(obj)
        return obj
    except:
        return 'not picklable'
    
def get_picklable_items_from_dict(dic):
#     return {attr: value for attr, value in dic.items() if is_picklable(value)}
    new_dic = {}
    for attr, value in dic.items():
        try:
            new_dic[attr] = is_picklable(copy(value))
        except:
            continue
    if 'self' in new_dic.keys():
        del new_dic['self']
    if '__class__' in new_dic.keys():
        del new_dic['__class__']
    return new_dic

    
def _get_class_hierarchy(cls):
    class_names = [cls.__name__]
    # Get names of parent classes recursively
    for base_class in cls.__bases__:
        class_names.extend(_get_class_hierarchy(base_class))
    return class_names

def get_class_hierarchy(cls_self):
    return _get_class_hierarchy(cls_self.__class__)[:-1]


def print_nested_dict(data, indent=0, elements_per_line=3):
    for key, value in data.items():
        if isinstance(key,str):
            if key[0] == '_':
                continue
        if isinstance(value, dict):
            if len(value) > 1:
                print("  " * indent + f"{key}:")
                print_nested_dict(value, indent + 2, elements_per_line)
            else:
                print("  " * indent + f"{key}: ",end='')
                print({key:val for key,val in value.items()})
#                 print_nested_dict(value, indent + 2, elements_per_line)
        elif isinstance(value, list):
            # Convert the list to a NumPy array
            np_array = np.array(value)
            
            if np_array.size > 8:
                dtype = np_array.dtype
                if dtype == 'object':
                    dtype = value[0].__class__.__name__
                print("  " * indent + f"{key} : list of shape {np_array.shape} and type {dtype}")
            else:
                print("  " * indent + f"{key}: [",end="")
                for i in range(0, len(value), elements_per_line):
                    elements = value[i:i + elements_per_line]
                    print()
                    print("  " * (indent + 2) + ', '.join(map(str, elements)),end="")
                print("]")
        elif isinstance(value, np.ndarray) and value.size > 8:
            dtype = value.dtype
            if dtype == 'object':
                dtype = value[0].__class__.__name__
            print("  " * indent + f"{key} : array of shape {value.shape} and type {dtype}")
        elif value is not None or value != "":
            print("  " * indent + f"{key}: {value}")


def _convert_to_sublists(string_list):
    sublists = []
    current_sublist = []
    current_length = 0
    
    for string in string_list:
        if current_length + len(string) <= 64:
            current_sublist.append(string)
            current_length += len(string)
        else:
            sublists.append(current_sublist)
            current_sublist = [string]
            current_length = len(string)
    
    if current_sublist:
        sublists.append(current_sublist)
    
    return sublists



def _multicolor_ylabel(ax, list_of_strings, list_of_colors=None, offset=-0.02, anchorpad=0, **kw):
    """This function creates axes labels with multiple colors.
    ax specifies the axes object where the labels should be drawn.
    list_of_strings is a list of all the text items.
    list_of_colors is a corresponding list of colors for the strings.
    anchorpad is the padding for the anchored offset box.
    Additional keyword arguments can be passed to the text properties."""

    if list_of_colors is None:
        list_of_colors = ['C' + str(i) for i in range(len(list_of_strings))]

    _list_of_strings = _convert_to_sublists(list_of_strings)
    
    n = 0
    vboxes = []
    for i in range(len(_list_of_strings)):
        _list_of_colors = list_of_colors[n:n+len(_list_of_strings[i])] 
        n += len(_list_of_strings[i])
        boxes = [TextArea(text, textprops=dict(color=color, ha='left', va='bottom', rotation=90, **kw))
                 for text, color in zip(_list_of_strings[i][::-1], _list_of_colors)]

        vboxes.append( VPacker(children=boxes, align="left", pad=0, sep=5) )

    # Create a VPacker to arrange the hbox vertically
    ybox = HPacker(children=vboxes, align="left", pad=0, sep=5)

    anchored_ybox = AnchoredOffsetbox(
        loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(offset-0.02*len(vboxes), 0),
        bbox_transform=ax.transAxes, borderpad=0.
    )
    ax.add_artist(anchored_ybox)
    
    return len(vboxes)



class plot_obj_history:
    def __init__(self,
                 history: Dict,
                 keys: Optional[List[List[str]]] = None,
                 inline = True,
                 hdisplay = None,
                 fig = None,
                 ax = None,
                 xaxis = None,
                 xlabel = None,
                 add_y_data = None,
                 add_y_label = None,
                 title = None,
                ):
        self.hist = history
        if keys is None:
            self.keys = [self.hist['names']]
        else:
            if type(keys[0]) is list:
                self.keys = [key for key in keys if len(key)>0]
            else:
                self.keys = [keys]
        self.n_yaxis = len(self.keys)
        self.index = [[self.hist['names'].index(k) for k in self.keys[i]] for i in range(self.n_yaxis)]
        self.colors = []
        self.inline = inline
        n=0
        for i in range(self.n_yaxis):
            self.colors.append(['C'+str(n+j) for j in range(len(self.keys[i]))])
            n+=len(self.keys[i])
        self.xaxis = xaxis 
        self.xlabel = xlabel
        self.hdisplay = hdisplay
        self.fig = fig
        self.ax = ax
        self._plot_constructed = False
        
        self.add_y_data = add_y_data
        self.add_y_label = add_y_label
        
        self.title = title
        
        
    def _construct_plot(self,
                        hdisplay=None,
                        fig=None,
                        ax=None,
                        inline=False
                        ):        
        self.ax = ax or self.ax
        self.fig = fig or self.fig
        self.hdisplay = hdisplay or self.hdisplay
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(15,6))
            if inline:
                self.hdisplay = display.display("",display_id=True)
            
        offset = 0
        self.axes = []
        for i,keys in enumerate(self.keys):
            if i==0:
                ax = self.ax
                ax.tick_params(axis="y",direction="in", pad=-30)
                ax.grid()
            else:
                ax = self.ax.twinx()
                ax.spines['right'].set_position(('axes', offset))
#                 offset -= 0.02
            for j,k in enumerate(keys):
                ax.plot(self._xaxis, self.values[:,self.index[i][j]], color=self.colors[i][j],label=k)
            ncol = _multicolor_ylabel(self.ax,self.keys[i],self.colors[i],offset=offset)
            offset -= 0.02*ncol + 0.06
            self.axes.append(ax)
        if self.add_y_data is not None:
            ax = self.ax.twinx()
            ax.plot(self._xaxis, self.add_y_data, color='k')
            ax.set_ylabel(self.add_y_label,color='k')
            self.axes.append(ax)
        self.ax.set_xlabel(self.xlabel)
        if self.xlabel in ['t','time','TIME','T']:
            self.fig.autofmt_xdate(rotation=45)
            
        self.ax.set_title(self.title)
        self.fig.tight_layout()
        
        if inline:
            self.hdisplay.update(self.fig)
        else:
            self.fig.show()
            self.fig.canvas.draw()

    def _update_plot(self,inline=False):
#         self.values = np.atleast_2d(self.hist['values']).T
#         if self.xaxis is None:
#             self.xaxis = np.arange(len(self.values[:,0]))
#         else:
#             self.xaxis = self.xaxis
#         xmin = np.min(self.xaxis)
#         xmax = np.max(self.xaxis)
#         
#         print("self.xaxis.shape",self.xaxis.shape)
        for i,keys in enumerate(self.keys):
            ax = self.axes[i]
            for j,k in enumerate(keys):
                line = ax.lines[j]
#                 print("self.values[self.index[i][j],:].shape",self.values[self.index[i][j],:].shape)
                line.set_xdata(self._xaxis)
                line.set_ydata(self.values[:,self.index[i][j]])
            ax.relim()
            ax.autoscale_view()
#             ymin = np.min(self.values[:,self.index[i]])
#             ymax = np.max(self.values[:,self.index[i]])
#             ax.set_xlim(-xmax*0.05,xmax)   
#             ax.set_ylim(ymin,ymax)   

        if self.add_y_data is not None:
            ax = self.axes[-1]
            line = ax.lines[0]
            line.set_xdata(self._xaxis)
            line.set_ydata(self.add_y_data)
            ax.relim()
            ax.autoscale_view()

#         self.ax.relim()
#         self.ax.autoscale_view()
#         self.fig.canvas.flush_events()
#         self.fig.canvas.draw()
        if inline:
            self.hdisplay.update(self.fig)
        else:
            self.fig.show()
            self.fig.canvas.draw()
          
    def close(self):
        plt.close(self.fig)
        self._plot_constructed = False
            
        
    def __call__(self,
                hdisplay=None,
                fig=None,
                ax=None,
                inline=False,
                ):
        inline = inline or self.inline
        self.values = np.array(self.hist['values'])
        if len(self.values) < 2:
            return
        if self.xaxis is None:
            self._xaxis = np.arange(len(self.values[:,0]))
        else:
            self._xaxis = self.xaxis
        if self._plot_constructed:
            self._update_plot(inline=inline)
        else:
            self._construct_plot(inline=inline)
            self._plot_constructed = True

            
            
class plot_multi_obj_history(plot_obj_history):
    def __init__(self,
                 histories: List[Dict],
                 history_labels: List[str],
                 keys: Optional[List[List[str]]] = None,
                 inline = True,
                 hdisplay = None,
                 fig = None,
                 ax = None,
                 xaxis = None,
                 xlabel = None,
                 add_y_data = None,
                 add_y_label = None,
                 title = None,
                ):
        
        self.names = []
        self.history = {"names": [], "values": []}
        for hist, label in zip(histories,history_labels):
            for name in hist['names']:
                self.history["names"].append(label+' '+name)
        self.histories = histories
        self.update_histories()
        
        new_keys = None
        if keys is not None:
            new_keys = []
            for i in range(len(keys)):
                new_keys.append([])
                for j in range(len(keys[i])):
                    for label in history_labels:
                        new_keys[-1].append(label+' '+keys[i][j])
                    
        super().__init__( 
            history = self.history,
            keys = new_keys,
            inline = inline,
            hdisplay = hdisplay,
            fig = fig,
            ax = ax,
            xaxis = xaxis,
            xlabel = xlabel,
            add_y_data = add_y_data,
            add_y_label = add_y_label,
            title = title,
        )
        
    def update_histories(self):
        self.history["values"] = []
        nrow = len(self.histories[0]["values"])
        for i in range(nrow):
            row_values = []
            for d in self.histories:
                row_values.extend(d["values"][i])
            self.history["values"].append(row_values)

    def __call__(self,
                hdisplay=None,
                fig=None,
                ax=None,
                inline=False,
                ):
        self.update_histories()
        super().__call__(
            hdisplay=hdisplay,
            fig=fig,
            ax=ax,
            inline=inline)    

            
def parse_operation_txt(txt):
    operators = set("+-*/()")  # Define the supported operators
    names = [char for char in txt if char.isalpha()]  # Extract data names
    operations = ''.join(char if char in operators else ' ' for char in txt)  # Extract operators
    operations = operations.replace(' ', '')  # Remove spaces
    return names, operations


def calculate_parsed_operation(txt,names,data,operations):
    txt = copy(txt)
    for i, name in enumerate(names):
        txt = data_string.replace(name, '('+str(data[i])+')')
    result = eval(txt)  # Evaluate the final expression
    return result





def read_BPMoverview_snapshot(
    fname,
    XPOS = False,
    YPOS = False,
    PHASE= True,
    Dnum_from = 1000,
    Dnum_to = 1400,
    Dnums = None
    ):
    try:
        with open(fname,'r') as f:
            lines = f.readlines()
    except:
        path = '/files/shared/phyapps-operations/data/bpm_overview/snapshots/'
        with open(path+fname,'r') as f:
            lines = f.readlines()  
        for i,line in enumerate(lines):
            if 'BPM DATA' in line:
                lines = lines[i+2:]
                lines[0] = lines[0][2:]
                break
            
    with open('tmp.csv','w') as f:
        f.writelines(lines)
    df = pd.read_csv('tmp.csv',delimiter='\t')
    os.remove('tmp.csv')
    
    
    if Dnums is not None:
        ibpms = []
        for i,bpm in enumerate(bpm_names):
            for d in Dnums[len(ibpms):]:
                if d == get_dnum_from_pv(bpm):
                    ibpms.append(i)
    else:
        ibpms = [i for i in range(len(bpm_names)) if Dnum_from <= get_dnum_from_pv(bpm_names[i]) <=Dnum_to  ]
   
    dic = {}
    if XPOS:
        vals = df['X'].values
        dic.update({bpm_names[i]+':XPOS_RD':float(vals[i]) for i in ibpms})
    if YPOS:
        vals = df['Y'].values
        dic.update({bpm_names[i]+':YPOS_RD':float(vals[i]) for i in ibpms})
    if PHASE:
        vals = df['PHASE'].values
        dic.update({bpm_names[i]+':PHASE_RD':float(vals[i]) for i in ibpms})
    return dic
            

    
def get_dnum_from_pv(pv):
    return int(re.search(r"_D(\d+)",pv).group(1))
    
    

def get_MEBT_objective_goal_from_BPMoverview(fname):
    try:
        with open(fname,'r') as f:
            lines = f.readlines()
    except:
        path = '/files/shared/phyapps-operations/data/bpm_overview/snapshots/'
        if path not in fname:
            warn(path + ' is not in file name. Searching from the BPM snapshot path.....')
        with open(path+fname,'r') as f:
            lines = f.readlines() 
    lines = lines[340:]
    for i,line in enumerate(lines):
        if 'BPM DATA' in line:
            break
            
    def float_vec(list_of_str):
        return [float(s) for s in list_of_str]
        
    MEBT_BPM_vals = [float_vec(line.split()) for line in lines[i+3:i+6]]
    
    return { 
            'FE_MEBT:BPM_D1056:XPOS_RD' : MEBT_BPM_vals[0][3],
            'FE_MEBT:BPM_D1056:YPOS_RD' : MEBT_BPM_vals[0][4],
            'FE_MEBT:BPM_D1056:PHASE_RD': MEBT_BPM_vals[0][1],
            'FE_MEBT:BPM_D1056:MAG_RD'  : {'more than': None},
            'FE_MEBT:BPM_D1072:XPOS_RD' : MEBT_BPM_vals[1][3],
            'FE_MEBT:BPM_D1072:YPOS_RD' : MEBT_BPM_vals[1][4],
            'FE_MEBT:BPM_D1072:PHASE_RD': MEBT_BPM_vals[1][1],
            'FE_MEBT:BPM_D1072:MAG_RD'  : {'more than': None},
            'FE_MEBT:BPM_D1094:XPOS_RD' : MEBT_BPM_vals[2][3],
            'FE_MEBT:BPM_D1094:YPOS_RD' : MEBT_BPM_vals[2][4],
            'FE_MEBT:BPM_D1094:PHASE_RD': MEBT_BPM_vals[2][1],
            'FE_MEBT:BPM_D1094:MAG_RD'  : {'more than': None},
            'FE_MEBT:BCM_D1055:AVGPK_RD/FE_LEBT:BCM_D0989:AVGPK_RD': {'more than': 1.00},
            'FE_MEBT:FC_D1102:PKAVG_RD': {'more than': None},
           } 
    
    

bpm_names = [
 'FE_MEBT:BPM_D1056',
 'FE_MEBT:BPM_D1072',
 'FE_MEBT:BPM_D1094',
 'FE_MEBT:BPM_D1111',
 'LS1_CA01:BPM_D1129',
 'LS1_CA01:BPM_D1144',
 'LS1_WA01:BPM_D1155',
 'LS1_CA02:BPM_D1163',
 'LS1_CA02:BPM_D1177',
 'LS1_WA02:BPM_D1188',
 'LS1_CA03:BPM_D1196',
 'LS1_CA03:BPM_D1211',
 'LS1_WA03:BPM_D1222',
 'LS1_CB01:BPM_D1231',
 'LS1_CB01:BPM_D1251',
 'LS1_CB01:BPM_D1271',
 'LS1_WB01:BPM_D1286',
 'LS1_CB02:BPM_D1295',
 'LS1_CB02:BPM_D1315',
 'LS1_CB02:BPM_D1335',
 'LS1_WB02:BPM_D1350',
 'LS1_CB03:BPM_D1359',
 'LS1_CB03:BPM_D1379',
 'LS1_CB03:BPM_D1399',
 'LS1_WB03:BPM_D1413',
 'LS1_CB04:BPM_D1423',
 'LS1_CB04:BPM_D1442',
 'LS1_CB04:BPM_D1462',
 'LS1_WB04:BPM_D1477',
 'LS1_CB05:BPM_D1486',
 'LS1_CB05:BPM_D1506',
 'LS1_CB05:BPM_D1526',
 'LS1_WB05:BPM_D1541',
 'LS1_CB06:BPM_D1550',
 'LS1_CB06:BPM_D1570',
 'LS1_CB06:BPM_D1590',
 'LS1_WB06:BPM_D1604',
 'LS1_CB07:BPM_D1614',
 'LS1_CB07:BPM_D1634',
 'LS1_CB07:BPM_D1654',
 'LS1_WB07:BPM_D1668',
 'LS1_CB08:BPM_D1677',
 'LS1_CB08:BPM_D1697',
 'LS1_CB08:BPM_D1717',
 'LS1_WB08:BPM_D1732',
 'LS1_CB09:BPM_D1741',
 'LS1_CB09:BPM_D1761',
 'LS1_CB09:BPM_D1781',
 'LS1_WB09:BPM_D1796',
 'LS1_CB10:BPM_D1805',
 'LS1_CB10:BPM_D1825',
 'LS1_CB10:BPM_D1845',
 'LS1_WB10:BPM_D1859',
 'LS1_CB11:BPM_D1869',
 'LS1_CB11:BPM_D1889',
 'LS1_CB11:BPM_D1909',
 'LS1_WB11:BPM_D1923',
 'LS1_BTS:BPM_D1967',
 'LS1_BTS:BPM_D1980',
 'LS1_BTS:BPM_D2027',
 'LS1_BTS:BPM_D2054',
 'LS1_BTS:BPM_D2116',
 'LS1_BTS:BPM_D2130',
 'FS1_CSS:BPM_D2212',
 'FS1_CSS:BPM_D2223',
 'FS1_CSS:BPM_D2248',
 'FS1_CSS:BPM_D2278',
 'FS1_CSS:BPM_D2313',
 'FS1_CSS:BPM_D2369',
 'FS1_CSS:BPM_D2383',
 'FS1_BBS:BPM_D2421',
 'FS1_BTS:BPM_D2424',
 'FS1_SEE:BPM_D2449',
 'FS1_BBS:BPM_D2466',
 'FS1_BTS:BPM_D2467',
 'FS1_BTS:BPM_D2486',
 'FS1_SEE:BPM_D2487',
 'FS1_BMS:BPM_D2502',
 'FS1_BMS:BPM_D2537',
 'FS1_BMS:BPM_D2587',
 'FS1_BMS:BPM_D2600',
 'FS1_BMS:BPM_D2665',
 'FS1_BMS:BPM_D2690',
 'FS1_BMS:BPM_D2702',
 'LS2_WC01:BPM_D2742',
 'LS2_WC02:BPM_D2782',
 'LS2_WC03:BPM_D2821',
 'LS2_WC04:BPM_D2861',
 'LS2_WC05:BPM_D2901',
 'LS2_WC06:BPM_D2941',
 'LS2_WC07:BPM_D2981',
 'LS2_WC08:BPM_D3020',
 'LS2_WC09:BPM_D3060',
 'LS2_WC10:BPM_D3100',
 'LS2_WC11:BPM_D3140',
 'LS2_WC12:BPM_D3180',
 'LS2_WD01:BPM_D3242',
 'LS2_WD02:BPM_D3304',
 'LS2_WD03:BPM_D3366',
 'LS2_WD04:BPM_D3428',
 'LS2_WD05:BPM_D3490',
 'LS2_WD06:BPM_D3552',
 'LS2_WD07:BPM_D3614',
 'LS2_WD08:BPM_D3676',
 'LS2_WD09:BPM_D3738',
 'LS2_WD10:BPM_D3800',
 'LS2_WD11:BPM_D3862',
 'LS2_WD12:BPM_D3924',
 'FS2_BTS:BPM_D3943',
 'FS2_BTS:BPM_D3958',
 'FS2_BTS:BPM_D4006',
 'FS2_BBS:BPM_D4019',
 'FS2_BBS:BPM_D4054',
 'FS2_BBS:BPM_D4087',
 'FS2_BMS:BPM_D4142',
 'FS2_BMS:BPM_D4164',
 'FS2_BMS:BPM_D4177',
 'FS2_BMS:BPM_D4216',
 'FS2_BMS:BPM_D4283',
 'FS2_BMS:BPM_D4326',
 'LS3_WD01:BPM_D4389',
 'LS3_WD02:BPM_D4451',
 'LS3_WD03:BPM_D4513',
 'LS3_WD04:BPM_D4575',
 'LS3_WD05:BPM_D4637',
 'LS3_WD06:BPM_D4699',
 'LS3_BTS:BPM_D4753',
 'LS3_BTS:BPM_D4769',
 'LS3_BTS:BPM_D4843',
 'LS3_BTS:BPM_D4886',
 'LS3_BTS:BPM_D4968',
 'LS3_BTS:BPM_D5010',
 'LS3_BTS:BPM_D5092',
 'LS3_BTS:BPM_D5134',
 'LS3_BTS:BPM_D5216',
 'LS3_BTS:BPM_D5259',
 'LS3_BTS:BPM_D5340',
 'LS3_BTS:BPM_D5381',
 'LS3_BTS:BPM_D5430',
 'LS3_BTS:BPM_D5445',
 'BDS_BTS:BPM_D5499',
 'BDS_BTS:BPM_D5513',
 'BDS_BTS:BPM_D5565',
 'BDS_BBS:BPM_D5625',
 'BDS_BTS:BPM_D5649',
 'BDS_BBS:BPM_D5653',
 'BDS_BBS:BPM_D5680',
 'BDS_FFS:BPM_D5742',
 'BDS_FFS:BPM_D5772',
 'BDS_FFS:BPM_D5790',
 'BDS_FFS:BPM_D5803',
 'BDS_FFS:BPM_D5818']
