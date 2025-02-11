# -*- coding: utf-8 -*-

"""
This is the model module for Averaged Neuron (AN) model. In this module, you 
can simulate various models based on channel and parameter modules. This module is created by modifying the module created in Yamada et al., 2021.\\

List of models
AN model: model used in Tatsuki et al., 2016.
SAN model: model used in Yoshida et al., 2018.
NAN1.0: NAN model used in Sato et al., 2024.
NAN1.1: NAN model without voltage-gated Ca2+ channels used in Sato et al., 2024.
NAN2.0 model: NAN model withNa+/K+ ATPase used in Sato et al., 2024.
FNAN model: Full-NAN model used in Sato et al., 2024.
NAN 1.0d model: NAN model used in Sato et al., 2024, but leak channels are divided into leak K+ and leak Na+ channels.
NAN 2.0d model: NAN model withNa+/K+ ATPase used in Sato et al., 2024, but leak channels are divided into leak K+ and leak Na+ channels.
FNANd model: Full-NAN model used in Sato et al., 2024, but leak channels are divided into leak K+ and leak Na+ channels.

"""


__author__ = 'Fumiya Tatsuki, Kensuke Yoshida, Tetsuya Yamada, Tomohide R. Sato, Takahiro Katsumata, Shoi Shi, Hiroki R. Ueda'
__status__ = 'Published'
__version__ = '1.0.0'
__date__ = '10 Dec 2024'


import os
import sys
"""
LIMIT THE NUMBER OF THREADS!
change local env variables BEFORE importing numpy
"""
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from collections import deque
import itertools
import numpy as np
import pandas as pd
from scipy.integrate import odeint
#from scipy.integrate import solve_ivp
from typing import Dict, Iterator, List, Optional, Union

import channels
import params


class ANmodel:

    def __init__(self, ion: bool=False, 
                 concentration: Optional[Union[Dict, str]]=None) -> None:
        self.params = params.Constants()
        self.ini = self.params.an_ini
        self.leak = channels.Leak()
        self.lek = channels.LeakK()
        self.lena = channels.LeakNa()
        self.nav = channels.NavHH()
        self.kvhh = channels.KvHH()
        self.kva = channels.KvA()
        self.kvsi = channels.KvSI()
        self.cav = channels.Cav()
        self.cavI=channels.CavI()
        self.nap = channels.NaP()
        self.kca = channels.KCa()
        self.kir = channels.KIR()
        self.ampar = channels.AMPAR()
        self.nmdar = channels.NMDAR()
        self.gabar = channels.GABAR()
        self.kna=channels.KNA()
        self.unav=channels.UNaV()
        self.nak=channels.NaK()


        self.ion = ion
        if ion:
            self.ion_params = params.Ion()
            if type(concentration) is not str:
                self.concentration = concentration
            else:
                if concentration == 'awake':
                    self.concentration = self.ion_params.awake_ion
                if concentration == 'sleep':
                    self.concentration = self.ion_params.sleep_ion
            self.set_equil_potential(concentration)
            self.nmdar = channels.NMDAR(ion=True, ex_mg=concentration['ex_mg'])

    def set_equil_potential(self, concentration: Dict) -> None:
        r: float = self.ion_params.r
        t: float = self.ion_params.t
        f: float = self.ion_params.f
        ex_na: float = concentration['ex_na']
        in_na: float = concentration['in_na']
        ex_k: float = concentration['ex_k']
        in_k: float = concentration['in_k']
        ex_cl: float = concentration['ex_cl']
        in_cl: float = concentration['in_cl']
        ex_ca: float = concentration['ex_ca']
        in_ca: float = concentration['in_ca']

        def __v(pk: float, pna: float, pcl: float, pca: float) -> float:
            ex_ion = pk * ex_k + pna * ex_na + pcl * in_cl + pca * ex_ca
            in_ion = pk * in_k + pna * in_na + pcl * ex_cl + pca * in_ca
            v = r * t / f * np.log(ex_ion/in_ion) * 1000
            return v

        vNa: float = r * t / f * np.log(ex_na/in_na) * 1000
        vK: float = r * t / f * np.log(ex_k/in_k) * 1000
        vCa: float = r * t / (f * 2) * np.log(ex_ca / in_ca) * 1000
        vL: float = __v(pk=1., pna=0.08, pcl=0.1, pca=0.)
        vAMPA: float = __v(pk=1., pna=1., pcl=0., pca=0.)
        vNMDA: float = __v(pk=1., pna=1., pcl=0., pca=1.)
        vGABA: float = r * t / f * np.log(ex_cl/in_cl) * 1000

        self.leak.set_e(new_e=vL)
        self.lek.set_e(new_e=vK)
        self.lena.set_e(new_e=vNaL)
        self.nav.set_e(new_e=vNa)
        self.kvhh.set_e(new_e=vK)
        self.kva.set_e(new_e=vK)
        self.kvsi.set_e(new_e=vK)
        self.cav.set_e(new_e=vCa)
        self.cavI.set_e(new_e=vCa)
        self.kca.set_e(new_e=vK)
        self.kir.set_e(new_e=vK)
        self.ampar.set_e(new_e=vAMPA)
        self.nmdar.set_e(new_e=vNMDA)
        self.gabar.set_e(new_e=vGABA)
        self.kna.set_e(new_e=vK)
        self.nak.set_e(new_e=vNa)
        self.unav.set_e(new_e=vNa)


    def set_vCa(self, in_ca: float) -> None:
        r: float = self.ion_params.r
        t: float = self.ion_params.t
        f: float = self.ion_params.f
        ex_ca: float = self.concentration['ex_ca']
        vCa: float = r * t / (f * 2) * np.log(ex_ca / in_ca) * 1000
        self.cav.set_e(new_e=vCa)

    def get_e(self) -> Dict:
        e_dict: Dict = {}
        e_dict['vL'] = self.leak.get_e()
        e_dict['vNa'] = self.nav.get_e()
        e_dict['vK'] = self.kvhh.get_e()
        e_dict['vCa'] = self.cav.get_e()
        e_dict['vAMPAR'] = self.ampar.get_e()
        e_dict['vNMDAR'] = self.nmdar.get_e()
        e_dict['vGABAR'] = self.gabar.get_e()
        return e_dict

    def gen_params(self) -> Dict:

        param_dict: Dict = {}

        gX_name: List[str] = ['g_leak', 'g_nav', 'g_kvhh', 'g_kva', 'g_kvsi', 
                         'g_cav', 'g_kca', 'g_nap', 'g_kir']
        gX_log: np.ndarray = 4 * np.random.rand(9) - 2  # from -2 to 2
        gX: np.ndarray = (10 * np.ones(9)) ** gX_log  # 0.01 ~ 100
        gX_itr: Iterator = zip(gX_name, gX)

        gR_name: List[str] = ['g_ampar', 'g_nmdar', 'g_gabar']
        gR_log: np.ndarray = 4 * np.random.rand(3) - 3  # from -3 to 1
        gR: np.ndarray = (10 * np.ones(3)) ** gR_log  # 0.001 ~ 10
        gR_itr: Iterator = zip(gR_name, gR)

        tCa_log: float = 2 * np.random.rand(1) + 1  # from 1 to 3
        tCa: float = 10 ** tCa_log    # 10 ~ 1000
        tCa_dict: Dict = {'t_ca': tCa}

        param_dict.update(gX_itr)
        param_dict.update(gR_itr)
        param_dict.update(tCa_dict)
        return param_dict

    def set_params(self, params: Dict) -> None:
        self.leak.set_g(params['g_leak'])
        self.nav.set_g(params['g_nav'])
        self.kvhh.set_g(params['g_kvhh'])
        self.kva.set_g(params['g_kva'])
        self.kvsi.set_g(params['g_kvsi'])
        self.cav.set_g(params['g_cav'])
        self.kca.set_g(params['g_kca'])
        self.nap.set_g(params['g_nap'])
        self.kir.set_g(params['g_kir'])
        self.ampar.set_g(params['g_ampar'])
        self.nmdar.set_g(params['g_nmdar'])
        self.gabar.set_g(params['g_gabar'])
        self.tau_ca = params['t_ca']
        

    def set_rand_params(self) -> Dict:
        new_params: Dict = self.gen_params()
        self.set_params(new_params)
        return new_params

    def set_sws_params(self) -> None:
        typ_params: Dict = params.TypicalParam().an_sws
        self.set_params(typ_params)

    def get_params(self) -> Dict:
        params: Dict = {}
        params['g_leak']: float = self.leak.get_g()
        params['g_nav']: float = self.nav.get_g()
        params['g_kvhh']: float = self.kvhh.get_g()
        params['g_kva']: float = self.kva.get_g()
        params['g_kvsi']: float = self.kvsi.get_g()
        params['g_cav']: float = self.cav.get_g()
        params['g_kca']: float = self.kca.get_g()
        params['g_nap']: float = self.nap.get_g()
        params['g_kir']: float = self.kir.get_g()
        params['g_ampar']: float = self.ampar.get_g()
        params['g.nmdar']: float = self.nmdar.get_g()
        params['g_gabar']: float = self.gabar.get_g()
        params['t_Ca']: float = self.tau_ca
        return params

    def dvdt(self, args: List[float]) -> float:
        v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, _, s_nmdar, s_gabar, ca = args
        return ((-10.0*self.params.area 
                * (self.leak.i(v)
                + self.nav.i(v, h=h_nav) 
                + self.kvhh.i(v, n=n_kvhh)
                + self.kva.i(v, h=h_kva)
                + self.kvsi.i(v, m=m_kvsi)
                + self.cav.i(v)
                + self.kca.i(v, ca=ca)
                + self.nap.i(v)
                + self.kir.i(v))
                - (self.ampar.i(v, s=s_ampar)
                + self.nmdar.i(v, s=s_nmdar)
                + self.gabar.i(v, s=s_gabar))) 
                / (10.0*self.params.cm*self.params.area))
 
    
    def dCadt(self, args: List[float]) -> float:
        v, s_nmdar, ca = args
        a_ca: float = self.params.a_ca
        area: float = self.params.area
        tau_ca: float= self.tau_ca
        dCadt: float = (- a_ca * (10.0*area*self.cav.i(v)+10.0*area*self.leak.ica(v))
                        - ca / tau_ca)
        return dCadt

    def diff_op(self, args: List[float], time: float) -> List[float]:

        v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, x_nmdar, s_nmdar, s_gabar, ca = args
        ca_args: List[float] = [v, s_nmdar, ca]



        dvdt: float = self.dvdt(args=args)
        dhNadt: float = self.nav.dhdt(v=v, h=h_nav)
        dnKdt: float = self.kvhh.dndt(v=v, n=n_kvhh)
        dhAdt: float = self.kva.dhdt(v=v, h=h_kva)
        dmKSdt: float = self.kvsi.dmdt(v=v, m=m_kvsi)
        dsAMPAdt: float = self.ampar.dsdt(v=v, s=s_ampar)
        dxNMDAdt: float = self.nmdar.dxdt(v=v, x=x_nmdar)
        dsNMDAdt: float = self.nmdar.dsdt(v=v, s=s_nmdar, x=x_nmdar)
        dsGABAdt: float = self.gabar.dsdt(v=v, s=s_gabar)
        dCadt: float = self.dCadt(args=ca_args)
        return [dvdt, 
                dhNadt, 
                dnKdt, 
                dhAdt, 
                dmKSdt,
                dsAMPAdt, 
                dxNMDAdt, 
                dsNMDAdt, 
                dsGABAdt, 
                dCadt,
                ]

    def run_odeint(self, samp_freq: int=1000, samp_len: int=20) -> (np.ndarray, Dict):

        solvetime: np.ndarray = np.linspace(1, 1000*samp_len, samp_freq*samp_len)
        s: np.ndarray
        info: Dict
        s, info = odeint(self.diff_op, self.ini, solvetime, 
                        atol=1.0e-5, rtol=1.0e-5, full_output=True)
        return s, info


class SANmodel(ANmodel):

    def __init__(self, ion: bool=False, concentration: Optional[Dict]=None) -> None:
        super().__init__(ion=ion, concentration=concentration)
        self.ini = self.params.san_ini

    def gen_params(self) -> Dict:

        param_dict: Dict = {}

        gX_name: List[str] = ['g_leak', 'g_kvhh', 'g_cav', 'g_kca', 'g_nap']
        gX_log: np.ndarray = 4 * np.random.rand(5) - 2  # from -2 to 2
        gX: np.ndarray = (10 * np.ones(5)) ** gX_log  # 0.01 ~ 100
        gX_itr: Iterator = zip(gX_name, gX)

        tCa_log: float = 2 * np.random.rand(1) + 1  # from 1 to 3
        tCa: float = 10 ** tCa_log    # 10 ~ 1000
        tCa_dict: Dict = {'t_ca': tCa}

        param_dict.update(gX_itr)
        param_dict.update(tCa_dict)
        return param_dict

    def set_params(self, params: Dict) -> None:

        self.leak.set_g(params["g_leak"])
        self.kvhh.set_g(params["g_kvhh"])
        self.cav.set_g(params["g_cav"])
        self.kca.set_g(params["g_kca"])
        self.nap.set_g(params["g_nap"])
        self.tau_ca = params["t_ca"]

    def set_sws_params(self) -> None:
        typ_params: Dict = params.TypicalParam().san_sws
        self.set_params(typ_params)

    def get_params(self) -> Dict:
        params: Dict = {}
        params['g_leak'] = self.leak.get_g()
        params['g_kvhh'] = self.kvhh.get_g()
        params['g_cav'] = self.cav.get_g()
        params['g_kca'] = self.kca.get_g()
        params['g_nap'] = self.nap.get_g()
        params['t_ca'] = self.tau_ca
        return params

    def dvdt(self, args: List[float]) -> float:
        v, n_kvhh, ca = args
        return ((-10.0*self.params.area 
                * (self.kvhh.i(v, n=n_kvhh) 
                + self.cav.i(v) 
                + self.kca.i(v, ca=ca) 
                + self.nap.i(v) 
                + self.leak.i(v))) 
                / (10.0*self.params.cm*self.params.area))

    def dCadt(self, args: List[float]) -> float:
        v, ca = args
        a_Ca: float = self.params.a_ca
        area: float = self.params.area
        tau_Ca: float = self.tau_ca
        dCadt: float = -a_Ca * (10.0*area*self.cav.i(v)+10.0*area*self.leak.ica(v)) - ca/tau_Ca
        return dCadt

    def diff_op(self, args: List[float], time: float) -> List[float]:
        v, nK, ca = args
        ca_args: List[float] = [v, ca]
        dvdt: float = self.dvdt(args=args)
        dnKdt: float = self.kvhh.dndt(v=v, n=nK)
        dCadt: float = self.dCadt(args=ca_args)
        return [dvdt, dnKdt, dCadt]





class NAN10model(ANmodel):

    def __init__(self,ion: bool=False, concentration: Optional[Dict]=None) ->None:
        super().__init__(ion=ion, concentration=concentration)
        self.ini=self.params.nan10_ini
        
    def gen_params(self) ->Dict:
        
        param_dict: Dict = {}
        gX_name: List[str] = ['g_kvhh',  'g_unav',  'g_kna',  'g_leak', 'g_cav']
        gX_log: np.ndarray=4*np.random.rand(5)-2 #from -2 to 2
        gX: np.ndarray=(10*np.ones(5)) ** gX_log  # 0.021 ~ 100
        gX_itr: Iterator = zip(gX_name, gX)
           
        tNa_log: float = 1.0 * np.random.rand(1) + 3  # from 3 to 4
        tNa: float = 10 ** tNa_log    # 1000 ~ 10000
        tNa_dict: Dict = {'t_na': tNa}
                    
        x_log: float=90*np.random.rand(1)+0#from 1 to 2
        x: float=(x_log)-45      #from -45 to +45
        x_dict: Dict={'x_na': x}

        y_log: float=90.0*np.random.rand(1)+0 #from 1 to 2
        y: float=(y_log)-45      #from -45 to +45
        y_dict: Dict={'y_na': y}
           
        param_dict.update(gX_itr)       
        param_dict.update(tNa_dict)
        param_dict.update(x_dict)
        param_dict.update(y_dict)
        
      
        return param_dict
    
    def set_params(self, params: Dict) -> None:
    
        self.kvhh.set_g(params["g_kvhh"])
        self.unav.set_g(params["g_unav"])
        self.kna.set_g(params["g_kna"])
        self.leak.set_g(params["g_leak"])
        self.cav.set_g(params["g_cav"])
        self.tau_na=params["t_na"]
        self.x=params["x_na"]
        self.y=params["y_na"]
        
     
        

    def set_sws_params(self) -> None:

        typ_params: Dict = params.TypicalParam().nan10_sws
        self.set_params(typ_params)
     
    def get_params(self) -> Dict:

        params: Dict = {}        
        params["g_kvhh"]=self.kvhh.get_g()
        params["g_unav"]=self.unav.get_g()
        params["g_kna"]=self.kna.get_g()
        params["g_leak"]=self.leak.get_g()
        params["g_cav"]=self.cav.get_g()
        params["t_na"]=self.tau_na
        params["x_na"]=self.x
        params["y_na"]=self.y
        
        return params
    
    
    def dvdt(self, args: List[float]) -> float:
        v, h_unav,n_kvhh,na = args
        x_: float=self.x
        return -10.0*self.params.area * (self.unav.i(v,h=h_unav, x=x_) + self.kna.i(v, na=na)  + self.leak.i(v)  +  self.kvhh.i(v, n=n_kvhh)+self.cav.i(v))/(10.0*self.params.cm*self.params.area)
    

    def dNadt(self, args: List[float]) -> float:
        v, h_unav, n_kvhh, na= args
        b_na: float = self.params.b_na
        area: float = self.params.area
        x_: float=self.x
        tau_na: float= self.tau_na

        dNadt: float = (- b_na * (10.0*area*self.unav.i(v,h=h_unav, x=x_)+10.0*area*self.leak.ina(v))
                        
                        - na / tau_na)
        return dNadt
                 
    def diff_op(self, args: List[float], time: float) -> List[float]:
        
        v, h_unav, n_kvhh,  na= args
        x_: float=self.x
        y_: float=self.y
                   
        na_args: List[float] = [v, h_unav, n_kvhh,  na]
            
        dvdt: float = self.dvdt(args=args)
        dh_unavdt: float= self.unav.dhdt(v=v,h=h_unav,y=y_)
        dnKdt: float = self.kvhh.dndt(v=v, n=n_kvhh)

        dNadt: float = self.dNadt(args=na_args)
        
        return [dvdt, dh_unavdt, dnKdt, dNadt]


class NAN11model(ANmodel):

    def __init__(self,ion: bool=False, concentration: Optional[Dict]=None) ->None:
        super().__init__(ion=ion, concentration=concentration)
        self.ini=self.params.nan11_ini
        
    def gen_params(self) ->Dict:

        param_dict: Dict = {}
        gX_name: List[str] = ['g_kvhh',  'g_unav',  'g_kna',  'g_leak']
        gX_log: np.ndarray=4*np.random.rand(4)-2 #from -2 to 2
        gX: np.ndarray=(10*np.ones(4)) ** gX_log  # 0.021 ~ 100
        gX_itr: Iterator = zip(gX_name, gX)
           
        tNa_log: float = 1.0 * np.random.rand(1) + 3  # from 3 to 4
        tNa: float = 10 ** tNa_log    # 1000 ~ 10000
        tNa_dict: Dict = {'t_na': tNa}
                   
        x_log: float=90*np.random.rand(1)+0#from 1 to 2
        x: float=(x_log)-45      #from -45 to +45
        x_dict: Dict={'x_na': x}

        y_log: float=90.0*np.random.rand(1)+0 #from 1 to 2
        y: float=(y_log)-45      #from -45 to +45
        y_dict: Dict={'y_na': y}
          
        param_dict.update(gX_itr)        
        param_dict.update(tNa_dict)
        param_dict.update(x_dict)
        param_dict.update(y_dict)
        
      
        return param_dict
    
    def set_params(self, params: Dict) -> None:

        self.kvhh.set_g(params["g_kvhh"])
        self.unav.set_g(params["g_unav"])
        self.kna.set_g(params["g_kna"])
        self.leak.set_g(params["g_leak"])
        self.tau_na=params["t_na"]
        self.x=params["x_na"]
        self.y=params["y_na"]
        
     
        

    def set_sws_params(self) -> None:

        typ_params: Dict = params.TypicalParam().nan11_sws
        self.set_params(typ_params)
     
    def get_params(self) -> Dict:

        params: Dict = {}        
        params["g_kvhh"]=self.kvhh.get_g()
        params["g_unav"]=self.unav.get_g()
        params["g_kna"]=self.kna.get_g()
        params["g_leak"]=self.leak.get_g()
        params["t_na"]=self.tau_na
        params["x_na"]=self.x
        params["y_na"]=self.y
        
        return params
    
    
    def dvdt(self, args: List[float]) -> float:

        v, h_unav,n_kvhh,na = args
        x_: float=self.x
        return -10.0*self.params.area * (self.unav.i(v,h=h_unav, x=x_) + self.kna.i(v, na=na)  + self.leak.i(v)  +  self.kvhh.i(v, n=n_kvhh))/(10.0*self.params.cm*self.params.area)
    

    def dNadt(self, args: List[float]) -> float:
        v, h_unav, n_kvhh, na= args
        b_na: float = self.params.b_na
        area: float = self.params.area
        x_: float=self.x
        tau_na: float= self.tau_na

        dNadt: float = (- b_na * (10.0*area*self.unav.i(v,h=h_unav, x=x_)+10.0*area*self.leak.ina(v))
                        
                        - na / tau_na)
        return dNadt
                 
    def diff_op(self, args: List[float], time: float) -> List[float]:

        v, h_unav, n_kvhh,  na= args
        x_: float=self.x
        y_: float=self.y
                   
        na_args: List[float] = [v, h_unav, n_kvhh,  na]
            
        dvdt: float = self.dvdt(args=args)
        dh_unavdt: float= self.unav.dhdt(v=v,h=h_unav,y=y_)
        dnKdt: float = self.kvhh.dndt(v=v, n=n_kvhh)
        dNadt: float = self.dNadt(args=na_args)        
        return [dvdt, dh_unavdt, dnKdt, dNadt]



    
class NAN20model(ANmodel):
    
    def __init__(self,ion: bool=False, concentration: Optional[Dict]=None) ->None:
        super().__init__(ion=ion, concentration=concentration)
        self.ini=self.params.nan20_ini
        
    def gen_params(self) ->Dict:

        param_dict: Dict = {}
        gX_name: List[str] = ['g_kvhh',  'g_unav',  'g_nak',  'g_leak', 'g_cav']
        gX_log: np.ndarray=4*np.random.rand(5)-2 #from -2 to 2
        gX: np.ndarray=(10*np.ones(5)) ** gX_log  # 0.021 ~ 100
        gX_itr: Iterator = zip(gX_name, gX)
            
        x_log: float=90*np.random.rand(1)+0#from 1 to 2
        x: float=(x_log)-45      #from -45 to +45
        x_dict: Dict={'x_na': x}

        y_log: float=90.0*np.random.rand(1)+0 #from 1 to 2
        y: float=(y_log)-45      #from -45 to +45
        y_dict: Dict={'y_na': y}
            
        param_dict.update(gX_itr)
        param_dict.update(x_dict)
        param_dict.update(y_dict)
        
      
        return param_dict
    
    def set_params(self, params: Dict) -> None:

        self.kvhh.set_g(params["g_kvhh"])
        self.unav.set_g(params["g_unav"])
        self.nak.set_g(params["g_nak"])
        self.leak.set_g(params["g_leak"])
        self.cav.set_g(params["g_cav"])
        self.x=params["x_na"]
        self.y=params["y_na"]

    def set_sws_params(self) -> None:

        typ_params: Dict = params.TypicalParam().nan20_sws
        self.set_params(typ_params)
     
    def get_params(self) -> Dict:

        params: Dict = {}        
        params["g_kvhh"]=self.kvhh.get_g()
        params["g_unav"]=self.unav.get_g()
        params["g_nak"]=self.nak.get_g()
        params["g_leak"]=self.leak.get_g()
        params["g_cav"]=self.cav.get_g()
        params["x_na"]=self.x
        params["y_na"]=self.y
        
        return params
    
    
    def dvdt(self, args: List[float]) -> float:

        v, h_unav,n_kvhh,na = args
        x_: float=self.x
        return -10.0*self.params.area * (self.unav.i(v,h=h_unav, x=x_) + self.nak.i(v, na=na)  + self.leak.i(v)  +  self.kvhh.i(v, n=n_kvhh)+self.cav.i(v))/(10.0*self.params.cm*self.params.area)
    

    def dNadt(self, args: List[float]) -> float:
        v, h_unav, n_kvhh, na= args
        b_na: float = self.params.b_na
        area: float = self.params.area
        x_: float=self.x

        dNadt: float = (- b_na * (10.0*area*self.unav.i(v,h=h_unav, x=x_)+10.0*area*self.leak.ina(v)+10.0*3*area*self.nak.i(v,na=na)))
        return dNadt
                 
    def diff_op(self, args: List[float], time: float) -> List[float]:

        v, h_unav, n_kvhh,  na= args
        x_: float=self.x
        y_: float=self.y
          
        na_args: List[float] = [v, h_unav, n_kvhh,  na]
            
        dvdt: float = self.dvdt(args=args)
        dh_unavdt: float= self.unav.dhdt(v=v,h=h_unav,y=y_)
        dnKdt: float = self.kvhh.dndt(v=v, n=n_kvhh)
        dNadt: float = self.dNadt(args=na_args)
        
        return [dvdt, dh_unavdt, dnKdt, dNadt]

   
    
class FNANmodel(ANmodel):
    
    def __init__(self,ion: bool=False, concentration: Optional[Dict]=None) ->None:
        super().__init__(ion=ion, concentration=concentration)
        self.ini=self.params.fnan_ini
        
    def gen_params(self) ->Dict:

        param_dict: Dict = {}

        gX_name: List[str] = ['g_leak', 'g_nav', 'g_kvhh', 'g_kva', 'g_kvsi', 
                         'g_cav', 'g_kca', 'g_nap', 'g_kir',   'g_unav', 'g_kna']
        gX_log: np.ndarray = 4 * np.random.rand(11) - 2  # from -2 to 2
        gX: np.ndarray = (10 * np.ones(11)) ** gX_log  # 0.01 ~ 100
        gX_itr: Iterator = zip(gX_name, gX)

        gR_name: List[str] = ['g_ampar', 'g_nmdar', 'g_gabar']
        gR_log: np.ndarray = 4 * np.random.rand(3) - 3  # from -3 to 1
        gR: np.ndarray = (10 * np.ones(3)) ** gR_log  # 0.001 ~ 10
        gR_itr: Iterator = zip(gR_name, gR)

        tCa_log: float = 2 * np.random.rand(1) + 1  # from 1 to 3
        tCa: float = 10 ** tCa_log    # 10 ~ 1000
        tCa_dict: Dict = {'t_ca': tCa}

        tNa_log: float = 1 * np.random.rand(1) + 3  # from 3 to 4
        tNa: float = 10 ** tNa_log    # 1000 ~ 10000
        tNa_dict: Dict = {'t_na': tNa}
        
        x_log: float=90*np.random.rand(1)+0#from 1 to 2
        x: float=(x_log)-45      #from -45 to +45
        x_dict: Dict={'x_na': x}

        y_log: float=90.0*np.random.rand(1)+0 #from 1 to 2
        y: float=(y_log)-45      #from -45 to +45
        y_dict: Dict={'y_na': y}
            
        param_dict.update(gX_itr)
        param_dict.update(gR_itr)
        param_dict.update(tCa_dict)
        param_dict.update(tNa_dict)
        param_dict.update(x_dict)
        param_dict.update(y_dict)
        return param_dict

    def set_params(self, params: Dict) -> None:

        self.leak.set_g(params['g_leak'])
        self.nav.set_g(params['g_nav'])
        self.kvhh.set_g(params['g_kvhh'])
        self.kva.set_g(params['g_kva'])
        self.kvsi.set_g(params['g_kvsi'])
        self.cav.set_g(params['g_cav'])
        self.kca.set_g(params['g_kca'])
        self.nap.set_g(params['g_nap'])
        self.kir.set_g(params['g_kir'])
        self.unav.set_g(params['g_unav'])
        self.kna.set_g(params['g_kna'])
        self.ampar.set_g(params['g_ampar'])
        self.nmdar.set_g(params['g_nmdar'])
        self.gabar.set_g(params['g_gabar'])
        self.tau_ca = params['t_ca']
        self.tau_na = params['t_na']
        self.x=params['x_na']
        self.y=params['y_na']

    def set_sws_params(self) -> None:

        typ_params: Dict = params.TypicalParam().fnan_sws
        self.set_params(typ_params)

    def get_params(self) -> Dict:

        params: Dict = {}
        params['g_leak']: float = self.leak.get_g()
        params['g_nav']: float = self.nav.get_g()
        params['g_kvhh']: float = self.kvhh.get_g()
        params['g_kva']: float = self.kva.get_g()
        params['g_kvsi']: float = self.kvsi.get_g()
        params['g_cav']: float = self.cav.get_g()
        params['g_kca']: float = self.kca.get_g()
        params['g_nap']: float = self.nap.get_g()
        params['g_kir']: float = self.kir.get_g()
        params['g_kna']: float=self.kna.get_g()
        params['g_unav']: flost=self_unav.get_g()
        params['g_ampar']: float = self.ampar.get_g()
        params['g.nmdar']: float = self.nmdar.get_g()
        params['g_gabar']: float = self.gabar.get_g()
        params['t_Ca']: float = self.tau_ca
        params['t_Na']: float = self.tau_na
        params['x_na']: float=self.x
        params['y_na']: float=self.y
        return params
    

    def dvdt(self, args: List[float]) -> float:

        v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, x_nmdar, s_nmdar, s_gabar, ca, na, h_unav = args
        x_: float=self.x
        return ((-10.0*self.params.area 
                * (self.leak.i(v)
                + self.nav.i(v, h=h_nav) 
                + self.kvhh.i(v, n=n_kvhh)
                + self.kva.i(v, h=h_kva)
                + self.kvsi.i(v, m=m_kvsi)
                + self.cav.i(v)
                + self.kca.i(v, ca=ca)
                + self.nap.i(v)
                + self.kir.i(v)
                +self.kna.i(v,na=na)
                +self.unav.i(v, h=h_unav, x=x_))
                - (self.ampar.i(v, s=s_ampar)
                + self.nmdar.i(v, s=s_nmdar)
                + self.gabar.i(v, s=s_gabar))) 
                / (10.0*self.params.cm*self.params.area))
    
    def dNadt(self, args: List[float]) -> float:
        v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, x_nmdar, s_nmdar, s_gabar, ca, na, h_unav = args
        x_: float=self.x
        b_na: float = self.params.b_na
        area: float = self.params.area
        tau_na: float= self.tau_na

        dNadt: float = (- b_na * (10.0*area*self.unav.i(v,h=h_unav, x=x_)+10.0*area*self.nav.i(v,h=h_nav)+10.0*area*self.nap.i(v)+10.0*area*self.leak.ina(v))- na / tau_na)
        return dNadt
    
    def dCadt(self, args: List[float]) -> float:

        v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, x_nmdar, s_nmdar, s_gabar, ca, na, h_unav = args
        a_Ca: float = self.params.a_ca
        x_: float=self.x
        area: float = self.params.area
        tau_Ca: float = self.tau_ca
        dCadt: float = -a_Ca * (10.0*area*self.cav.i(v)+10.0*area*self.leak.ica(v)) - ca/tau_Ca
        return dCadt

    def diff_op(self, args: List[float], time: float) -> List[float]:

        v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, x_nmdar, s_nmdar, s_gabar, ca, na, h_unav = args
        ca_args: List[float] = [v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, x_nmdar, s_nmdar, s_gabar, ca, na, h_unav]
        x_: float=self.x
        y_: float=self.y

        dvdt: float = self.dvdt(args=args)
        dhNadt: float = self.nav.dhdt(v=v, h=h_nav)
        dnKdt: float = self.kvhh.dndt(v=v, n=n_kvhh)
        dhAdt: float = self.kva.dhdt(v=v, h=h_kva)
        dmKSdt: float = self.kvsi.dmdt(v=v, m=m_kvsi)
        dsAMPAdt: float = self.ampar.dsdt(v=v, s=s_ampar)
        dxNMDAdt: float = self.nmdar.dxdt(v=v, x=x_nmdar)
        dsNMDAdt: float = self.nmdar.dsdt(v=v, s=s_nmdar, x=x_nmdar)
        dsGABAdt: float = self.gabar.dsdt(v=v, s=s_gabar)
        dCadt: float = self.dCadt(args=ca_args)
        dh_unavdt: float= self.unav.dhdt(v=v,h=h_unav,y=y_)
        dNadt: float = self.dNadt(args=ca_args)
        
        return [dvdt, 
                dhNadt, 
                dnKdt, 
                dhAdt, 
                dmKSdt,
                dsAMPAdt, 
                dxNMDAdt, 
                dsNMDAdt, 
                dsGABAdt, 
                dCadt,
                dNadt,
                dh_unavdt,
                ]

      
class FNANdmodel(ANmodel):

    def __init__(self,ion: bool=False, concentration: Optional[Dict]=None) ->None:
        super().__init__(ion=ion, concentration=concentration)
        self.ini=self.params.fnand_ini
        
    def gen_params(self) ->Dict:

        param_dict: Dict = {}

        gX_name: List[str] = ['g_lek', 'g_lena',  'g_nav', 'g_kvhh', 'g_kva', 'g_kvsi', 
                         'g_cav', 'g_kca', 'g_nap', 'g_kir',   'g_unav', 'g_kna']
        gX_log: np.ndarray = 4 * np.random.rand(12) - 2  # from -2 to 2
        gX: np.ndarray = (10 * np.ones(12)) ** gX_log  # 0.01 ~ 100
        gX_itr: Iterator = zip(gX_name, gX)

        gR_name: List[str] = ['g_ampar', 'g_nmdar', 'g_gabar']
        gR_log: np.ndarray = 4 * np.random.rand(3) - 3  # from -3 to 1
        gR: np.ndarray = (10 * np.ones(3)) ** gR_log  # 0.001 ~ 10
        gR_itr: Iterator = zip(gR_name, gR)

        tCa_log: float = 2 * np.random.rand(1) + 1  # from 1 to 3
        tCa: float = 10 ** tCa_log    # 10 ~ 1000
        tCa_dict: Dict = {'t_ca': tCa}

        tNa_log: float = 1 * np.random.rand(1) + 3  # from 3 to 4
        tNa: float = 10 ** tNa_log    # 1000 ~ 10000
        tNa_dict: Dict = {'t_na': tNa}
        
        x_log: float=90*np.random.rand(1)+0#from 1 to 2
        x: float=(x_log)-45      #from -45 to +45
        x_dict: Dict={'x_na': x}

        y_log: float=90.0*np.random.rand(1)+0 #from 1 to 2
        y: float=(y_log)-45      #from -45 to +45
        y_dict: Dict={'y_na': y}
            
        param_dict.update(gX_itr)
        param_dict.update(gR_itr)
        param_dict.update(tCa_dict)
        param_dict.update(tNa_dict)
        param_dict.update(x_dict)
        param_dict.update(y_dict)
        return param_dict

    def set_params(self, params: Dict) -> None:

        self.lek.set_g(params['g_lek'])
        self.lena.set_g(params['g_lena'])
        self.nav.set_g(params['g_nav'])
        self.kvhh.set_g(params['g_kvhh'])
        self.kva.set_g(params['g_kva'])
        self.kvsi.set_g(params['g_kvsi'])
        self.cav.set_g(params['g_cav'])
        self.kca.set_g(params['g_kca'])
        self.nap.set_g(params['g_nap'])
        self.kir.set_g(params['g_kir'])
        self.unav.set_g(params['g_unav'])
        self.kna.set_g(params['g_kna'])
        self.ampar.set_g(params['g_ampar'])
        self.nmdar.set_g(params['g_nmdar'])
        self.gabar.set_g(params['g_gabar'])
        self.tau_ca = params['t_ca']
        self.tau_na = params['t_na']
        self.x=params['x_na']
        self.y=params['y_na']

    def set_sws_params(self) -> None:

        typ_params: Dict = params.TypicalParam().fnan_sws
        self.set_params(typ_params)

    def get_params(self) -> Dict:

        params: Dict = {}
        params['g_lek']: float = self.lek.get_g()
        params['g_lena']: float = self.lena.get_g()
        params['g_nav']: float = self.nav.get_g()
        params['g_kvhh']: float = self.kvhh.get_g()
        params['g_kva']: float = self.kva.get_g()
        params['g_kvsi']: float = self.kvsi.get_g()
        params['g_cav']: float = self.cav.get_g()
        params['g_kca']: float = self.kca.get_g()
        params['g_nap']: float = self.nap.get_g()
        params['g_kir']: float = self.kir.get_g()
        params['g_kna']: float=self.kna.get_g()
        params['g_unav']: flost=self_unav.get_g()
        params['g_ampar']: float = self.ampar.get_g()
        params['g.nmdar']: float = self.nmdar.get_g()
        params['g_gabar']: float = self.gabar.get_g()
        params['t_Ca']: float = self.tau_ca
        params['t_Na']: float = self.tau_na
        params['x_na']: float=self.x
        params['y_na']: float=self.y
        return params
    

    def dvdt(self, args: List[float]) -> float:

        v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, x_nmdar, s_nmdar, s_gabar, ca, na, h_unav = args
        x_: float=self.x
        return ((-10.0*self.params.area 
                * (self.lek.i(v)
                +self.lena.i(v)
                + self.nav.i(v, h=h_nav) 
                + self.kvhh.i(v, n=n_kvhh)
                + self.kva.i(v, h=h_kva)
                + self.kvsi.i(v, m=m_kvsi)
                + self.cav.i(v)
                + self.kca.i(v, ca=ca)
                + self.nap.i(v)
                + self.kir.i(v)
                +self.kna.i(v,na=na)
                +self.unav.i(v, h=h_unav, x=x_))
                - (self.ampar.i(v, s=s_ampar)
                + self.nmdar.i(v, s=s_nmdar)
                + self.gabar.i(v, s=s_gabar))) 
                / (10.0*self.params.cm*self.params.area))
    
    def dNadt(self, args: List[float]) -> float:
        v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, x_nmdar, s_nmdar, s_gabar, ca, na, h_unav = args
        x_: float=self.x
        b_na: float = self.params.b_na
        area: float = self.params.area
        tau_na: float= self.tau_na

        dNadt: float = (- b_na * (10.0*area*self.unav.i(v,h=h_unav, x=x_)+10.0*area*self.nav.i(v,h=h_nav)+10.0*area*self.nap.i(v)+10.0*area*self.lena.ina(v))- na / tau_na)
        return dNadt
    
    def dCadt(self, args: List[float]) -> float:

        v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, x_nmdar, s_nmdar, s_gabar, ca, na, h_unav = args
        a_Ca: float = self.params.a_ca
        x_: float=self.x
        area: float = self.params.area
        tau_Ca: float = self.tau_ca
        dCadt: float = -a_Ca * (10.0*area*self.cav.i(v)+10.0*area*self.lena.ica(v)) - ca/tau_Ca
        return dCadt

    def diff_op(self, args: List[float], time: float) -> List[float]:

        v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, x_nmdar, s_nmdar, s_gabar, ca, na, h_unav = args
        ca_args: List[float] = [v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, x_nmdar, s_nmdar, s_gabar, ca, na, h_unav]
        x_: float=self.x
        y_: float=self.y

        dvdt: float = self.dvdt(args=args)
        dhNadt: float = self.nav.dhdt(v=v, h=h_nav)
        dnKdt: float = self.kvhh.dndt(v=v, n=n_kvhh)
        dhAdt: float = self.kva.dhdt(v=v, h=h_kva)
        dmKSdt: float = self.kvsi.dmdt(v=v, m=m_kvsi)
        dsAMPAdt: float = self.ampar.dsdt(v=v, s=s_ampar)
        dxNMDAdt: float = self.nmdar.dxdt(v=v, x=x_nmdar)
        dsNMDAdt: float = self.nmdar.dsdt(v=v, s=s_nmdar, x=x_nmdar)
        dsGABAdt: float = self.gabar.dsdt(v=v, s=s_gabar)
        dCadt: float = self.dCadt(args=ca_args)
        dh_unavdt: float= self.unav.dhdt(v=v,h=h_unav,y=y_)
        dNadt: float = self.dNadt(args=ca_args)
        
        return [dvdt, 
                dhNadt, 
                dnKdt, 
                dhAdt, 
                dmKSdt,
                dsAMPAdt, 
                dxNMDAdt, 
                dsNMDAdt, 
                dsGABAdt, 
                dCadt,
                dNadt,
                dh_unavdt,
                ]

  
    
    

class NAN10dmodel(ANmodel):

    def __init__(self,ion: bool=False, concentration: Optional[Dict]=None) ->None:
        super().__init__(ion=ion, concentration=concentration)
        self.ini=self.params.nan10d_ini
        
    def gen_params(self) ->Dict:

        param_dict: Dict = {}
        gX_name: List[str] = ['g_kvhh',  'g_unav',  'g_kna',  'g_lek','g_lena',  'g_cav']
        gX_log: np.ndarray=4*np.random.rand(6)-2 #from -2 to 2
        gX: np.ndarray=(10*np.ones(6)) ** gX_log  # 0.021 ~ 100
        gX_itr: Iterator = zip(gX_name, gX)

       
            
        tNa_log: float = 1.0 * np.random.rand(1) + 3  # from 3 to 4
        tNa: float = 10 ** tNa_log    # 1000 ~ 10000
        tNa_dict: Dict = {'t_na': tNa}
        

            
        x_log: float=90*np.random.rand(1)+0#from 1 to 2
        x: float=(x_log)-45      #from -45 to +45
        x_dict: Dict={'x_na': x}

        y_log: float=90.0*np.random.rand(1)+0 #from 1 to 2
        y: float=(y_log)-45      #from -45 to +45
        y_dict: Dict={'y_na': y}
          
        param_dict.update(gX_itr)        
        param_dict.update(tNa_dict)
        param_dict.update(x_dict)
        param_dict.update(y_dict)
        
      
        return param_dict
    
    def set_params(self, params: Dict) -> None:
     
        self.kvhh.set_g(params["g_kvhh"])
        self.unav.set_g(params["g_unav"])
        self.kna.set_g(params["g_kna"])
        self.lek.set_g(params["g_lek"])
        self.lena.set_g(params["g_lena"])
        self.cav.set_g(params["g_cav"])
        self.tau_na=params["t_na"]
        self.x=params["x_na"]
        self.y=params["y_na"]
        
     
        

    def set_sws_params(self) -> None:

        typ_params: Dict = params.TypicalParam().nan10_sws
        self.set_params(typ_params)
     
    def get_params(self) -> Dict:

        params: Dict = {}        
        params["g_kvhh"]=self.kvhh.get_g()
        params["g_unav"]=self.unav.get_g()
        params["g_kna"]=self.kna.get_g()
        params["g_lek"]=self.lek.get_g()
        params["g_lena"]=self.lena.get_g()
        params["g_cav"]=self.cav.get_g()
        params["t_na"]=self.tau_na
        params["x_na"]=self.x
        params["y_na"]=self.y
        
        return params
    
    
    def dvdt(self, args: List[float]) -> float:

        v, h_unav,n_kvhh,na = args
        x_: float=self.x
        return -10.0*self.params.area * (self.unav.i(v,h=h_unav, x=x_) + self.kna.i(v, na=na)  + self.lek.i(v)+ self.lena.i(v)  +  self.kvhh.i(v, n=n_kvhh)+self.cav.i(v))/(10.0*self.params.cm*self.params.area)
    

    def dNadt(self, args: List[float]) -> float:
        v, h_unav, n_kvhh, na= args
        b_na: float = self.params.b_na
        area: float = self.params.area
        x_: float=self.x
        tau_na: float= self.tau_na

        dNadt: float = (- b_na * (10.0*area*self.unav.i(v,h=h_unav, x=x_)+10.0*area*self.lena.ina(v))
                        
                        - na / tau_na)
        return dNadt
                 
    def diff_op(self, args: List[float], time: float) -> List[float]:

        v, h_unav, n_kvhh,  na= args
        x_: float=self.x
        y_: float=self.y
         
        na_args: List[float] = [v, h_unav, n_kvhh,  na]
            
        dvdt: float = self.dvdt(args=args)
        dh_unavdt: float= self.unav.dhdt(v=v,h=h_unav,y=y_)
        dnKdt: float = self.kvhh.dndt(v=v, n=n_kvhh)
        dNadt: float = self.dNadt(args=na_args)
         
        return [dvdt, dh_unavdt, dnKdt, dNadt]


class NAN20dmodel(ANmodel):

    def __init__(self,ion: bool=False, concentration: Optional[Dict]=None) ->None:
        super().__init__(ion=ion, concentration=concentration)
        self.ini=self.params.nan20d_ini
        
    def gen_params(self) ->Dict:

        param_dict: Dict = {}
        gX_name: List[str] = ['g_kvhh',  'g_unav',  'g_nak', 'g_lek','g_lena','g_cav']
        gX_log: np.ndarray=4*np.random.rand(6)-2 #from -2 to 2
        gX: np.ndarray=(10*np.ones(6)) ** gX_log  # 0.021 ~ 100
        gX_itr: Iterator = zip(gX_name, gX)
            
        x_log: float=90*np.random.rand(1)+0#from 1 to 2
        x: float=(x_log)-45      #from -45 to +45
        x_dict: Dict={'x_na': x}

        y_log: float=90.0*np.random.rand(1)+0 #from 1 to 2
        y: float=(y_log)-45      #from -45 to +45
        y_dict: Dict={'y_na': y}
            

        param_dict.update(gX_itr)
        param_dict.update(x_dict)
        param_dict.update(y_dict)
        
      
        return param_dict
    
    def set_params(self, params: Dict) -> None:

        self.kvhh.set_g(params["g_kvhh"])
        self.unav.set_g(params["g_unav"])
        self.nak.set_g(params["g_nak"])
        self.lek.set_g(params["g_lek"])
        self.lena.set_g(params["g_lena"])
        self.cav.set_g(params["g_cav"])
        self.x=params["x_na"]
        self.y=params["y_na"]
        
     
        

    def set_sws_params(self) -> None:

        typ_params: Dict = params.TypicalParam().nan20_sws
        self.set_params(typ_params)
     
    def get_params(self) -> Dict:

        params: Dict = {}       
        params["g_kvhh"]=self.kvhh.get_g()
        params["g_unav"]=self.unav.get_g()
        params["g_nak"]=self.nak.get_g()
        params["g_lek"]=self.lek.get_g()
        params["g_lena"]=self.lena.get_g()
        params["g_cav"]=self.cav.get_g()
        params["x_na"]=self.x
        params["y_na"]=self.y
        
        return params
    
    
    def dvdt(self, args: List[float]) -> float:

        v, h_unav,n_kvhh,na = args
        x_: float=self.x
        return -10.0*self.params.area * (self.unav.i(v,h=h_unav, x=x_) + self.nak.i(v, na=na)  + self.lek.i(v)  +self.lena.i(v)+  self.kvhh.i(v, n=n_kvhh)+self.cav.i(v))/(10.0*self.params.cm*self.params.area)
    
    def dNadt(self, args: List[float]) -> float:
        v, h_unav, n_kvhh, na= args
        b_na: float = self.params.b_na
        area: float = self.params.area
        x_: float=self.x

        dNadt: float = (- b_na * (10.0*area*self.unav.i(v,h=h_unav, x=x_)+10.0*area*self.lena.ina(v)+10.0*3*area*self.nak.i(v,na=na)))
        return dNadt
                 
    def diff_op(self, args: List[float], time: float) -> List[float]:

        v, h_unav, n_kvhh,  na= args
        x_: float=self.x
        y_: float=self.y
                    
        na_args: List[float] = [v, h_unav, n_kvhh,  na]
            
        dvdt: float = self.dvdt(args=args)
        dh_unavdt: float= self.unav.dhdt(v=v,h=h_unav,y=y_)
        dnKdt: float = self.kvhh.dndt(v=v, n=n_kvhh)
        dNadt: float = self.dNadt(args=na_args)
        
        return [dvdt, dh_unavdt, dnKdt, dNadt]