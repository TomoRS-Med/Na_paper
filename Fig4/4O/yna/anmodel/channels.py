# -*- coding: utf-8 -*-

"""
This is the channel module for Averaged Neuron (AN) model. 
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

import numpy as np
from typing import Optional

import params


params = params.Constants()


class Base:
    """ Keep basic attributes and helper functions for each channel.

    Parameters
    ----------
    g : float
        channel condactance
    e : float
        equilibrium (reversal) potential for a channel
    
    Attributes
    ----------
    g : float
        channel conductance
    e : float
        equilibrium (reversal) potential for a channel
    """

    def __init__(self, g: float, e: float) -> None:
        self.g = g
        self.e = e
    
    def set_g(self, new_g: float) -> None:
        """ Set a new conductance for a channel.

        Parameters
        ----------
        new_g : float
            new conductance set for a channel
        """
        self.g = new_g

    def get_g(self) -> float:
        ''' Get current channel conductance value.

        Returns
        ----------
        float
            current conductance
        '''
        return self.g

    def set_e(self, new_e: float) -> None:
        """ Set a new equiribrium potential for a channel

        Parameters
        ----------
            new equiribrium potential for a channel
        """
        self.e = new_e

    def get_e(self) -> float:
        ''' Get current equilibrium potential.

        Returns:
        ----------
        float
            current equilibrium potential.
        '''
        return self.e

class Leak(Base):

    def __init__(self, g: Optional[int]=None, e: float=params.vL) -> None:
        super().__init__(g, e)

    def i(self, v:float) -> float:
        return self.g * (v - self.e)

    def set_div(self, vnal: float=params.vNaL, vkl: float=params.vK) -> None:
        self.vnal = vnal
        self.vkl = vkl
        self.gnal = self.g * (self.e - self.vkl) / (self.vnal - self.vkl)
        self.gkl = self.g * (self.e - self.vnal) / (self.vkl - self.vnal)

    def set_gna(self, new_gnal: float) -> None:
        self.gnal = new_gnal
    
    def set_gk(self, new_gkl: float) -> None:
        self.gkl = new_gkl

    def ikl(self, v: float) -> float:
        return self.gkl * (v - self.vkl)

    def inal(self, v: float) -> float:
        return self.gnal * (v - self.vnal)

    def ina(self, v: float) -> float:
        return self.g * (self.e +100.0) / (0.0 +100.0)*0.44*(v-55.0)
    
    def ica(self, v: float) -> float:
        return self.g * (self.e +100.0) / (0.0 +100.0)*0.25* (v -120.0)
    
    def i_div(self, v: float) -> float:
        return self.inal(v) + self.ikl(v)
    

class LeakK(Base):

    def __init__(self, g: Optional[int]=None, e: float=params.vK) -> None:
        super().__init__(g, e)

    def i(self, v:float) -> float:
        return self.g * (v - self.e)

class LeakNa(Base):

    def __init__(self, g: Optional[int]=None, e: float=params.vNaL) -> None:
        super().__init__(g, e)

    def i(self, v:float) -> float:
        return self.g * (v - self.e)

    def ina(self, v: float) -> float:
        return self.g *0.44*(v-55.0)
    
    def ica(self, v: float) -> float:
        return self.g *0.25* (v -120.0)

class NavHH(Base):

    def __init__(self, g: Optional[float]=None, e: float=params.vNa) -> None:
        super().__init__(g, e)

    def am(self, v: float) -> float:
        if v == -33.:
            return 1.
        else:
            return 0.1 * (v+33.0) / (1.0-np.exp(-(v+33.0)/10.0))

    def bm(self, v: float) -> float:
        return 4.0 * np.exp(-(v+53.7)/12.0)

    def m_inf(self, v: float) -> float:
        return self.am(v) / (self.am(v) + self.bm(v))

    def ah(self, v: float) -> float:
        return 0.07 * np.exp(-(v+50.0)/10.0)

    def bh(self, v: float) -> float:
        return 1.0 / (1.0 + np.exp(-(v+20.0)/10.0))

    def h_inf(self, v: float) -> float:
        return self.ah(v) / (self.ah(v) + self.bh(v))

    def h_tau(self, v: float) -> float:
        return 1 / 4 * (self.ah(v)+self.bh(v))

    def dhdt(self, v: float, h: float) -> float:
        return 4.0 * (self.ah(v)*(1-h) - self.bh(v)*h)

    def i(self, v: float, h: float) -> float:
        return self.g * (self.m_inf(v)**3) * h * (v-self.e)


class KvHH(Base):

    def __init__(self, g: Optional[float]=None, e: float=params.vK) -> None:
        super().__init__(g, e)

    def an(self, v: float) -> float:
        if v == -34.:
            return 0.1
        else:
            return 0.01 * (v+34.0) / (1.0-np.exp(-(v+34.0)/10.0))

    def bn(self, v: float) -> float:
        return 0.125 * np.exp(-(v+44.0)/25.0)

    def n_inf(self, v: float) -> float:
        return self.an(v) / (self.an(v)+self.bn(v))

    def n_tau(self, v: float) -> float:
        return 1 / (4 * (self.an(v) + self.bn(v)))

    def dndt(self, v: float, n: float) -> float:
        return 4.0 * (self.an(v)*(1-n)-self.bn(v)*n)

    def i(self, v: float, n: float) -> float:
        return self.g * n**4 * (v-self.e)


class KvA(Base):

    def __init__(self, g: Optional[float]=None, e: float=params.vK, 
                 tau: float=params.tau_a) -> None:
        super().__init__(g, e)
        self.tau = tau

    def m_inf(self, v: float) -> float:
        return 1.0 / (1.0 + np.exp(-(v+50.0)/20.0))

    def h_inf(self, v: float) -> float:
        return 1.0 / (1.0 + np.exp((v+80.0)/6.0))

    def dhdt(self, v: float, h: float) -> float:
        return (self.h_inf(v)-h) / self.tau

    def i(self, v: float, h: float) -> float:
        return self.g * (self.m_inf(v)**3) * h * (v-self.e)


class KvSI(Base):

    def __init__(self, g: Optional[float]=None, e: float=params.vK) -> None:
        super().__init__(g, e)

    def m_inf(self, v: float) -> float:
        return 1.0 / (1.0 + np.exp(-(v+34.0)/6.5))

    def m_tau(self, v: float) -> float:
        return 8.0 / (np.exp(-(v+55.0)/30.0) + np.exp((v+55.0)/30.0))

    def dmdt(self, v: float, m: float) -> float:
        return (self.m_inf(v)-m) / self.m_tau(v)

    def i(self, v: float, m: float) -> float:
        return self.g * m * (v-self.e)


class Cav(Base):

    def __init__(self, g: Optional[float]=None, e: float=params.vCa) -> None:
        super().__init__(g, e)

    def m_inf(self, v: float) -> float:
        return 1.0 / (1.0 + np.exp(-(v+20.0)/9.0))

    def i(self, v: float) -> float:
        return self.g * self.m_inf(v)**2 * (v-self.e)

class CavI(Base):
    def __init__(self, g: Optional[float]=None, e: float=params.vCa) -> None:
        super().__init__(g, e)

    def m_inf(self, v: float) -> float:
        return 1.0 / (1.0 + np.exp(-(v+20.0)/9.0))

    def i(self, v: float) -> float:
        return self.g * self.m_inf(v)**2 * (v-self.e)

class NaP(Base):

    def __init__(self, g: Optional[float]=None, e: float=params.vNa) -> None:
        super().__init__(g, e)

    def m_inf(self, v: float) -> float:
        return 1.0 / (1.0 + np.exp(-(v+55.7)/7.7))

    def i(self, v: float) -> float:
        return self.g * self.m_inf(v)**3 * (v-self.e)


class KCa(Base):

    def __init__(self, g: Optional[float]=None, e: float=params.vK, 
                 kd_ca: float=params.kd_ca) -> None:
        super().__init__(g, e)
        self.kd_ca = kd_ca

    def m_inf(self, ca: float) -> float:

        return 1.0 / (1.0 + (self.kd_ca/ca)**(3.5))

    def i(self, v: float, ca: float) -> float:
        return self.g * self.m_inf(ca) * (v-self.e)


class KIR(Base):

    def __init__(self, g: Optional[float]=None, e: float=params.vK) -> None:
        super().__init__(g, e)

    def h_inf(self, v: float) -> float:
        return 1.0/(1.0 + np.exp((v + 75.0)/4.0))

    def i(self, v: float) -> float:
        return self.g * self.h_inf(v) * (v-self.e)


class AMPAR(Base):

    def __init__(self, g: Optional[float]=None, e: float=params.vAMPAR, 
                 s_a: float=params.s_a_ampar, s_tau: float=params.s_tau_ampar) -> None:
        super().__init__(g, e)
        self.s_a = s_a
        self.tau_a = s_tau

    def f(self, v: float) -> float:
        return 1.0 / (1.0 + np.exp(-(v-20.0)/2.0))

    def dsdt(self, v: float, s: float) -> float:
        return self.s_a * self.f(v) - s/self.tau_a

    def i(self, v: float, s: float) -> float:
        return self.g * s * (v - self.e)

    def ina(self, v: float, s: float) -> float:
        return self.g*0.65* s * (v - 55.0)
     
class NMDAR(Base):

    def __init__(self, g: Optional[float]=None, e: float=params.vNMDAR, 
                 s_a: float=params.s_a_nmdar, s_tau: float=params.s_tau_nmdar, 
                 x_a: float=params.x_a_nmdar, x_tau: float=params.x_tau_nmdar,
                 ion: bool=False, ex_mg: Optional[float]=None) -> None:
        super().__init__(g, e)
        self.s_a = s_a
        self.s_tau = s_tau
        self.x_a = x_a
        self.x_tau = x_tau
        self.ion = ion
        self.ex_mg = ex_mg

    def f(self, v: float) -> float:
        return 1.0 / (1.0 + np.exp(-(v-20.0)/2.0))

    def dxdt(self, v: float, x: float) -> float:
        return self.x_a * self.f(v) - x/self.x_tau

    def dsdt(self, v: float, s: float, x: float) -> float:
        return self.s_a * x * (1-s) - s/self.s_tau

    def i(self, v: float, s: float) -> float:
        if self.ion:
            return 1.1 / (1.0+self.ex_mg/8.0) * self.g * s * (v-self.e)
        else:
            return self.g * s * (v-self.e)
        
        
    def ina(self, v: float, s: float) -> float:
        return self.g*0.14* s * (v - 55.0)
    
    
    def ica(self, v: float, s: float) -> float:
        return self.g*0.24* s * (v -120.0)


class GABAR(Base):
    def __init__(self, g: Optional[float]=None, e: float=params.vGABAR, 
                 s_a: float=params.s_a_gabar, s_tau: float=params.s_tau_gabar) -> None:
        super().__init__(g, e)
        self.s_a = s_a
        self.s_tau = s_tau

    def f(self, v: float) -> float:
        return 1.0 / (1.0 + np.exp(-(v-20.0)/2.0))

    def dsdt(self, v: float, s: float) -> float:
        return self.s_a * self.f(v) - s/self.s_tau

    def i(self, v: float, s: float) -> float:
        return self.g * s * (v-self.e)
    

    
class KNA(Base):

    def __init__(self, g: Optional[float]=None, e: float=params.vK, 
                 ke_na: float=params.ke_na) -> None:
        super().__init__(g, e)
        self.ke_na = ke_na
      
    def m_inf(self, na: float) -> float:
        return 1.0/(1.0+(self.ke_na/na)**(3.0))
    
    def i(self, v: float, na: float) ->float:
        return self.g * self.m_inf(na) * (v-self.e)

class UNaV(Base):

    def __init__(self, g: Optional[float]=None, e: float=params.vNa) -> None:
        super().__init__(g, e)

    def am(self, v: float, x: float) -> float:
        if v == -33.0-x:
            return 1.
        else:
            return 0.1 * (v+33.0+x) / (1.0-np.exp(-(v+33.0+x)/10.0))

    def bm(self, v: float, x: float) -> float:
        return 4.0 * np.exp(-(v+53.7+x)/12.0)

    def m_inf(self, v: float, x: float) -> float:
        return self.am(v,x) / (self.am(v, x) + self.bm(v, x))

    def ah(self, v: float, y: float) -> float:
        return 0.07 * np.exp(-(v+50.0+y)/10.0)

    def bh(self, v: float, y: float) -> float:
        return 1.0 / (1.0 + np.exp(-(v+20.0+y)/10.0))

    def h_inf(self, v: float, y: float) -> float:
        return self.ah(v,y) / (self.ah(v,y) + self.bh(v,y))

    def h_tau(self, v: float, y: float) -> float:
        return 1 / (4 * (self.ah(v,y)+self.bh(v,y)))

    def dhdt(self, v: float, h: float,y: float) -> float:
        return (self.h_inf(v, y)-h) / self.h_tau(v,y)

    def i(self, v: float, h: float, x: float) -> float:
        return self.g * (self.m_inf(v, x)**3) * h * (v-self.e)
        
    
    
class NaK(Base):
    def __init__(self, g: Optional[float]=None, e: float=params.vNa, 
                 ke_na: float=params.ke_na) -> None:
        super().__init__(g, e)
        self.ke_na = ke_na
      
    
    def apump(self, v: float, na: float) ->float:
        x=1+(3.5/4.0)
        y=1+(10.0/na)
        z1=x**-2
        z2=y**-3
        z=z1*z2
        return z
    
    def i(self, v: float, na: float) ->float:
        return self.g * self.apump(v,na)
    


