#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import requests
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from time import time
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')
sys.path.append('../')
sys.path.append('../anmodel')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
import anmodel
from tqdm import tqdm

class TomoRS():
  
    def __init__(self,  ncore: int=50) -> None:
        self.ncore = ncore
    def singleprocess(self, args: List) -> None:
        mnnan1=anmodel.models.FNANmodel()
        S=pd.read_csv("A.csv")
        core, now, time_start = args
        date: str = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd()
        res_p: Path = p / 'results' 
        res_p.mkdir(parents=True, exist_ok=True)
        save_p: Path = res_p / f'{date}_{core}.pickle'


        y=-45.0+1.8*core
        als = anmodel.analysis.WaveCheck()
        C=0
        M=0
        Y=0
        K=0
        L=0
        Const=255
        for j in range(len(S["g_kvhh"])):
            if(j==0):
                continue
            A=[]
            A.append(S['g_leak'][j])
            A.append(S['g_nav'][j])
            A.append(S['g_kvhh'][j])
            A.append(S['g_kva'][j])
            A.append(S['g_kvsi'][j])
            A.append(S['g_cav'][j])
            A.append(S['g_kca'][j])
            A.append(S['g_nap'][j])
            A.append(S['g_kir'][j])
            A.append(S['g_unav'][j])
            A.append(S['g_kna'][j])
            A.append(S['g_ampar'][j])
            A.append(S['g_nmdar'][j])
            A.append(S['g_gabar'][j])
            A.append(S['t_ca'][j])
            A.append(S['t_na'][j])
            A.append(S['x_na'][j])
            A.append(S['y_na'][j]+y)

            
            BBB=['g_leak', 'g_nav', 'g_kvhh', 'g_kva', 'g_kvsi', 'g_cav', 'g_kca', 'g_nap', 'g_kir', 'g_unav', 'g_kna', 'g_ampar','g_nmdar', 'g_gabar', 't_ca', 't_na', 'x_na', 'y_na']
            dic={key:  val for key, val in zip(BBB,A)}

            new_params=dic
            mnnan1.set_params(new_params)
            s,info=mnnan1.run_odeint()
            st=als.pattern(s[4999:,0]).name
            if(st=="SWS" or st=="SWS_FEW_SPIKES"):
                C=C+1
            elif(st=="RESTING"):
                M=M+1
            elif(st=="AWAKE"):
                Y=Y+1
            else:
                L=L+1
                
            
        A=[C,M,Y,L]
        BBB=[ 'SWS', 'RESTING', 'AWAKE','ELSE']
        dic={key:  val for key, val in zip(BBB,A)}
        new_params: pd.DataFrame=pd.DataFrame.from_dict(dic,orient='index').T
        new_params.to_csv("{}.csv" .format(core))

                

    def multi_singleprocess(self) -> None:
        args: List = []
        now: datetime = datetime.now()
        time_start: float = time()
        for core in range(self.ncore):
            args.append((core, now, time_start))


        print(f'Random search: using {self.ncore} core(s) to explore')
        with Pool(processes=self.ncore) as pool:
            pool.map(self.singleprocess,args)


if __name__ == '__main__':
    XXZZ=input("Core: ")
    rps = TomoRS()
    rps.ncore=int(XXZZ)
    rps.multi_singleprocess()

