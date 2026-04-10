import pandas as pd
import numpy as np
import pickle
import os
from rcwa4d import *
from tqdm import tqdm
from typing import Callable
number = int | float | complex
DEG = np.pi/180

def plate_pattern(n_total: number = 4):
    Ny = Nx = 1000 
    eps = np.ones([Ny,Nx]) * n_total 
    
    radius = 0.25
    thickness = 0.2
    xs, ys = np.linspace(-0.5,0.5,Nx), np.linspace(-0.5,0.5,Ny)
    xs, ys = np.meshgrid(xs,ys)
    eps[xs**2 + ys**2 < radius**2] = 1 
    
    ind = 1
    NM = (2*ind+1)**2
    NMNM = NM**2
    freqs = np.linspace(0.7,0.84,141)
    twists = np.linspace(0,45,46) * DEG
    kxs = np.linspace(0,0.5,51) 
    
    return eps, thickness, freqs, twists, kxs, ind, NM, NMNM

def rod_pattern(n_total: number = 4):
    Ny = Nx = 1000 
    eps = np.ones([Ny,Nx]) * n_total
    
    radius = 0.25
    thickness = 0.2
    xs, ys = np.linspace(-0.5,0.5,Nx), np.linspace(-0.5,0.5,Ny)
    xs, ys = np.meshgrid(xs,ys)
    eps[ys**2 < radius**2] = 1 
    
    ind = 1
    NM = (2*ind+1)**2
    NMNM = NM**2
    freqs = np.linspace(0.7,0.84,141)
    twists = np.linspace(0,45,46) * DEG
    kxs = np.linspace(0,0.5,51) 
    
    return eps, thickness, freqs, twists, kxs, ind, NM, NMNM

def calculate_freq_respect_kxs(n_total: number = 4, pattern: Callable = plate_pattern):
    eps, thickness, freqs, twists, kxs, ind, NM, NMNM = pattern(n_total)
    trans = []
    twist = 1*DEG
    for kx in tqdm(kxs, desc=f"Calculando para n={n_total:.2f}"):
        for freq in freqs:
            obj2 = rcwa([eps,eps], [thickness,thickness], [1,2], twist=twist, N=ind, M=ind, verbose=0)
            obj2.set_freq_k(freq, (kx, 0))
            (r,t), (reflected,transmitted) = obj2.get_RT(0,1)
            trans.append(t)
    return trans

def calculate_freq_respect_angle(n_total: number = 4, pattern: Callable = plate_pattern):
    eps, thickness, freqs, twists, kxs, ind, NM, NMNM = pattern(n_total)
    trans = []
    for twist in tqdm(twists):
        for freq in freqs:
            obj2 = rcwa([eps,eps], [thickness,thickness], [1,2],twist=twist, N=ind, M=ind, verbose=0)
            obj2.set_freq_k(freq, (0, 0))
            (r,t), (reflected,transmitted) = obj2.get_RT(0,1)
            trans.append(t)
    return trans

"""
TODO: Respecto al angulo contra Frecuencia (Solamente para los 5 materiales que sobreviven).
"""
