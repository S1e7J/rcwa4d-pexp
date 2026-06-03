import pandas as pd
import numpy as np
import pickle
import os
from rcwa4d import *
from tqdm import tqdm
from typing import Callable
from cut_pattern import *

number = int | float | complex
DEG = np.pi/180

def plate_pattern(n_total: number = 4, thickness = 0.2):
    Ny = Nx = 1000 
    eps = np.ones([Ny,Nx]) * n_total 
    
    radius = 0.25
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

def custom_pattern(pickle_filepath: str, thickness = 0.2):
    # Cargar la matriz eps desde el archivo pickle
    with open(pickle_filepath, 'rb') as f:
        eps = pickle.load(f)
    
    ind = 1
    NM = (2*ind+1)**2
    NMNM = NM**2
    freqs = np.linspace(0.7, 0.84, 141)
    twists = np.linspace(0, 45, 46) * DEG
    kxs = np.linspace(0, 0.5, 51) 
    
    return eps, thickness, freqs, twists, kxs, ind, NM, NMNM

def rod_pattern(n_total: number = 4, thickness = 0.2):
    Ny = Nx = 1000 
    eps = np.ones([Ny,Nx]) * n_total
    
    radius = 0.25
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

def _aux_general_pattern(figuras_internas : List[Figure], lado_celda : number = 10, n_total: number = 4, thickness = 0.2):
    eps = generar_patron(figuras_internas, lado_celda, n_total)

    radius = 0.25
    ind = 1
    NM = (2*ind+1)**2
    NMNM = NM**2
    freqs = np.linspace(0.7,0.84,141)
    twists = np.linspace(0,45,46) * DEG
    kxs = np.linspace(0,0.5,51) 
    return eps, thickness, freqs, twists, kxs, ind, NM, NMNM

def general_pattern(figuras_internas : List[Figure], lado_celda : number = 10):
    def aux(n_total: number = 4, thickness = 0.2):
        return _aux_general_pattern(figuras_internas, lado_celda, n_total)
    return aux

def calculate_freq_respect_kxs(n_total: number = 4, pattern: Callable = plate_pattern, thickness = 0.2):
    eps, thicness, freqs, twists, kxs, ind, NM, NMNM = pattern(n_total, thickness)
    trans = []
    twist = 1*DEG
    for kx in tqdm(kxs, desc=f"Calculando para n={n_total:.2f}"):
        for freq in freqs:
            obj2 = rcwa([eps,eps], [thickness,thickness], [1,2], twist=twist, N=ind, M=ind, verbose=0)
            obj2.set_freq_k(freq, (kx, 0))
            (r,t), (reflected,transmitted) = obj2.get_RT(0,1)
            trans.append(t)
    return trans

def calculate_freq_respect_angle(n_total: number = 4, pattern: Callable = plate_pattern, thickness = 0.2):
    eps, thickness, freqs, twists, kxs, ind, NM, NMNM = pattern(n_total, thickness)
    trans = []
    for twist in tqdm(twists):
        for freq in freqs:
            obj2 = rcwa([eps,eps], [thickness,thickness], [1,2],twist=twist, N=ind, M=ind, verbose=0)
            obj2.set_freq_k(freq, (0, 0))
            (r,t), (reflected,transmitted) = obj2.get_RT(0,1)
            trans.append(t)
    return trans

def generate_eps_pickle(filename: str, radius: float, n_total: float = 4, Nx: int = 1000, Ny: int = 1000):
    """
    Genera la matriz eps con un radio específico y la guarda como un archivo pickle.
    """
    # Crear la matriz
    eps = np.ones([Ny, Nx]) * n_total 
    
    # Crear la malla
    xs, ys = np.linspace(-0.5, 0.5, Nx), np.linspace(-0.5, 0.5, Ny)
    xs, ys = np.meshgrid(xs, ys)
    
    # Aplicar la condición del círculo
    eps[xs**2 + ys**2 < radius**2] = 1 
    
    # Guardar en formato pickle
    with open(filename, 'wb') as f:
        pickle.dump(eps, f)
        
    print(f"Matriz eps (radio={radius}) guardada exitosamente en '{filename}'.")
