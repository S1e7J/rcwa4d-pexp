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
    # CORRECCIÓN: Usar n_total. Si n_total es el índice de refracción (n), 
    # recuerda que eps = n**2. Si ya es la constante dieléctrica, úsalo directo.
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
    eps = np.ones([Ny,Nx]) * n_total # CORRECCIÓN AQUÍ
    
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
    
    # CORRECCIÓN: Retornar los 8 valores para que coincida con plate_pattern
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

# ==========================================
# EJECUCIÓN Y GUARDADO EN PICKLE
# ==========================================

# 1. Crear una carpeta para guardar los resultados si no existe
carpeta_resultados = "resultados_rcwa"
os.makedirs(carpeta_resultados, exist_ok=True)

# 2. Cargar el Excel
df_materiales = pd.read_csv('./dielectricos_absolutamente_todo.csv')

# 3. Iterar y guardar
for index, fila in df_materiales.iterrows():
    mat_id = fila['material_id']
    compuesto = fila['formula_pretty']  # Usamos la fórmula bonita (ej: SiO2)
    valor_dielectrico = fila['e_total']
    
    # Ignorar si no hay valor dieléctrico
    if pd.isna(valor_dielectrico):
        continue
        
    # Nombre del archivo: ej. "resultados_rcwa/mp-149_Si.pkl"
    nombre_archivo = f"{carpeta_resultados}/{mat_id}_{compuesto}.pkl"
    
    # Comprobar si el archivo ya existe (ideal si el script se corta y quieres retomarlo)
    if os.path.exists(nombre_archivo):
        print(f"Saltando {mat_id} ({compuesto}) - Ya calculado.")
        continue
        
    print(f"\nCalculando: {mat_id} - {compuesto} (e_total={valor_dielectrico})")
    
    # Calcular
    transmision_plate = calculate_freq_respect_kxs(n_total=valor_dielectrico, pattern=plate_pattern)
    transmision_rod = calculate_freq_respect_kxs(n_total=valor_dielectrico, pattern=rod_pattern)
    
    # Guardar en pickle
    with open(nombre_archivo, 'wb') as f:
        # Guardamos un diccionario por si a futuro quieres agregar más datos (ej: las frecuencias o ks)
        datos_a_guardar = {
            'material_id': mat_id,
            'formula': compuesto,
            'e_total': valor_dielectrico,
            'transmision_plate': transmision_plate,
            'transmision_rod': transmision_rod
        }
        pickle.dump(datos_a_guardar, f)

print("¡Todas las simulaciones han terminado!")
