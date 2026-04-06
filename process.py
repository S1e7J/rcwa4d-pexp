import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# Dimensiones de tu simulación original
NUM_KXS = 51
NUM_FREQS = 141

def analizar_transmision(ruta_archivo, transmision_a_analizar="transmision_plate"):
    with open(ruta_archivo, 'rb') as f:
        datos = pickle.load(f)
        
    trans_1d = np.array(datos[transmision_a_analizar])
    
    # 1. Reconstruir la matriz 2D (Eje Y: kx, Eje X: freqs)
    # Según tu bucle original: for kx in kxs: for freq in freqs:
    trans_2d = trans_1d.reshape((NUM_KXS, NUM_FREQS))
    
    # 2. Búsqueda de "Bandgaps" (Espacios de no transmisión)
    # Un bandgap absoluto ocurre si, para una frecuencia dada, 
    # la transmisión es casi cero en TODOS los ángulos/vectores kx.
    umbral_transmision = 0.05 # 5% de transmisión
    transmision_maxima_por_frecuencia = np.max(trans_2d, axis=0)
    indices_bandgap = np.where(transmision_maxima_por_frecuencia < umbral_transmision)[0]
    
    tiene_bandgap = len(indices_bandgap) > 0
    
    # 3. Búsqueda de "Anomalías" (Resonancias abruptas tipo Fano)
    # Buscamos cambios extremadamente bruscos en la transmisión usando el gradiente.
    gradiente_k, gradiente_f = np.gradient(trans_2d)
    magnitud_gradiente = np.sqrt(gradiente_k**2 + gradiente_f**2)
    
    # Si el gradiente es muy alto, hay una resonancia aguda
    umbral_anomalia = np.percentile(magnitud_gradiente, 99) # Tomamos el 1% de cambios más bruscos
    anomalias_kx, anomalias_f = np.where(magnitud_gradiente > umbral_anomalia)
    
    return trans_2d, tiene_bandgap, anomalias_kx, anomalias_f, datos['formula']

def visualizar_analisis(trans_2d, anomalias_kx, anomalias_f, formula):
    plt.figure(figsize=(10, 6))
    
    # Dibujar el mapa de calor de las bandas
    plt.imshow(trans_2d, aspect='auto', origin='lower', cmap='viridis',
               extent=[0.7, 0.84, 0, 0.5]) # Límites de freqs y kxs de tu script original
    
    plt.colorbar(label='Transmisión')
    plt.xlabel('Frecuencia normalizada (a/λ)')
    plt.ylabel('Vector de onda kx (2π/a)')
    plt.title(f'Estructura de Bandas de Transmisión: {formula}')
    
    # Superponer las anomalías detectadas (en rojo)
    # Convertimos los índices de vuelta a la escala física para el gráfico
    freqs_fisicas = 0.7 + (anomalias_f / (NUM_FREQS - 1)) * (0.84 - 0.7)
    kxs_fisicos = (anomalias_kx / (NUM_KXS - 1)) * 0.5
    
    plt.scatter(freqs_fisicas, kxs_fisicos, color='red', s=5, alpha=0.5, label='Anomalía/Resonancia')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==========================================
# EJECUCIÓN DEL ANÁLISIS
# ==========================================
archivos_pkl = glob("resultados_rcwa/*.pkl")
materiales_interesantes = []

print("Buscando características especiales en las simulaciones...")

for archivo in archivos_pkl:
    trans_2d, tiene_bandgap, anomalias_kx, _, formula = analizar_transmision(archivo)
    
    if tiene_bandgap:
        print(f"¡Bandgap completo detectado en {formula}!")
        materiales_interesantes.append(archivo)
    
    # Puedes ajustar tu criterio aquí. Por ejemplo, si hay muchas anomalías juntas.
    if len(anomalias_kx) > 50: 
        print(f"Múltiples resonancias/anomalías detectadas en {formula}.")
        if archivo not in materiales_interesantes:
            materiales_interesantes.append(archivo)

# Graficar el primer material interesante que hayamos encontrado
if materiales_interesantes:
    print(f"\nGraficando el material más prometedor: {materiales_interesantes[0]}")
    trans_2d, _, ak, af, formula = analizar_transmision(materiales_interesantes[0])
    visualizar_analisis(trans_2d, ak, af, formula)
else:
    print("No se encontraron bandgaps fuertes o anomalías destacables con los umbrales actuales.")
