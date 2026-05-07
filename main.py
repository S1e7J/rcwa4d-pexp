import os
import pandas as pd
import pickle
import argparse
from util import *

# ==========================================
# CONFIGURACIÓN DE ARGUMENTOS CLI
# ==========================================
parser = argparse.ArgumentParser(description="Script para calcular transmisiones RCWA de materiales.")
parser.add_argument(
    '-r', '--rangulo', 
    action='store_true',
    help="Si se usa esta flag se calcula tambien frecuencia respecto al angulo"
)
parser.add_argument(
    '-t', '--thickness', 
    action='store_true',
    help="Si se usa esta flag se cambia el grosor por el que este en la entrada 'thickness' del .csv"
)
parser.add_argument(
    '-f', '--formula', 
    type=str, 
    nargs='+',
    default=None, 
    help="Filtra por fórmula química (ej: SiO2). Si se omite, calcula todos los materiales."
)
parser.add_argument(
    '-i', '--information', 
    type=str, 
    default="./dielectricos_absolutamente_todo.csv", 
    help="Define el archivo de donde sacar la información, tiene que tener como claves: material_id, formula_pretty, e_total"
)
args = parser.parse_args()

# ==========================================
# EJECUCIÓN Y GUARDADO EN PICKLE
# ==========================================

# 1. Crear una carpeta para guardar los resultados si no existe
carpeta_resultados = "resultados_rcwa"
os.makedirs(carpeta_resultados, exist_ok=True)

# 2. Cargar el CSV
df_materiales = pd.read_csv(args.information)

# 3. Filtrar por fórmula si el usuario pasó el argumento
if args.formula:
    print(f"Aplicando filtro: Buscando solo materiales con fórmula en '{args.formula}'...")
    df_materiales = df_materiales[df_materiales['formula_pretty'].isin(args.formula)]
    
    # Validar si el filtro dejó el DataFrame vacío
    if df_materiales.empty:
        print(f"No se encontró el compuesto '{args.formula}' en el CSV. Saliendo del script.")
        exit(0)
else:
    print("No se especificó fórmula. Se procesarán todos los materiales del CSV.")

print(f"Total de materiales a calcular: {len(df_materiales)}")

# 4. Iterar y guardar
for index, fila in df_materiales.iterrows():
    mat_id = fila['material_id']
    compuesto = fila['formula_pretty']  # Usamos la fórmula bonita (ej: SiO2)
    valor_dielectrico = fila['e_total']
    thickness = fila['thickness'] if args.thickness else 0.2
    
    # Ignorar si no hay valor dieléctrico
    if pd.isna(valor_dielectrico):
        continue
        
    # Nombre del archivo: ej. "resultados_rcwa/mp-149_Si.pkl"
    nombre_archivo = f"{carpeta_resultados}/{mat_id}_{compuesto}{"_rangle" if args.rangulo else ""}.pkl"
    
    # Comprobar si el archivo ya existe (ideal si el script se corta y quieres retomarlo)
    if os.path.exists(nombre_archivo):
        print(f"Saltando {mat_id} ({compuesto}) - Ya calculado.")
        continue
        
    print(f"\nCalculando: {mat_id} - {compuesto} (e_total={valor_dielectrico})")
    
    # Calcular
    transmision_plate = calculate_freq_respect_kxs(n_total=valor_dielectrico, pattern=plate_pattern, thickness = thickness)
    transmision_rod = calculate_freq_respect_kxs(n_total=valor_dielectrico, pattern=rod_pattern, thickness = thickness)
    transmision_rangle_plate = calculate_freq_respect_angle(n_total=valor_dielectrico, pattern=plate_pattern, thickness = thickness) if args.rangulo else None
    transmision_rangle_rod = calculate_freq_respect_angle(n_total=valor_dielectrico, pattern=rod_pattern, thickness = thickness) if args.rangulo else None
    
    # Guardar en pickle
    with open(nombre_archivo, 'wb') as f:
        # Guardamos un diccionario por si a futuro quieres agregar más datos
        datos_a_guardar = {
            'material_id': mat_id,
            'formula': compuesto,
            'e_total': valor_dielectrico,
            'transmision_plate': transmision_plate,
            'transmision_rod': transmision_rod,
            'transmision_rangle_plate' : transmision_rangle_plate,
            'transmision_rangle_rod' : transmision_rangle_rod,
        }
        pickle.dump(datos_a_guardar, f)

print("\n¡Todas las simulaciones han terminado!")
