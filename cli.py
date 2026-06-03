import os
import pandas as pd
import pickle
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from util import *

# --- CONFIGURACIÓN GLOBAL PARA REPORTES ---
plt.switch_backend('Agg') 
NUM_FREQS = 141
FREQ_INICIO = 0.7
FREQ_FIN = 0.84
TWIST_INICIO = 0.0
TWIST_FIN = 90.0

# ==========================================
# FUNCIONES DE APOYO (REPORTE)
# ==========================================

def extraer_rangos_bandgap(indices_frecuencia):
    if len(indices_frecuencia) == 0: return []
    frecuencias_reales = np.linspace(FREQ_INICIO, FREQ_FIN, NUM_FREQS)
    rangos, inicio = [], indices_frecuencia[0]
    prev = inicio
    for idx in indices_frecuencia[1:]:
        if idx == prev + 1: prev = idx
        else:
            rangos.append((frecuencias_reales[inicio], frecuencias_reales[prev]))
            inicio = prev = idx
    rangos.append((frecuencias_reales[inicio], frecuencias_reales[prev]))
    return rangos

def generar_imagen(trans_2d, mat_id, geometria, dir_figures, es_rangle=False):
    plt.figure(figsize=(5, 4))
    cmap, label_y, ext, sufijo = ('plasma', 'Ángulo de rotación ($^\\circ$)', [FREQ_INICIO, FREQ_FIN, TWIST_INICIO, TWIST_FIN], '_rangle') if es_rangle else \
                                 ('viridis', r'Vector de onda $k_x$ ($2\pi/a$)', [FREQ_INICIO, FREQ_FIN, 0, 0.5], '')
    
    plt.imshow(trans_2d, aspect='auto', origin='lower', cmap=cmap, extent=ext)
    plt.colorbar(label='Transmisión')
    plt.xlabel(r'Frecuencia normalizada ($a/\lambda$)')
    plt.ylabel(label_y)
    plt.title(f'{"Rotaciones" if es_rangle else "Bandas"}: {mat_id} ({geometria.capitalize()})')
    plt.tight_layout()
    
    nombre_archivo = f"{mat_id}_{geometria}{sufijo}.png"
    plt.savefig(os.path.join(dir_figures, nombre_archivo), dpi=150)
    plt.close()
    return f"figures/{nombre_archivo}"

def analizar_geometria(datos, llave_transmision, mat_id, dir_figures, es_rangle=False):
    if llave_transmision not in datos or datos[llave_transmision] is None: return None
    trans_1d = np.array(datos[llave_transmision])
    num_filas = len(trans_1d) // NUM_FREQS
    if num_filas == 0 or len(trans_1d) % NUM_FREQS != 0: return None

    trans_2d = trans_1d.reshape((num_filas, NUM_FREQS))
    indices_bandgap = np.where(np.max(trans_2d, axis=0) < 0.12)[0]
    
    gradiente_y, gradiente_f = np.gradient(trans_2d)
    magnitud_gradiente = np.sqrt(gradiente_y**2 + gradiente_f**2)
    anomalias_y, _ = np.where(magnitud_gradiente > np.percentile(magnitud_gradiente, 99.5))
    
    geometria = "plate" if "plate" in llave_transmision else "rod"
    return {
        'rangos_bg': extraer_rangos_bandgap(indices_bandgap),
        'num_anomalias': len(anomalias_y),
        'imagen': generar_imagen(trans_2d, mat_id, geometria, dir_figures, es_rangle),
        'es_rangle': es_rangle
    }

def formatear_celda_bg(rangos):
    if not rangos: return "Ninguno"
    return ", ".join([f"[{r[0]:.3f} - {r[1]:.3f}]" for r in rangos])

# ==========================================
# LÓGICA DE LOS SUBCOMANDOS
# ==========================================

def handle_generate_ranges(args):
    output_dir = "ranges"
    os.makedirs(output_dir, exist_ok=True)
    
    # np.arange no incluye el límite superior, le sumamos la mitad del paso para asegurar que incluya 'fin'
    radios = np.arange(args.inicio, args.fin + (args.paso / 2), args.paso)
    
    print(f"\nIniciando generación de {len(radios)} archivos en el directorio '{output_dir}'...")
    
    for r in radios:
        r_clean = round(r, 4) # Evita errores de precisión flotante en los nombres
        filename = os.path.join(output_dir, f"eps_radio_{r_clean}.pkl")
        
        # Llamamos a la función que ya importaste desde util
        generate_eps_pickle(filename, radius=r)
        
    print("¡Generación de rangos terminada!\n")

def handle_simulate(args):
    carpeta_resultados = "resultados_rcwa"
    os.makedirs(carpeta_resultados, exist_ok=True)

    # --- NUEVA LÓGICA: Si se especifica un directorio de archivos pickle customizados ---
    if args.dir_pickles:
        if not os.path.exists(args.dir_pickles):
            print(f"Error: El directorio de pickles especificado '{args.dir_pickles}' no existe.")
            return
        
        # Buscar todos los archivos .pickle o .pkl en ese directorio
        archivos_custom = [f for f in os.listdir(args.dir_pickles) if f.endswith('.pickle') or f.endswith('.pkl')]
        
        if not archivos_custom:
            print(f"No se encontraron archivos .pickle o .pkl en '{args.dir_pickles}'.")
            return
            
        print(f"Total de perfiles customizados a calcular desde directorio: {len(archivos_custom)}")
        
        for archivo in archivos_custom:
            ruta_pickle = os.path.join(args.dir_pickles, archivo)
            nombre_base = os.path.splitext(archivo)[0] # Nombre sin extensión (ej: 'eps_radio_0.25')
            
            nombre_archivo_salida = f"{carpeta_resultados}/custom_{nombre_base}{'_rangle' if args.rangulo else ''}.pkl"
            if os.path.exists(nombre_archivo_salida):
                print(f"Saltando {archivo} - Ya existe resultado."); continue
                
            print(f"\nCalculando perfil customizado desde archivo: {archivo}...")
            
            # Creamos el lambda que intercepta los argumentos.
            # Ignora 'n_total' porque la matriz eps ya viene dada dentro del archivo pickle cargado por plate_pattern
            custom_pattern_lambda = lambda n_total=None, thickness=0.2: custom_pattern(ruta_pickle, thickness=thickness)
            
            # El grosor por defecto será 0.2 a menos que se use un valor fijo (o el script maneje otra lógica)
            thickness_sim = 0.2 
            
            datos = {
                'material_id': f"custom_{nombre_base}", 
                'formula': f"Custom ({nombre_base})", 
                'e_total': None, # No viene de un CSV de dieléctricos, es una geometría arbitraria
                'archivo_origen_pickle': archivo, # <--- Agregado el nombre del pickle representación al diccionario
                'transmision_plate': calculate_freq_respect_kxs(pattern=custom_pattern_lambda, thickness=thickness_sim),
                'transmision_rangle_plate': calculate_freq_respect_angle(pattern=custom_pattern_lambda, thickness=thickness_sim) if args.rangulo else None,
            }
            
            with open(nombre_archivo_salida, 'wb') as f: 
                pickle.dump(datos, f)
                
        print("\n¡Simulaciones de perfiles customizados terminadas!")
        return # Terminamos aquí para que no intente correr el CSV si se llamó este modo

    # --- LÓGICA ORIGINAL: Simulación basada en el CSV de materiales ---
    df_materiales = pd.read_csv(args.information)

    if args.formula:
        df_materiales = df_materiales[df_materiales['formula_pretty'].isin(args.formula)]
        if df_materiales.empty:
            print(f"No se encontró el compuesto '{args.formula}' en el CSV."); return

    print(f"Total de materiales a calcular: {len(df_materiales)}")

    for _, fila in df_materiales.iterrows():
        mat_id, compuesto, valor_dielectrico = fila['material_id'], fila['formula_pretty'], fila['e_total']
        thickness = fila['thickness'] if args.thickness else 0.2
        if pd.isna(valor_dielectrico): continue
        
        nombre_archivo = f"{carpeta_resultados}/{mat_id}_{compuesto}{'_rangle' if args.rangulo else ''}.pkl"
        if os.path.exists(nombre_archivo):
            print(f"Saltando {mat_id} - Ya existe."); continue
            
        print(f"\nCalculando: {mat_id} - {compuesto}...")
        datos = {
            'material_id': mat_id, 'formula': compuesto, 'e_total': valor_dielectrico,
            'archivo_origen_pickle': None, # No aplica para el flujo estándar
            'transmision_plate': calculate_freq_respect_kxs(n_total=valor_dielectrico, pattern=plate_pattern, thickness=thickness),
            'transmision_rod': calculate_freq_respect_kxs(n_total=valor_dielectrico, pattern=rod_pattern, thickness=thickness),
            'transmision_rangle_plate': calculate_freq_respect_angle(n_total=valor_dielectrico, pattern=plate_pattern, thickness=thickness) if args.rangulo else None,
            'transmision_rangle_rod': calculate_freq_respect_angle(n_total=valor_dielectrico, pattern=rod_pattern, thickness=thickness) if args.rangulo else None
        }
        with open(nombre_archivo, 'wb') as f: pickle.dump(datos, f)
    print("\n¡Simulaciones terminadas!")

def handle_report(args):
    dir_resultados = "resultados_rcwa"
    if not os.path.exists(dir_resultados):
        print(f"Error: '{dir_resultados}' no existe."); return

    regex_pattern = re.compile(args.pattern)
    archivos_pkl = sorted([os.path.join(dir_resultados, f) for f in os.listdir(dir_resultados) if regex_pattern.search(f)])

    if not archivos_pkl:
        print("No hay archivos que coincidan con el patrón."); return

    print(f"\nSe encontraron {len(archivos_pkl)} archivo(s) para procesar:")
    for archivo in archivos_pkl:
        print(f"  - {archivo}")
    
    if input("¿Continuar? (s/n): ").lower() not in ['s', 'si', 'y']: return

    today = datetime.now().strftime("%Y_%m_%d")
    directorio_base = f"reportes/{today}"
    dir_figures = os.path.join(directorio_base, "figures")
    os.makedirs(dir_figures, exist_ok=True)

    datos_procesados = []
    for ruta in archivos_pkl:
        try:
            with open(ruta, 'rb') as f: datos = pickle.load(f)
            res = {
                'material_id': datos['material_id'], 'formula': datos['formula'], 'e_total': datos['e_total'],
                'plate': analizar_geometria(datos, 'transmision_plate', datos['material_id'], dir_figures),
                'rod': analizar_geometria(datos, 'transmision_rod', datos['material_id'], dir_figures),
                'rangle_plate': analizar_geometria(datos, 'transmision_rangle_plate', datos['material_id'], dir_figures, True) if args.rangle else None,
                'rangle_rod': analizar_geometria(datos, 'transmision_rangle_rod', datos['material_id'], dir_figures, True) if args.rangle else None
            }
            if any([res['plate'], res['rod'], res['rangle_plate'], res['rangle_rod']]):
                datos_procesados.append(res)
        except: continue

    if datos_procesados:
        # Aquí llamarías a generar_reporte_latex_modular (omitida por brevedad, pero debe estar en tu script)
        # Nota: Asegúrate de copiar la función generar_reporte_latex_modular del script original aquí.
        print(f"Reporte generado en {directorio_base}")

# ==========================================
# MAIN CON SUBCOMANDOS
# ==========================================

def process():
    parser = argparse.ArgumentParser(description="Herramienta RCWA de Sergio Montoya")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # Subcomando: simulate
    sim_parser = subparsers.add_parser('simulate', help='Ejecuta simulaciones RCWA')
    sim_parser.add_argument('-r', '--rangulo', action='store_true', help="Calcula frecuencia respecto al ángulo")
    sim_parser.add_argument('-t', '--thickness', action='store_true', help="Usa grosor del CSV")
    sim_parser.add_argument('-f', '--formula', type=str, nargs='+', help="Filtra por fórmula")
    sim_parser.add_argument('-i', '--information', type=str, default="./dielectricos_absolutamente_todo.csv")
    sim_parser.add_argument('-d', '--dir_pickles', type=str, default=None, help="Directorio con archivos .pickle/.pkl de geometrías eps personalizadas")

    # Subcomando: report
    rep_parser = subparsers.add_parser('report', help='Genera reporte LaTeX e imágenes')
    rep_parser.add_argument('-p', '--pattern', type=str, default=r'.*\.pkl$', help="Regex para archivos pkl")
    rep_parser.add_argument('--rangle', action='store_true', help="Procesar datos de rotación")

    # Subcomando: generate_ranges
    gen_parser = subparsers.add_parser('generate_ranges', help='Genera archivos pickle para un rango de radios')
    gen_parser.add_argument('inicio', type=float, help='Radio inicial')
    gen_parser.add_argument('fin', type=float, help='Radio final')
    gen_parser.add_argument('paso', type=float, help='Tamaño del paso entre radios')

    args = parser.parse_args()

    if args.command == 'simulate':
        handle_simulate(args)
    elif args.command == 'report':
        handle_report(args)
    elif args.command == 'generate_ranges':
        handle_generate_ranges(args)
    else:
        parser.print_help()
