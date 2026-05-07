import pickle
import numpy as np
import os
import argparse
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt

# Evitar que matplotlib intente abrir ventanas visuales durante el bucle
plt.switch_backend('Agg') 

# Dimensiones de la simulación
NUM_FREQS = 141
FREQ_INICIO = 0.7
FREQ_FIN = 0.84

# Configuraciones de rotación (asumiendo 0 a 90 grados por defecto, ajustables si es necesario)
TWIST_INICIO = 0.0
TWIST_FIN = 90.0

def extraer_rangos_bandgap(indices_frecuencia):
    if len(indices_frecuencia) == 0:
        return []
        
    frecuencias_reales = np.linspace(FREQ_INICIO, FREQ_FIN, NUM_FREQS)
    rangos = []
    inicio = indices_frecuencia[0]
    prev = inicio
    
    for idx in indices_frecuencia[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            rangos.append((frecuencias_reales[inicio], frecuencias_reales[prev]))
            inicio = idx
            prev = idx
            
    rangos.append((frecuencias_reales[inicio], frecuencias_reales[prev]))
    return rangos

def generar_imagen_bandas(trans_2d, mat_id, geometria, dir_figures):
    """Genera y guarda el mapa de calor de las bandas (kx)."""
    plt.figure(figsize=(5, 4))
    plt.imshow(trans_2d, aspect='auto', origin='lower', cmap='viridis', 
               extent=[FREQ_INICIO, FREQ_FIN, 0, 0.5])
    
    plt.colorbar(label='Transmisión')
    plt.xlabel(r'Frecuencia normalizada ($a/\lambda$)')
    plt.ylabel(r'Vector de onda $k_x$ ($2\pi/a$)')
    plt.title(f'Bandas: {mat_id} ({geometria.capitalize()})')
    plt.tight_layout()
    
    nombre_archivo = f"{mat_id}_{geometria}.png"
    ruta_absoluta = os.path.join(dir_figures, nombre_archivo)
    plt.savefig(ruta_absoluta, dpi=150)
    plt.close()
    
    return f"figures/{nombre_archivo}"

def generar_imagen_rangle(trans_2d, mat_id, geometria, dir_figures):
    """Genera y guarda el mapa de calor de las bandas por rotación (twist)."""
    plt.figure(figsize=(5, 4))
    # Usamos otro colormap (plasma) para diferenciar visualmente rotaciones de kx
    plt.imshow(trans_2d, aspect='auto', origin='lower', cmap='plasma', 
               extent=[FREQ_INICIO, FREQ_FIN, TWIST_INICIO, TWIST_FIN])
    
    plt.colorbar(label='Transmisión')
    plt.xlabel(r'Frecuencia normalizada ($a/\lambda$)')
    plt.ylabel(r'Ángulo de rotación ($^\circ$)')
    plt.title(f'Rotaciones: {mat_id} ({geometria.capitalize()})')
    plt.tight_layout()
    
    nombre_archivo = f"{mat_id}_{geometria}_rangle.png"
    ruta_absoluta = os.path.join(dir_figures, nombre_archivo)
    plt.savefig(ruta_absoluta, dpi=150)
    plt.close()
    
    return f"figures/{nombre_archivo}"

def analizar_geometria(datos, llave_transmision, mat_id, dir_figures, es_rangle=False):
    if llave_transmision not in datos:
        return None
        
    trans_1d = np.array(datos[llave_transmision])
    
    # Calcular dinámicamente las filas (kxs o twists) basándose en NUM_FREQS
    num_filas = len(trans_1d) // NUM_FREQS
    if num_filas == 0 or len(trans_1d) % NUM_FREQS != 0:
        return None

    trans_2d = trans_1d.reshape((num_filas, NUM_FREQS))
    
    umbral_transmision = 0.12 
    transmision_maxima_por_frecuencia = np.max(trans_2d, axis=0)
    indices_bandgap = np.where(transmision_maxima_por_frecuencia < umbral_transmision)[0]
    rangos_bg = extraer_rangos_bandgap(indices_bandgap)
    
    gradiente_y, gradiente_f = np.gradient(trans_2d)
    magnitud_gradiente = np.sqrt(gradiente_y**2 + gradiente_f**2)
    umbral_anomalia = np.percentile(magnitud_gradiente, 99.5)
    anomalias_y, _ = np.where(magnitud_gradiente > umbral_anomalia)
    
    geometria = "plate" if "plate" in llave_transmision else "rod"
    
    if es_rangle:
        ruta_img = generar_imagen_rangle(trans_2d, mat_id, geometria, dir_figures)
    else:
        ruta_img = generar_imagen_bandas(trans_2d, mat_id, geometria, dir_figures)
    
    return {
        'rangos_bg': rangos_bg,
        'num_anomalias': len(anomalias_y),
        'imagen': ruta_img,
        'es_rangle': es_rangle
    }

def analizar_para_reporte(ruta_archivo, dir_figures, procesar_rangle=False):
    try:
        with open(ruta_archivo, 'rb') as f:
            datos = pickle.load(f)
    except Exception:
        return None

    formula = datos.get('formula', '')
    mat_id = datos.get('material_id', 'N_A')
    
    if not formula:
        return None

    resultados = {
        'material_id': mat_id,
        'formula': formula,
        'e_total': datos.get('e_total', 'N/A'),
        'plate': None,
        'rod': None,
        'rangle_plate': None,
        'rangle_rod': None
    }
    
    # Procesamiento estándar (kx)
    resultados['plate'] = analizar_geometria(datos, 'transmision_plate', mat_id, dir_figures)
    resultados['rod'] = analizar_geometria(datos, 'transmision_rod', mat_id, dir_figures)
    
    # Procesamiento de rotaciones (twist)
    if procesar_rangle:
        resultados['rangle_plate'] = analizar_geometria(datos, 'transmision_rangle_plate', mat_id, dir_figures, es_rangle=True)
        resultados['rangle_rod'] = analizar_geometria(datos, 'transmision_rangle_rod', mat_id, dir_figures, es_rangle=True)

    # Si el diccionario no tiene NADA válido, ignorar
    if not any([resultados['plate'], resultados['rod'], resultados['rangle_plate'], resultados['rangle_rod']]):
        return None

    return resultados

def formatear_celda_bg(rangos):
    if not rangos:
        return "Ninguno"
    lista_str = [f"[{r[0]:.3f} - {r[1]:.3f}]" for r in rangos]
    return ", ".join(lista_str)

def generar_reporte_latex_modular(datos_procesados, directorio_base):
    dir_materiales = os.path.join(directorio_base, "materiales")
    os.makedirs(dir_materiales, exist_ok=True)
    
    datos_por_material = {}
    for d in datos_procesados:
        mat = d['formula']
        if mat not in datos_por_material:
            datos_por_material[mat] = []
        datos_por_material[mat].append(d)

    archivos_incluidos = []

    for formula, lista_resultados in datos_por_material.items():
        lista_resultados.sort(key=lambda x: str(x['material_id']))
        ruta_archivo_mat = os.path.join(dir_materiales, f"{formula}.tex")
        
        lineas_mat = [f"\\section{{{formula}}}"]
        
        for d in lista_resultados:
            mat_id = str(d['material_id']).replace("_", "\\_")
            e_tot = f"{d['e_total']:.2f}" if isinstance(d['e_total'], (int, float)) else str(d['e_total'])
            
            # --- CREACIÓN DE LA TABLA ---
            lineas_mat.extend([
                f"\\subsection*{{Material ID: {mat_id}}}",
                f"\\textbf{{Constante Dieléctrica ($\\epsilon_{{total}}$):}} {e_tot} \\\\[0.5em]",
                r"\begin{table}[h!]",
                r"\centering",
                r"\begin{tabular}{lcc}",
                r"\toprule",
                r"\textbf{Métrica} & \textbf{Lámina (Plate)} & \textbf{Cilindros (Rod)} \\",
                r"\midrule"
            ])
            
            # Extraer métricas estándar
            bg_plate = formatear_celda_bg(d['plate']['rangos_bg']) if d['plate'] else "N/A"
            res_plate = str(d['plate']['num_anomalias']) if d['plate'] else "N/A"
            bg_rod = formatear_celda_bg(d['rod']['rangos_bg']) if d['rod'] else "N/A"
            res_rod = str(d['rod']['num_anomalias']) if d['rod'] else "N/A"
            
            lineas_mat.extend([
                f"Bandgaps ($k_x$) & {bg_plate} & {bg_rod} \\\\",
                f"Resonancias ($k_x$) & {res_plate} & {res_rod} \\\\"
            ])

            # Extraer métricas de rotación (si existen)
            if d.get('rangle_plate') or d.get('rangle_rod'):
                bg_rangle_plate = formatear_celda_bg(d['rangle_plate']['rangos_bg']) if d['rangle_plate'] else "N/A"
                res_rangle_plate = str(d['rangle_plate']['num_anomalias']) if d['rangle_plate'] else "N/A"
                bg_rangle_rod = formatear_celda_bg(d['rangle_rod']['rangos_bg']) if d['rangle_rod'] else "N/A"
                res_rangle_rod = str(d['rangle_rod']['num_anomalias']) if d['rangle_rod'] else "N/A"
                
                lineas_mat.extend([
                    r"\midrule",
                    f"Bandgaps (Rotación) & {bg_rangle_plate} & {bg_rangle_rod} \\\\",
                    f"Resonancias (Rotación) & {res_rangle_plate} & {res_rangle_rod} \\\\"
                ])

            lineas_mat.extend([
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}"
            ])
            
            # --- INSERCIÓN DE LAS IMÁGENES ESTÁNDAR (KX) ---
            if d.get('plate') or d.get('rod'):
                lineas_mat.extend([
                    r"\begin{figure}[H]",
                    r"\centering"
                ])
                if d['plate']:
                    lineas_mat.append(f"\\includegraphics[width=0.48\\textwidth]{{{d['plate']['imagen']}}}")
                if d['rod']:
                    lineas_mat.append(f"\\hfill\n\\includegraphics[width=0.48\\textwidth]{{{d['rod']['imagen']}}}")
                
                lineas_mat.extend([
                    f"\\caption{{Estructuras de bandas fotónicas ($k_x$) para {mat_id}. Izquierda: Lámina. Derecha: Cilindros.}}",
                    r"\end{figure}",
                    r"\vspace{1em}"
                ])

            # --- INSERCIÓN DE LAS IMÁGENES DE ROTACIÓN (RANGLE) ---
            if d.get('rangle_plate') or d.get('rangle_rod'):
                lineas_mat.extend([
                    r"\begin{figure}[H]",
                    r"\centering"
                ])
                if d['rangle_plate']:
                    lineas_mat.append(f"\\includegraphics[width=0.48\\textwidth]{{{d['rangle_plate']['imagen']}}}")
                if d['rangle_rod']:
                    lineas_mat.append(f"\\hfill\n\\includegraphics[width=0.48\\textwidth]{{{d['rangle_rod']['imagen']}}}")
                
                lineas_mat.extend([
                    f"\\caption{{Mapas de transmisión por rotación (twist) para {mat_id}. Izquierda: Lámina. Derecha: Cilindros.}}",
                    r"\end{figure}",
                    r"\vspace{1em}"
                ])
            
            lineas_mat.extend([
                r"\hrule", 
                r"\vspace{1em}"
            ])
            
        lineas_mat.append(r"\clearpage")
        
        with open(ruta_archivo_mat, 'w', encoding='utf-8') as f:
            f.write("\n".join(lineas_mat))
            
        archivos_incluidos.append(f"materiales/{formula}")

    ruta_main = os.path.join(directorio_base, "main.tex")
    lineas_main = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}", 
        r"\usepackage{float}", 
        r"\usepackage{geometry}",
        r"\geometry{a4paper, margin=1in}",
        r"\begin{document}",
        r"",
        r"\title{Resultados de Simulación RCWA - Comparativa de Geometrías y Rotaciones}",
        r"\author{Sergio Montoya Ramirez (Reporte Automatizado)}",
        r"\date{\today}",
        r"\maketitle",
        r"",
        r"El presente reporte detalla los \textit{Photonic Bandgaps} y la estructura de bandas detectados en simulaciones RCWA, evaluando tanto barridos de vector de onda ($k_x$) como de rotación (twist).",
        r"Se asume un umbral de transmisión del 12\% para el cálculo de bandgaps, mitigando los efectos de fuga numérica por bajos órdenes de difracción.",
        r"\clearpage"
    ]
    
    for archivo in archivos_incluidos:
        lineas_main.append(f"\\input{{{archivo}}}")
        
    lineas_main.append(r"\end{document}")
    
    with open(ruta_main, 'w', encoding='utf-8') as f:
        f.write("\n".join(lineas_main))
        
    print(f"\n[✓] Estructura modular generada en: {directorio_base}/")
    print(f"    -> Documento principal: main.tex")
    print(f"    -> Imágenes generadas en: {os.path.join(directorio_base, 'figures')}/")

# ==========================================
# EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar reportes de simulaciones RCWA.")
    parser.add_argument('--rangle', action='store_true', help="Procesar también los archivos de rotación (rangle.pkl) y generar sus gráficas.")
    args = parser.parse_args()

    today = datetime.now().strftime("%Y_%m_%d")
    directorio_base = f"reportes/{today}"
    dir_figures = os.path.join(directorio_base, "figures")
    
    os.makedirs(directorio_base, exist_ok=True)
    os.makedirs(dir_figures, exist_ok=True)
    
    # Filtrar archivos según la flag
    if args.rangle:
        archivos_pkl = glob("resultados_rcwa/*rangle.pkl")
        print("Modo RANGLE activado. Procesando archivos *rangle.pkl ...")
    else:
        # Si no está la flag, procesamos todos los pkl excepto los rangle para evitar errores
        archivos_pkl = [f for f in glob("resultados_rcwa/*.pkl") if not f.endswith('rangle.pkl')]
        print("Modo estándar. Procesando archivos *.pkl (omitiendo rotaciones)...")

    datos_procesados = []

    print("Procesando simulaciones y generando gráficos. Esto puede tomar unos segundos...")
    
    for archivo in archivos_pkl:
        resultado = analizar_para_reporte(archivo, dir_figures, procesar_rangle=args.rangle)
        if resultado is not None:
            datos_procesados.append(resultado)

    if datos_procesados:
        generar_reporte_latex_modular(datos_procesados, directorio_base)
    else:
        print("No se encontraron datos válidos para generar el reporte.")

