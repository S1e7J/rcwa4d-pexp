from figures import *
from general_figure_utils import *
import ezdxf


def _aux_dibujar(figura: Figure, msp):
    if isinstance(figura, Circle):
        msp.add_circle(figura.center, radius = figura.radius)
    elif isinstance(figura, Trace):
        msp.add_lwpolyline(figura.points)

def _dibujar(figures: List[Figure], msp):
    for figura in figures:
        _aux_dibujar(figura, msp)


def generar_malla_cad(figuras_internas: List[Figure], n_rep_x, n_rep_y, lado_celda, nombre_archivo="resultado.dxf"):
    """
    puntos_celda: Lista de coordenadas (x, y) que forman la figura interna.
    n_rep_x, n_rep_y: Cuántas veces se repite la celda.
    lado_celda: Tamaño del cuadrado contenedor (ej: 10.0 mm).
    """
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    for i in range(n_rep_x):
        for j in range(n_rep_y):
            off_x = i * lado_celda
            off_y = j * lado_celda

            geometrias_desplazada = desplazar_y_cerrar(figuras_internas, off_x, off_y)

            _dibujar(geometrias_desplazada, msp)
            

    ancho_total = n_rep_x * lado_celda
    alto_total = n_rep_y * lado_celda
    puntos_borde = [
        (0, 0), (ancho_total, 0), 
        (ancho_total, alto_total), (0, alto_total), (0, 0)
    ]
    msp.add_lwpolyline(puntos_borde)

    doc.saveas(nombre_archivo)
    print(f"Patrón de {n_rep_x}x{n_rep_y} guardado en {nombre_archivo}")
