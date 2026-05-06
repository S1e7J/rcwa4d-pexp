import numpy as np
from matplotlib.path import Path
from figures import *

number = int | float | complex
DEG = np.pi/180

def _put_figure(figure: Figure, eps, xs, ys):
    if isinstance(figure, Circle):
        h, k = figure.center
        eps[(xs - h)**2 + (ys - k)**2 < figure.radius**2] = 1 
    elif isinstance(figure, Trace):
        vertices = np.array(figure.points)
        if not np.allclose(vertices[0], vertices[-1]):
            vertices = np.vstack([vertices, vertices[0]])
        polygon = Path(vertices)
        points = np.column_stack((xs.ravel(), ys.ravel()))
        mask = polygon.contains_points(points)
        eps.ravel()[mask] = 1

def generar_patron(figuras_internas: List[Figure], lado_celda, n_total: number = 4):
    Ny = Nx = 1000 
    eps = np.ones([Ny,Nx]) * n_total 
    
    xs, ys = np.linspace(0, lado_celda,Nx), np.linspace(lado_celda, 0,Ny)
    xs, ys = np.meshgrid(xs,ys)
    for figure in figuras_internas:
        _put_figure(figure, eps, xs, ys)
    
    return eps
