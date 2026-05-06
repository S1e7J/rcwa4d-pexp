from figures import *

def _aux_desplazar_figura(figura: Figure, dx: float, dy: float) -> Figure:
    """Devuelve una nueva figura con el desplazamiento aplicado.
    Para Trace además cierra el polígono si no lo está.
    """
    if isinstance(figura, Circle):
        x, y = figura.center
        return Circle(center=(x + dx, y + dy), radius=figura.radius)
    elif isinstance(figura, Trace):
        nuevos_puntos = [(x + dx, y + dy) for x, y in figura.points]
        # Cerrar si es necesario
        if nuevos_puntos and nuevos_puntos[0] != nuevos_puntos[-1]:
            nuevos_puntos.append(nuevos_puntos[0])
        return Trace(points=nuevos_puntos)

def desplazar_y_cerrar(puntos, dx, dy):
    """Desplaza una lista de puntos y cierra el polígono si no lo está."""
    return [_aux_desplazar_figura(figura, dx, dy) for figura in puntos]
