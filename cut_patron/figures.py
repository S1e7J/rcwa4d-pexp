from dataclasses import dataclass
from typing import Union, List, Tuple

@dataclass
class Circle:
    center: Tuple[float, float]
    radius: float

@dataclass
class Trace:
    points: List[Tuple[float, float]]

Figure = Union[Circle, Trace]
