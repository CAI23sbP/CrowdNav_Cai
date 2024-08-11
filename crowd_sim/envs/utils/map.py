import dataclasses
import numpy as np 
from typing import Union, List

@dataclasses.dataclass(frozen=False)
class Map:
    map_size_m: float
    map_resolution: float
    map_data: np.ndarray
    map_as_obs: List
    obstacle_vertices: List
    border: List
    BORDER_OFFSET: float