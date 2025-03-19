from __future__ import annotations

from enum import auto
from typing import NamedTuple, Optional
from enum import Enum
from typing import Union
import numpy as np


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    def __str__(self) -> str:
        return self.value

    @classmethod
    def get(cls, name: Union[str, AutoName]) -> AutoName:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unsupported alias! got: {name}. expected one of: {cls.keys()}")

    @classmethod
    def keys(cls) -> list[str]:
        return [e.value for e in cls]


class IntegrationDomain(AutoName):
    THRESHOLD = auto()
    CLUSTER_RATIO = auto()
    LOG_CLUSTER_RATIO = auto()
    
    
def distance_matrix(X):
    '''
    X: np.ndarray
    
    return: upper-triangle of the distance matrix.
    [(0,1), (0,2), ..., (0,n), (1,2), ...]
    '''
    n_data_points = X.shape[0]
    Dx = []
    for i in range(n_data_points):
        dist = np.array(np.linalg.norm(X[i] - X[i + 1:], axis=1)) #distance euclidean
        Dx.extend(list(dist))
        
    Dx = Dx/np.max(Dx) # TODO standardized max distance to 1.0, as ROGI source code does

    return Dx