import numpy as np
from typing import Optional


class Arm:

    def __init__(self) -> None:
        self._start_pose = None

    def move_to_start(self) -> None:
        raise NotImplementedError

    def get_states(self) -> np.ndarray:
        raise NotImplementedError
    
    def move(self, target: np.ndarray, absolute: Optional[bool] = True) -> None:
        raise NotImplementedError
    
    @property
    def start_pose(self):
        return self._start_pose