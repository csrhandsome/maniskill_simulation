import numpy as np


class Gripper:

    def homing(self) -> None:
        raise NotImplementedError
    
    def get_open_range(self) -> np.ndarray:
        raise NotImplementedError

    def move(self, open_range: float) -> None:
        raise NotImplementedError
    
    def stop(self) -> None:
        raise NotImplementedError