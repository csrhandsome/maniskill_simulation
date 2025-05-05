import time
import numpy as np
from pymagiclaw import franka
from typing import Optional, Tuple
from pyrobot.robots.arms.base_arm import Arm
from pyrobot.utils.transform_utils import vector2matrix, matrix2vector


class FrankaPanda(Arm):

    def __init__(
        self,
        hostname: str,
        stiffness: Optional[Tuple[int, int]] = (400, 40)
    ) -> None:
        self._start_pose = np.array([0.33913067, 0.01485967, 0.66365492, 3.09444394, -0.06070136, 0.03317684])
        self.robot = franka.Franka(hostname, False)
        self.robot.start_control(*stiffness)
        print(f"Collected to franka {hostname}.")
        
    def move_to_start(self) -> None:
        self.move(self._start_pose)
        time.sleep(3)

    def get_states(self) -> np.ndarray:
        """Get end-effector pose"""
        time.sleep(0.01)
        return matrix2vector(self.robot.read_state())
    
    def move(self, target: np.ndarray, absolute: Optional[bool] = True) -> None:
        """Move end-effector to target"""
        if target.shape == (7,):
            target = vector2matrix(target, rotation_type='quat')
        elif target.shape == (6,):
            target = vector2matrix(target, rotation_type='euler')
        assert target.shape == (4, 4)
        if absolute:
            self.robot.move_absolute_cartesian(target)
        else:
            self.robot.move_relative_cartesian(target)