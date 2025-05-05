import os
import time
import threading
import numpy as np
import torch.nn as nn
from collections import deque
from typing import Optional, Dict, Union
from pyrobot.robots.arms.base_arm import Arm
from pyrobot.robots.arms.franka import FrankaPanda
from pyrobot.robots.grippers.base_gripper import Gripper
from pyrobot.robots.grippers.magiclaw import MagiClawGripper
from pyrobot.utils.magiclaw_client import MagiClawClient


def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class RealWorldEnv:

    def __init__(
        self,
        camera_ips: Dict,
        arm: Arm,
        gripper: Gripper
    ) -> None:
        self.cameras = list()
        self.clients = dict()
        for camera, ip in camera_ips.items():
            client = MagiClawClient(ip, ['rgb'])
            self.clients[camera] = client
            self.cameras.append(camera)
        self.arm = arm
        self.gripper = gripper
        time.sleep(2)
        
    def get_obs(self) -> Dict:
        obs = dict()
        for camera, client in self.clients.items():
            obs[f'{camera}_images'] = client.get_obs()['rgb']
        ee_pose = self.arm.get_states()
        gripper_width = np.array([self.gripper.get_open_range()], dtype=np.float32)
        low_dims = np.concatenate([ee_pose, gripper_width], dtype=np.float32)
        obs['low_dims'] = low_dims
        return obs

    def reset(self) -> Dict:
        self.gripper.homing()
        self.arm.move_to_start()
        return self.get_obs()

    def step(self, action: np.ndarray) -> None:
        self.arm.move(action[:6])
        self.gripper.move(float(action[6]))

    def shutdown(self) -> None:
        self.gripper.stop()
        os._exit(0)


class MultiStepWrapper:

    def __init__(
        self,
        env: RealWorldEnv,
        obs_horizon: Optional[int] = 2,
        action_horizon: Optional[int] = 8,
        max_episode_steps: Union[int, None] = None,
        enable_temporal_ensemble: Optional[bool] = True,
        frequency: Optional[int] = 30
    ) -> None:
        self.env = env
        self.global_step = 0
        self.obs = deque(maxlen=obs_horizon + 1)
        if enable_temporal_ensemble:
            assert max_episode_steps is not None
            T = max_episode_steps
            H = action_horizon
            D = 7
            self.all_time_actions = np.zeros((T, T + H, D), dtype=np.float32)
        
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.max_episode_steps = max_episode_steps
        self.enable_temporal_ensemble = enable_temporal_ensemble
        self.frequency = frequency


class TaskRunner:

    def __init__(
        self,
        camera_ips: Dict,
        arm: Arm,
        gripper: Gripper,
        obs_horizon: Optional[int] = 2,
        action_horizon: Optional[int] = 8,
        max_episode_steps: Union[int, None] = None,
        enable_temporal_ensemble: Optional[bool] = True,
        frequency: Optional[int] = 30
    ) -> None:
        self.cameras = list()
        self.clients = dict()
        for camera, ip in camera_ips.items():
            client = MagiClawClient(ip, ['rgb'])
            self.clients[camera] = client
            self.cameras.append(camera)

        self.arm = arm
        self.gripper = gripper
        
        self.obs = deque(maxlen=obs_horizon + 1)
        if enable_temporal_ensemble:
            assert max_episode_steps is not None
            T = max_episode_steps
            H = action_horizon
            D = 7
            self.all_time_actions = np.zeros((T, T + H, D), dtype=np.float32)
        
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.max_episode_steps = max_episode_steps
        self.enable_temporal_ensemble = enable_temporal_ensemble
        self.frequency = frequency
        self.global_step = 0
        time.sleep(2)
        # self.thread = threading.Thread(target=self.obs_buffer)
        # self.thread.start()
        # time.sleep(1)

    # def obs_buffer(self) -> None:
    #     dt = 1 / self.frequency
    #     start_time = time.time()
    #     obs = dict()
    #     for camera, client in self.clients.items():
    #         obs[camera] = client.get_obs()['rgb']
    #     ee_pose = self.arm.get_states()
    #     gripper_width = np.array([self.gripper.get_open_range()], dtype=np.float32)
    #     low_dims = np.concatenate([ee_pose, gripper_width], dtype=np.float32)
    #     obs['low_dims'] = low_dims
    #     self.obs.append(obs)
    #     time.sleep(max(0, dt - (time.time() - start_time)))

    # def get_obs(self) -> Dict:
    #     images = {
    #         camera: stack_last_n_obs([obs[camera] for obs in self.obs], self.obs_horizon)
    #         for camera in self.cameras
    #     }
    #     low_dims = stack_last_n_obs([obs['low_dims'] for obs in self.obs], self.obs_horizon)
    #     obs = dict(
    #         images=images,
    #         low_dims=low_dims
    #     )
    #     return obs

    def get_obs(self) -> Dict:
        obs = dict()
        for camera, client in self.clients.items():
            obs[f'{camera}_images'] = client.get_obs()['rgb']
        ee_pose = self.arm.get_states()
        gripper_width = np.array([self.gripper.get_open_range()], dtype=np.float32)
        low_dims = np.concatenate([ee_pose, gripper_width], dtype=np.float32)
        obs['low_dims'] = low_dims
        return obs

    def reset(self) -> Dict:
        self.global_step = 0
        self.obs = deque(maxlen=self.obs_horizon + 1)
        if self.enable_temporal_ensemble:
            self.all_time_actions = np.zeros_like(self.all_time_actions, dtype=np.float32)
        self.gripper.homing()
        self.arm.move_to_start()
        return self.get_obs()

    def move(self, action: np.ndarray) -> None:
        self.arm.move(action[:6])
        self.gripper.move(float(action[6]))

    def step(self, actions: np.ndarray) -> None:
        """
        Args:
            actions (np.ndarray): (action_horizon, action_shape).
        """
        if self.enable_temporal_ensemble:
            t = self.global_step
            H = self.action_horizon
            self.all_time_actions[[t], t: t + H] = actions
            current_actions = self.all_time_actions[:, t]
            actions_populated = np.all(current_actions != 0, axis=1)
            current_actions = current_actions[actions_populated]
            exp_weights = np.exp(-0.01 * np.arange(len(current_actions)))
            exp_weights = (exp_weights / exp_weights.sum())[..., np.newaxis]
            action = (current_actions * exp_weights).sum(axis=0)
            self.move(action)
            self.global_step += 1
        else:
            for action in actions:
                if self.max_episode_steps is not None:
                    if self.global_step >= self.max_episode_steps:
                        break
                self.move(action)
                self.global_step += 1

    def shutdown(self) -> None:
        self.gripper.stop()
        self.thread.join()
        os._exit(0)


import cv2
import numpy as np
class TestGripper:

    def homing(self) -> None:
        pass
    
    def get_open_range(self) -> np.ndarray:
        return 0

    def move(self, open_range: float) -> None:
        pass
    
    def stop(self) -> None:
        pass
    

if __name__ == "__main__":
    camera_ips = dict(
        front_camera="ws://10.16.63.1:8080",
        # wrist_camera="ws://10.16.3.51:8080"
    )
    arm = FrankaPanda("192.168.1.100", stiffness=(400, 40))
    # gripper = MagiClawGripper()
    gripper = TestGripper()
    task_runner = TaskRunner(
        camera_ips=camera_ips,
        arm=arm,
        gripper=gripper,
        max_episode_steps=200,
        frequency=30
    )
    while True:
        obs = task_runner.get_obs()
        image = obs['front_camera_images']
        low_dims = obs['low_dims']
        time.sleep(1 / 30)
