import os
import time
import tqdm
import torch
import hydra
import numpy as np
from pathlib import Path
from collections import deque
from omegaconf import OmegaConf
from typing import Optional, Dict
from pyrobot.robots.arms.franka import FrankaPanda
from pyrobot.robots.grippers.magiclaw import MagiClawGripper
from pyrobot.utils.magiclaw_client import MagiClawClient
from icon.policies.base_policy import BasePolicy
from icon.utils.pytorch_utils import to
from icon.utils.gym_utils.multistep_wrapper import stack_last_n_obs


class TaskRunner:

    def __init__(
        self,
        policy: BasePolicy,
        camera_ips: Dict,
        arm: FrankaPanda,
        gripper: MagiClawGripper,
        obs_horizon: int,
        action_horizon: int,
        max_episode_steps: Optional[int] = 100,
        frequency: Optional[int] = 30,
        device: Optional[str] = 'cuda'
    ) -> None:
        self.device = torch.device(device)
        self.policy = policy
        self.policy.to(device)
        self.policy.eval()

        self.cameras = list()
        self.clients = dict()
        for camera, ip in camera_ips.items():
            client = MagiClawClient(ip, ['rgb'])
            self.clients[camera] = client
            self.cameras.append(camera)
        self.arm = arm
        self.gripper = gripper
        time.sleep(2)
        
        self.global_step = 0
        self.obs = deque(maxlen=obs_horizon + 1)
        self.all_time_actions = np.zeros(
            (max_episode_steps, max_episode_steps + action_horizon, 7),
            dtype=np.float32
        )
        
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.max_episode_steps = max_episode_steps
        self.frequency = frequency

    def obs_buffer(self) -> None:
        obs = dict()
        for camera, client in self.clients.items():
            obs[f'{camera}_images'] = client.get_obs()['rgb']
        ee_pose = self.arm.get_states()
        gripper_width = np.array([self.gripper.get_open_range()], dtype=np.float32)
        low_dims = np.concatenate([ee_pose, gripper_width], dtype=np.float32)
        obs['low_dims'] = low_dims
        self.obs.append(obs)

    def get_obs(self) -> Dict:
        images = dict()
        for camera in self.cameras:
            image = stack_last_n_obs([obs[f'{camera}_images'] for obs in self.obs], self.obs_horizon)
            image = torch.from_numpy(image).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
            images[camera] = image
        low_dims = stack_last_n_obs([obs['low_dims'] for obs in self.obs], self.obs_horizon)
        low_dims = torch.from_numpy(low_dims).float().unsqueeze(0)
        obs = dict(
            images=images,
            low_dims=low_dims
        )
        return obs

    def reset(self) -> None:
        self.gripper.homing()
        self.arm.move_to_start()
        time.sleep(1)

    def step(self, actions: np.ndarray) -> None:
        """
        Args:
            actions (np.ndarray): (action_horizon, action_shape).
        """
        t = self.global_step
        H = self.action_horizon
        self.all_time_actions[[t], t: t + H] = actions
        current_actions = self.all_time_actions[:, t]
        actions_populated = np.all(current_actions != 0, axis=1)
        current_actions = current_actions[actions_populated]
        exp_weights = np.exp(-0.01 * np.arange(len(current_actions)))
        exp_weights = (exp_weights / exp_weights.sum())[..., np.newaxis]
        action = (current_actions * exp_weights).sum(axis=0)
        self.arm.move(action[:6])
        self.gripper.move(float(action[6]))
        self.global_step += 1

    def run(self) -> None:
        self.reset()
        dt = 1 / self.frequency
        pbar = tqdm.tqdm(
            total=self.max_episode_steps,
            leave=False,
            mininterval=5.0
        )
        self.obs_buffer()
        obs = self.get_obs()
        for _ in range(self.max_episode_steps):
            start_time = time.time()
            to(obs, self.device)
            with torch.no_grad():
                actions = self.policy.predict_action(obs)['actions']
            print("policy inference: ", time.time() - start_time)
            actions = actions.detach().to('cpu').squeeze(0).numpy()
            self.step(actions)
            print("step: ", time.time() - start_time)
            time.sleep(max(0, dt - (time.time() - start_time)))
            # policy inference: 0.085s
            self.obs_buffer()
            obs = self.get_obs()
            pbar.update(1)
        self.shutdown()

    def shutdown(self) -> None:
        self.gripper.stop()
        os._exit(0)


class TestGripper:
    def homing(self) -> None:
        pass
    def get_open_range(self) -> np.ndarray:
        return 0
    def move(self, open_range: float) -> None:
        pass
    def stop(self) -> None:
        pass

OmegaConf.register_new_resolver("eval", eval, replace=True)

if __name__ == "__main__":
    task = "put_doll_in_box"
    algo = "diffusion_transformer"
    checkpoint = "/home/wangjl/Downloads/checkpoint.pth"
    with hydra.initialize_config_dir(
        config_dir=str(Path(__file__).parent.parent.joinpath("icon/configs")),
        version_base="1.2" 
    ):
        overrides = [f'task={task}', f'algo={algo}']
        cfg = hydra.compose(config_name="config", overrides=overrides)
        policy: BasePolicy = hydra.utils.instantiate(cfg.algo.policy)
        state_dicts = torch.load(checkpoint, map_location='cpu')
        policy.load_state_dicts(state_dicts)
        camera_ips = dict(
            front_camera="ws://192.168.31.71:8080",
            wrist_camera="ws://192.168.31.193:8080"
        )
        arm = FrankaPanda("192.168.1.100", stiffness=(400, 40))
        gripper = MagiClawGripper()
        # gripper = TestGripper()
        task_runner = TaskRunner(
            policy=policy,
            camera_ips=camera_ips,
            arm=arm,
            gripper=gripper,
            obs_horizon=cfg.algo.obs_horizon,
            action_horizon=cfg.algo.action_horizon,
            max_episode_steps=200,
            frequency=10
        )
        task_runner.run()