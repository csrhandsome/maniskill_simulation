import os
import h5py
import time
import imageio
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List
from pynput.keyboard import Listener
from pyrobot.utils.magiclaw_client import MagiClawClient
from pyrobot.utils.transform_utils import matrix2vector
from icon.envs.robosuite_env import RobosuiteEnv


class DemoCollector:

    def __init__(
        self,
        save_dir: str,
        teleop_ip: str,
        task: str,
        cameras: List,
        shape_meta: Dict,
    ) -> None:
        self.save_dir = Path(save_dir).expanduser().absolute()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.teleop = MagiClawClient(teleop_ip, ['pose', 'width'])
        self.cameras = cameras
        self.env = RobosuiteEnv(
            task=task,
            cameras=cameras,
            shape_meta=shape_meta,
            render_mode='human',
            gpu_id=0
        )
        
        self.global_step = 0
        self.local_step = 0
        self.is_recording = False
        self.is_running = True
        self.teleop_init_pose = None
        self.last_gripper_width = 1.0
        self.data = {
            **{f'{camera}_images': list() for camera in self.cameras},
            'low_dims': list(),
            'actions': list()
        }

        self.listener = Listener(on_press=self.on_press)
        self.listener.start()
        self.thread_teleop = threading.Thread(target=self.run_teleop)
        self.thread_teleop.start()
        self.thread = threading.Thread(target=self.env.render)
        self.thread.start()

    def on_press(self, key: str) -> None:
        try:
            if hasattr(key, 'char') and key.char:
                if key.char == 'r':
                    # Reset
                    self.local_step = 0
                    self.reset()
                elif key.char == 'b':
                    # Begin recording
                    self.is_recording = True
                elif key.char == 'e':
                    # Stop recording
                    self.is_recording = False
                elif key.char == 's':
                    # Save data
                    self.is_recording = False
                    self.save()
                elif key.char == 'q':
                    # Quit
                    self.quit()
        except AttributeError as e:
            print(e)
        
    def reset(self) -> None:
        self.data = {
            **{f'{camera}_images': list() for camera in self.cameras},
            'low_dims': list(),
            'actions': list()
        }
        obs = self.env.reset()
        for key, val in obs.items():
            self.data[key].append(val)

    def save(self) -> None:
        self.global_step += 1
        if len(self.data['actions']) > 0:
            save_dir = self.save_dir.joinpath(f'episode_{str(self.global_step).zfill(3)}')
            save_dir.mkdir(exist_ok=True)

            video_dir = save_dir.joinpath("videos")
            video_dir.mkdir(exist_ok=True)
            for camera in self.cameras:
                writer = imageio.get_writer(str(video_dir.joinpath(f"{camera}.mp4")), fps=24)
                images = self.data[f'{camera}_images']
                for image in images:
                    writer.append_data(image)

            save_dir.joinpath("masks").mkdir(exist_ok=True)

            with h5py.File(save_dir.joinpath("states.h5"), 'w') as f:
                f['/low_dims'] = np.stack(self.data['low_dims'])
                actions = np.stack(self.data['actions'])
                # Convert to delta action
                actions = np.concatenate([actions[1:] - actions[:-1], np.zeros((1, 7))])
                f['/actions'] = actions
                f.close()

    def quit(self) -> None:
        self.is_recording = False
        self.is_running = False
        self.thread_teleop.join()
        self.thread.join()
        os._exit(0)

    def run_teleop(self) -> None:
        while self.is_running:
            if self.is_recording:
                if self.local_step == 0:
                    self.teleop_init_pose = matrix2vector(self.teleop.get_obs()['pose'])
                gripper_width = self.teleop.get_obs()['width']
                if gripper_width == -1.0:
                    gripper_width = self.last_gripper_width
                else:
                    self.last_gripper_width = gripper_width
                gripper_action = np.array([1 - 2 * gripper_width])
                teleop_pose = matrix2vector(self.teleop.get_obs()['pose'])
                teleop_delta_pose = teleop_pose - self.teleop_init_pose
                self.teleop_init_pose = teleop_pose
                arm_action = np.array([
                    -teleop_delta_pose[2],
                    -teleop_delta_pose[0],
                    teleop_delta_pose[1],
                    -teleop_delta_pose[5],
                    -teleop_delta_pose[3],
                    teleop_delta_pose[4]]
                )
                action = np.concatenate([arm_action, gripper_action])
                obs, _, _, _ = self.env.step(action)
                # self.env.render()
                for key, val in obs.items():
                    self.data[key].append(val)
                self.local_step += 1
            else:
                time.sleep(0.1)
    
    # def run(self) -> None:
    #     dt = 1 / self.frequency
    #     while self.is_running:
    #         if self.is_recording:
    #             start_time = time.time()
    #             # RGB and proprioception observations
    #             images = dict()
    #             for camera, client in self.clients.items():
    #                 images[camera] = client.get_obs()['rgb']

    #             ee_pose = self.arm.get_states()
    #             gripper_width = np.array([self.gripper.get_open_range()], dtype=np.float64)
    #             low_dims = np.concatenate([ee_pose, gripper_width], dtype=np.float64)
    #             # Actions
    #             actions = np.concatenate([ee_pose, gripper_width], dtype=np.float64)
                
    #             for camera in self.cameras:
    #                 self.data[f'{camera}_images'].append(images[camera])
    #             self.data['low_dims'].append(low_dims)
    #             self.data['actions'].append(actions)
    #             self.local_step += 1
    #             time.sleep(max(0, dt - (time.time() - start_time)))
    #         else:
    #             time.sleep(0.1)


if __name__ == "__main__":
    # task = "stack_cube"
    # split = "train"

    # dc = DemoCollector(
    #     save_dir=f"data/{task}/{split}",
    #     teleop_ip="ws://192.168.31.193:8080",
    #     task=task,
    #     cameras=['agentview', 'robot0_eye_in_hand'],
    #     shape_meta=dict(
    #         images=1024,
    #         low_dims=14,
    #         actions=7
    #     )
    # )

    client = MagiClawClient("ws://192.168.31.193:8080")
    time.sleep(2)
    env = RobosuiteEnv(
        task="stack_cube",
        cameras=['agentview', 'robot0_eye_in_hand'],
        shape_meta=dict(
            images=1024,
            low_dims=14,
            actions=7
        ),
        render_mode='human',
        gpu_id=0
    )
    env.reset()
    last_pose = None
    for i in range(200):
        if i == 0:
            last_pose = matrix2vector(client.get_obs()['pose'])
        pose = matrix2vector(client.get_obs()['pose'])
        teleop_delta_pose = pose - last_pose
        last_pose = pose
        scale = 3
        action = np.array([
            -teleop_delta_pose[2] * scale,
            -teleop_delta_pose[0] * scale,
            teleop_delta_pose[1] * scale,
            -teleop_delta_pose[5],
            -teleop_delta_pose[3],
            teleop_delta_pose[4],
            1.0
        ])
        # start_time = time.time()
        # action = np.random.rand(7)
        env.step(action)
        env.render()
        # print(time.time() - start_time)
        # time.sleep(0.01)