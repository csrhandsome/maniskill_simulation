import tqdm
import torch
import numpy as np
import gymnasium as gym
from typing import Optional, Union, Dict
from icon.utils.gym_utils.multistep_wrapper import MultiStepWrapper
from icon.utils.gym_utils.video_recording_wrapper import VideoRecordingWrapper
from icon.utils.pytorch_utils import to


class EnvRunner:

    def __init__(
        self,
        env: gym.Env,
        obs_horizon: int,
        action_horizon: int,
        max_episode_steps: Optional[int] = 2000,
        num_trials: Optional[int] = 50,
        initial_seed: Optional[int] = 10000,
        video_save_dir: Union[str, None] = None
    ) -> None:
        # # 检查是否为ManiskillEnv类型或名为'maniskill'的环境
        # if env == 'maniskill' or 'ManiskillEnv' in str(type(env)):
        #     self.env = env
        # else:
        env = VideoRecordingWrapper(
            env=env,
            video_save_dir=video_save_dir
        )
        env = MultiStepWrapper(
            env=env,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            max_episode_steps=max_episode_steps,
            enable_temporal_ensemble=True
        )
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.num_trials = num_trials
        self.initial_seed = initial_seed
    
    def _process_obs(self, raw_obs: Dict) -> Dict:
        """
        Process observations from gym environments such that
        their formats satisfy the requirements of policies.
        """
        obs = dict()
        images = dict()
        obs['low_dims'] = torch.from_numpy(raw_obs['low_dims']).float().unsqueeze(0)
        for key, val in raw_obs.items():
            if key.endswith('images'):
                images[key.replace('_images', '')] = torch.from_numpy(val).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
        obs['images'] = images
        return obs
    
    def run(self, policy, device: torch.device) -> None:
        # Take Lid off Saucepan
        # episodes = [0, 1, 2, 4, 7, 10, 11, 12, 13, 15, 17, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 47, 48, 49, 50, 51, 52, 53, 54, 57, 58, 60, 63, 64, 66, 67, 68, 69]
        
        # Play Jenga
        # episodes = [3, 7, 9, 10, 14, 15, 20, 23, 24, 25, 26, 28, 31, 36, 40, 41, 43, 45, 47]
        
        # Put Rubbish In Bin
        # episodes = [0, 1, 2, 4, 9, 11, 12, 13, 20, 26, 28, 33, 34, 35, 36, 37, 38, 39, 40, 41, 45, 50, 52, 53, 54, 56, 57, 58, 60, 64, 65, 68, 69, 70, 71, 72, 78, 80, 84, 85, 87, 89, 90, 91, 94, 97, 100, 101, 102, 104]
        # old_episodes = [0, 1, 2, 4, 9, 11, 12, 13, 20, 26, 28, 33, 34, 35, 36, 37, 38, 39, 40, 41, 45, 50, 52, 53, 54, 56, 57, 58, 60, 64, 65, 68, 69, 70, 71, 72, 78, 80, 84, 85, 87, 89, 90, 91, 94, 97, 100, 101, 102, 104]
        # idx = [6]
        # episodes = []
        # for i in idx:
        #     episodes.append(old_episodes[i - 1])

        success = 0
        
        for t in range(self.num_trials):
        # for t in episodes:
            seed = self.initial_seed + t
            obs = self.env.reset(seed=seed)
            #print(f'obs: {obs}')
            pbar = tqdm.tqdm(
                total=self.max_episode_steps,
                desc=f"Trial {t + 1}/{self.num_trials}", 
                leave=False,
                mininterval=5.0
            )
            done = False
            steps = 0
            while not done:
                obs = self._process_obs(obs)
                to(obs, device)
                with torch.no_grad():
                    action = policy.predict_action(obs)['actions']
                action = action.detach().to('cpu').squeeze(0).numpy()
                # # only for put rubbish in bin
                # if obs['low_dims'][:, 1, -1] < 0.5:
                #     action[:, 2] *= 3
                obs, reward, done, _ = self.env.step(action)
                self.env.render()
                done = np.all(done)
                #if reward:   之前的
                if done:
                    success += 1
                    print(f"cost step: {steps}")
                if self.env.enable_temporal_ensemble:
                    pbar.update(1)
                    steps += 1
                else:
                    pbar.update(action.shape[1])
                    steps += action.shape[1]
            pbar.close()
        print(f"Success rate: {(success / self.num_trials) * 100}%")
        # Clear out video buffer
        self.env.reset()