import imageio
import numpy as np
import gymnasium as gym
from typing import Tuple, Union
from icon.utils.file_utils import str2path, mkdir


class VideoRecordingWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        video_save_dir: Union[str, None] = None
    ) -> None:
        super().__init__(env)
        self.global_step = 0
        if video_save_dir is None:
            self.record_videos = False
        else:
            self.record_videos = True
            self.video_save_dir = mkdir(str2path(video_save_dir), parents=True, exist_ok=True)
            self.frames = list()
            
    def reset(self, **kwargs) -> None:
        obs = super().reset(**kwargs)
        self.global_step += 1
        if self.record_videos:
            self.frames = list()
            self.video_recoder = imageio.get_writer(self.video_save_dir.joinpath(f"trial_{str(self.global_step).zfill(3)}.mp4"), fps=24)
        return obs
    
    def step(self, action: np.ndarray) -> Tuple:
        outputs = super().step(action)
        if self.record_videos:
            frame = self.env.render()
            assert frame.dtype == np.uint8
            self.video_recoder.append_data(frame)
        return outputs
    
    def close(self) -> None:
        super().close()