import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mani_skill.envs.sapien_env import BaseEnv
from scipy.spatial.transform import Rotation as R
from typing import Dict, Optional, Union, Tuple, Literal, List
import sapien

def task_to_env_name(task: str) -> str:
    if task == 'lift_peg_upright':
        return 'lift_peg_upright-v1'
    if task == 'peg_insertion_side':
        return 'peg_insertion_side-v1'
    elif task == 'pull_cube':
        return 'PullCube-v1'
    elif task == 'stack_cube':
        return 'StackCube-v1'
    elif task == 'place_sphere':
        return 'PlaceSphere-v1'

class ManiskillEnv():
    def __init__(
        self,
        task: str,
        cameras: List,
        shape_meta: Dict,
        obs_mode: str = "rgb",
        control_mode: str = "pd_joint_pos",
        render_mode: str = "rgb_array",
        reward_mode: Optional[str] = "dense",
        shader: str = "default",
        sim_backend: str = "auto",
        render_camera: Optional[str] = 'frontview',
        gpu_id: Union[int, None] = None,
        robot: str = "panda",
        action_mode: str = "delta_ee_pose",
        render_size: int = 512,
        image_mask_keys: List[str] = None
    ) -> None:
        self.cameras = cameras
        self.shape_meta = shape_meta
        self.render_camera = render_camera
        self.render_mode = render_mode
        self.render_cache = None
        self.robot = robot
        self.action_mode = action_mode
        self.render_size = render_size
        self.image_mask_keys = image_mask_keys or ['front_camera_masks']  # 默认使用front_camera_masks
        
        # 为了与MultiStepWrapper兼容，设置这些属性
        self.enable_temporal_ensemble = False
        self.obs_horizon = 2  # 设置为2，与配置文件中的obs_horizon一致
        self.action_horizon = 8  # 设置为8，与配置文件中的action_horizon一致
        
        # 相机名称映射 - 从环境相机到训练数据中使用的相机名称
        self.camera_mapping = {
            'upper_camera': 'front_camera',  # 上方相机映射到front_camera
            'phone_camera': 'wrist_camera'   # 机器人末端相机映射到wrist_camera
        }
        
        # 获取环境ID
        env_id = task_to_env_name(task)
        
        self.env = gym.make(
            env_id,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            reward_mode=reward_mode,
            sensor_configs=dict(shader_pack=shader),
            human_render_camera_configs=dict(shader_pack=shader),
            viewer_camera_configs=dict(shader_pack=shader),
            sim_backend=sim_backend
        )
        
        # 初始化空间
        self._init_observation_space()
        self._init_action_space()
        
        # 单个观察缓存
        self.obs_buffer = None
    
    def _init_observation_space(self):
        """初始化观察空间并确保格式一致"""
        # 创建一个符合gym标准的observation_space字典
        obs_spaces = {}
        
        # 添加低维观测空间
        if 'low_dims' in self.shape_meta:
            low_dim_shape = self.shape_meta['low_dims']
            low_dims_space = spaces.Box(
                low=-float('inf'), 
                high=float('inf'), 
                shape=(low_dim_shape,), 
                dtype=np.float32
            )
            obs_spaces['low_dims'] = low_dims_space
        
        # 添加图像观测空间
        img_size = self.shape_meta.get('images', 256)
        for camera in self.cameras:
            img_space = spaces.Box(
                low=0, 
                high=255, 
                shape=(img_size, img_size, 3), 
                dtype=np.uint8
            )
            obs_spaces[f'{camera}_images'] = img_space
            
            # 如果需要掩码，也为其添加空间
            if self.image_mask_keys and f'{camera}_masks' in self.image_mask_keys:
                mask_space = spaces.Box(
                    low=0,
                    high=1,
                    shape=(img_size, img_size),
                    dtype=np.uint8
                )
                obs_spaces[f'{camera}_masks'] = mask_space
        
        # 保存观测空间字典
        self._observation_space = spaces.Dict(obs_spaces)

    def _init_action_space(self):
        """初始化动作空间"""
        # 创建与配置一致的7维动作空间(delta_ee_pose + gripper)
        # 注意：内部ManiSkill环境期望8维动作(position + quaternion + gripper)
        # 但我们对外暴露7维动作空间(position + euler + gripper)以与其他环境保持一致
        if 'actions' in self.shape_meta:
            action_dim = self.shape_meta['actions']
            self._action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(action_dim,),
                dtype=np.float32
            )
        else:
            # 默认使用7维动作空间
            self._action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(7,),
                dtype=np.float32
            )

    def reset(self, seed=None, options=None):
        """重置环境"""
        result = self.env.reset(seed=seed, options=options)
        
        # 处理新版本gymnasium返回的(obs, info)格式
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            # 处理观察数据为所需格式
            processed_obs = self._process_obs(obs)
        else:
            # 处理观察数据为所需格式
            processed_obs = self._process_obs(result)
            info = {}
        
        # 确保所有必要的键都存在
        processed_obs = self._ensure_obs_keys(processed_obs)
        
        # 保存当前观察作为缓冲区
        self.obs_buffer = processed_obs
        
        return processed_obs
    
    def _process_obs(self, obs):
        """
        处理观察数据为所需格式，模拟训练数据的格式
        参考HDF5数据的结构和EnvRunner._process_obs的格式要求
        以及MultiStepWrapper的要求
        """
        processed_obs = {}
        
        # 1. 处理低维度观察数据 (low_dims)
        if hasattr(self, 'shape_meta') and 'low_dims' in self.shape_meta:
            # 创建低维状态向量
            low_dim_array = np.zeros(self.shape_meta['low_dims'], dtype=np.float32)
            
            try:
                # 提取关节位置和夹爪状态
                if 'agent' in obs and 'qpos' in obs['agent']:
                    qpos = obs['agent']['qpos']
                    if hasattr(qpos, 'cpu'):
                        qpos = qpos.cpu().numpy()
                    
                    # 处理qpos数据，确保是一维数组
                    qpos_flat = qpos.flatten() if hasattr(qpos, 'flatten') else qpos
                    
                    # 复制关节位置（确保不超出数组长度）
                    joint_pos_len = min(7, len(qpos_flat), len(low_dim_array))
                    low_dim_array[:joint_pos_len] = qpos_flat[:joint_pos_len]
                    
                    # 夹爪状态通常是qpos的最后部分
                    if len(qpos_flat) > 7:
                        gripper_idx = min(13, len(low_dim_array) - 1)  # 确保索引有效
                        low_dim_array[gripper_idx] = qpos_flat[7]
                
                # 提取TCP姿态
                if 'extra' in obs and 'tcp_pose' in obs['extra']:
                    tcp_pose = obs['extra']['tcp_pose']
                    if hasattr(tcp_pose, 'cpu'):
                        tcp_pose = tcp_pose.cpu().numpy()
                    
                    # 处理tcp_pose数据，确保是一维数组
                    tcp_pose_flat = tcp_pose.flatten() if hasattr(tcp_pose, 'flatten') else tcp_pose
                    
                    # TCP位置 (前3个元素)
                    if len(tcp_pose_flat) >= 3:
                        pos_idx = min(7, len(low_dim_array) - 3)
                        pos_len = min(3, len(low_dim_array) - pos_idx)
                        low_dim_array[pos_idx:pos_idx+pos_len] = tcp_pose_flat[:pos_len]
                    
                    # TCP方向 (四元数转欧拉角)
                    if len(tcp_pose_flat) >= 7:
                        rot_idx = min(10, len(low_dim_array) - 3)
                        try:
                            rot = R.from_quat(tcp_pose_flat[3:7])
                            euler = rot.as_euler('xyz')
                            rot_len = min(3, len(low_dim_array) - rot_idx)
                            low_dim_array[rot_idx:rot_idx+rot_len] = euler[:rot_len]
                        except Exception as e:
                            print(f"四元数转欧拉角出错: {e}")
            except Exception as e:
                print(f"处理低维观测数据时出错: {e}")
            
            # 保存最终的低维数组
            processed_obs['low_dims'] = low_dim_array
        
        # 2. 处理图像数据
        # 这里进行图像处理，确保摄像机图像以正确的格式和键名存在
        if 'sensor_data' in obs:
            for env_camera, target_camera in self.camera_mapping.items():
                if env_camera in obs['sensor_data'] and 'rgb' in obs['sensor_data'][env_camera]:
                    # 获取RGB图像
                    rgb = obs['sensor_data'][env_camera]['rgb']
                    
                    # 确保是numpy数组
                    if hasattr(rgb, 'cpu'):
                        rgb = rgb.cpu().numpy()
                    
                    # 转换格式并确保是正确的维度 [H, W, C]
                    if rgb.ndim > 3:  # 可能有batch维度，去掉
                        rgb = rgb.squeeze()
                    
                    # 有时图像可能是[C, H, W]格式，需要转换
                    if rgb.shape[0] == 3 and rgb.ndim == 3:
                        rgb = np.transpose(rgb, (1, 2, 0))
                    
                    # 确保图像是uint8类型，范围0-255
                    if rgb.dtype != np.uint8:
                        if rgb.max() <= 1.0:
                            rgb = (rgb * 255).astype(np.uint8)
                        else:
                            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                    
                    # 保存图像
                    processed_obs[f'{target_camera}_images'] = rgb
                    
                    # 3. 处理掩码数据（如果有）
                    if 'segmentation' in obs['sensor_data'][env_camera]:
                        seg = obs['sensor_data'][env_camera]['segmentation']
                        
                        # 确保是numpy数组
                        if hasattr(seg, 'cpu'):
                            seg = seg.cpu().numpy()
                        
                        # 确保正确的维度
                        if seg.ndim > 3:
                            seg = seg.squeeze()
                        
                        # 如果是多通道图像，取第一个通道
                        if seg.ndim == 3 and seg.shape[-1] > 1:
                            seg = seg[..., 0]
                        
                        # 创建二值掩码：机器人ID通常是特定值
                        robot_ids = [1, 2, 10, 12, 14, 16, 17, 18, 19]  # 这些可能需要根据环境调整
                        mask = np.isin(seg, robot_ids).astype(np.uint8)
                        
                        # 保存掩码
                        if f'{target_camera}_masks' in self.image_mask_keys:
                            processed_obs[f'{target_camera}_masks'] = mask
        
        return processed_obs
    
    def step(self, action):
        """执行一步动作"""
        # 检查动作形状
        if isinstance(action, np.ndarray):
            # 处理不同形状的动作
            if len(action.shape) == 2 and action.shape[0] > 1:
                # 如果是动作序列，例如(action_horizon, action_dim)，只取第一个动作
                action = action[0]  # 变为(action_dim,)
            
            # 确保action是一维数组
            action_flat = action.reshape(-1) if len(action.shape) > 1 else action
            
            # 如果是7维动作(delta_ee_pose + gripper)，处理为ManiSkill格式
            if action_flat.shape[0] == 7:
                # 分解为平移、旋转和夹爪
                translation = action_flat[:3]
                rotation_euler = action_flat[3:6]
                gripper_action = np.array([action_flat[6]])
                
                # 将欧拉角转换为四元数
                rotation_quat = R.from_euler('xyz', rotation_euler).as_quat()
                
                # 合并为ManiSkill格式的动作（8维：xyz位置、xyzw四元数、夹爪）
                maniskill_action = np.concatenate([translation, rotation_quat, gripper_action])
                
                # 添加batch维度
                maniskill_action = maniskill_action.reshape(1, -1)
                action = maniskill_action
            elif action_flat.shape[0] != 8:
                # 如果不是7维或8维，尝试调整为8维
                print(f"警告：收到的动作维度为{action_flat.shape[0]}，期望7或8维")
                # 尝试填充到8维动作
                pad_action = np.zeros(8)
                pad_action[:min(len(action_flat), 8)] = action_flat[:min(len(action_flat), 8)]
                action = pad_action.reshape(1, -1)
            else:
                # 如果已经是8维，确保有batch维度
                action = action_flat.reshape(1, -1)
        else:
            print(f"警告：收到非numpy数组类型的动作：{type(action)}")
            # 创建零动作
            action = np.zeros((1, 8))
        
        # 执行环境的step
        try:
            result = self.env.step(action)
        except Exception as e:
            print(f"环境step出错：{e}，动作形状：{action.shape if isinstance(action, np.ndarray) else '非numpy数组'}")
            # 如果出错，尝试用零动作
            action = np.zeros((1, 8))
            result = self.env.step(action)
        
        # 处理gymnasium格式的返回值
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            # 转换为老格式
            done = terminated or truncated
        elif len(result) == 4:
            obs, reward, done, info = result
        else:
            raise ValueError(f"环境返回值格式异常：{result}")
        
        # 处理观察数据
        processed_obs = self._process_obs(obs)
        
        # 确保所有必要的键都存在
        processed_obs = self._ensure_obs_keys(processed_obs)
        
        # 保存当前观察
        self.obs_buffer = processed_obs
        
        return processed_obs, reward, done, info
    
    def render(self):
        """渲染环境"""
        # 获取原始渲染结果
        frame = self.env.render()
        
        # 确保帧数据是numpy数组，并且类型是uint8
        if frame is not None:
            # 如果是PyTorch张量，转换为numpy
            if hasattr(frame, 'cpu'):
                frame = frame.cpu().numpy()
            
            # 确保数据类型是uint8
            if frame.dtype != np.uint8:
                # 如果是浮点数并且范围在[0, 1]，乘以255并转换为uint8
                if frame.dtype in [np.float32, np.float64] and frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                # 如果是其他类型或范围，确保在0-255范围内
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            # 检查图像维度并确保是HWC格式
            if frame.ndim == 2:  # 如果是灰度图，转为RGB
                frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
            elif frame.ndim > 3:  # 如果有额外维度，去掉
                frame = frame.squeeze()
                # 确保挤压后仍有正确的维度
                if frame.ndim == 2:
                    frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
            
            # 如果维度顺序是通道优先(CHW)，转为HWC
            if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[2] != 3:
                frame = np.transpose(frame, (1, 2, 0))
            
            # 确保最终图像是3通道的
            if frame.ndim == 3 and frame.shape[2] not in [1, 2, 3, 4]:
                # 如果通道数异常，取前3个通道或者重复到3个通道
                if frame.shape[2] > 3:
                    frame = frame[:, :, :3]
                else:
                    # 如果通道数不足，这很奇怪，创建一个合适大小的空图像
                    h, w = frame.shape[:2]
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            # 如果没有渲染结果，创建一个空图像
            frame = np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)
        
        # 保存渲染缓存
        self.render_cache = frame
        
        return frame
    
    def close(self):
        """关闭环境"""
        return self.env.close()
    
    @property
    def observation_space(self):
        """获取观察空间"""
        if hasattr(self, '_observation_space'):
            return self._observation_space
        else:
            return self.env.observation_space
    
    @property
    def action_space(self):
        """获取动作空间"""
        if hasattr(self, '_action_space'):
            return self._action_space
        else:
            self._init_action_space()
            return self._action_space

    def _ensure_obs_keys(self, obs):
        """确保观测包含所有必要的键"""
        if obs is None:
            # 如果观测为空，创建完整的默认观测
            obs = {}
        
        # 创建目标键集
        target_keys = set(['low_dims'])
        
        # 添加相机相关的键
        for camera in self.cameras:
            target_keys.add(f'{camera}_images')
            if self.image_mask_keys and f'{camera}_masks' in self.image_mask_keys:
                target_keys.add(f'{camera}_masks')
        
        # 检查并添加缺失的键
        for key in target_keys:
            if key not in obs:
                if key == 'low_dims':
                    # 添加默认的低维观测
                    obs[key] = np.zeros(self.shape_meta.get('low_dims', 14), dtype=np.float32)
                elif key.endswith('_images'):
                    # 添加默认的图像观测
                    img_size = self.shape_meta.get('images', 256)
                    obs[key] = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                elif key.endswith('_masks'):
                    # 添加默认的掩码观测
                    img_size = self.shape_meta.get('images', 256)
                    obs[key] = np.zeros((img_size, img_size), dtype=np.uint8)
            else:
                # 确保已存在的键具有正确的数据类型和形状
                if key == 'low_dims':
                    # 检查低维观测的形状
                    if isinstance(obs[key], np.ndarray):
                        if obs[key].shape != (self.shape_meta.get('low_dims', 14),):
                            # 重新塑形或创建正确大小的数组
                            try:
                                old_data = obs[key].flatten()
                                new_data = np.zeros(self.shape_meta.get('low_dims', 14), dtype=np.float32)
                                new_data[:min(len(old_data), len(new_data))] = old_data[:min(len(old_data), len(new_data))]
                                obs[key] = new_data
                            except:
                                obs[key] = np.zeros(self.shape_meta.get('low_dims', 14), dtype=np.float32)
                    else:
                        # 如果不是数组，创建一个新的
                        obs[key] = np.zeros(self.shape_meta.get('low_dims', 14), dtype=np.float32)
                elif key.endswith('_images'):
                    # 检查图像的形状和类型
                    img_size = self.shape_meta.get('images', 256)
                    if isinstance(obs[key], np.ndarray):
                        if obs[key].shape != (img_size, img_size, 3) or obs[key].dtype != np.uint8:
                            # 尝试调整图像大小或创建新图像
                            try:
                                # 确保是uint8类型
                                if obs[key].dtype != np.uint8:
                                    if obs[key].max() <= 1.0:
                                        obs[key] = (obs[key] * 255).astype(np.uint8)
                                    else:
                                        obs[key] = np.clip(obs[key], 0, 255).astype(np.uint8)
                                
                                # 如果形状不对，创建新数组
                                if obs[key].shape != (img_size, img_size, 3):
                                    obs[key] = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                            except:
                                obs[key] = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                    else:
                        obs[key] = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                elif key.endswith('_masks'):
                    # 检查掩码的形状和类型
                    img_size = self.shape_meta.get('images', 256)
                    if isinstance(obs[key], np.ndarray):
                        if obs[key].shape != (img_size, img_size) or obs[key].dtype != np.uint8:
                            # 尝试调整掩码大小或创建新掩码
                            try:
                                # 确保是uint8类型
                                if obs[key].dtype != np.uint8:
                                    obs[key] = obs[key].astype(np.uint8)
                                
                                # 如果形状不对，创建新数组
                                if obs[key].shape != (img_size, img_size):
                                    obs[key] = np.zeros((img_size, img_size), dtype=np.uint8)
                            except:
                                obs[key] = np.zeros((img_size, img_size), dtype=np.uint8)
                    else:
                        obs[key] = np.zeros((img_size, img_size), dtype=np.uint8)
        
        return obs