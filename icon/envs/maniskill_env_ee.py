import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mani_skill.envs.sapien_env import BaseEnv
from scipy.spatial.transform import Rotation as R
from typing import Dict, Optional, Union, Tuple, Literal, List
import sapien
import torch

def task_to_env_name(task: str) -> str:
    if task == 'lift_peg_upright':
        return 'LiftPegUpright-v1'
    if task == 'peg_insertion_side':
        return 'PegInsertionSide-v1'
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
        obs_mode: str = "rgb+segmentation",
        control_mode: str = "pd_joint_delta_pos",# 非常重要
        render_mode: str = "rgb_array",
        reward_mode: Optional[str] = "dense",
        shader: str = "default",
        sim_backend: str = "auto",
        render_camera: Optional[str] = 'frontview',
        gpu_id: Union[int, None] = None,
        robot: str = "panda",
        render_size: int = 512,
        image_mask_keys: List[str] = None,
        enable_sapien_viewer: bool = True
    ) -> None:
        self.cameras = cameras
        self.shape_meta = shape_meta
        self.render_camera = render_camera
        self.render_mode = render_mode
        self.render_cache = None
        self.robot = robot
        self.render_size = render_size
        self.image_mask_keys = image_mask_keys or ['front_camera_masks']  # 默认使用front_camera_masks
        self.enable_sapien_viewer = enable_sapien_viewer
        
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
        
        # 确保控制模式与action_mode一致
        # 注意：ManiSkill环境接受的控制模式与我们的action_mode可能不同
        # 常见的控制模式包括：
        # - pd_joint_pos: 控制关节角度
        # - pd_joint_delta_pos: 控制关节角度增量
        # - pd_ee_pose: 控制末端姿态(xyz位置+四元数)
        # - pd_ee_delta_pose: 控制末端姿态增量
        self.control_mode = control_mode
        
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
        

        if hasattr(self.env.unwrapped, 'agent') and hasattr(self.env.unwrapped, 'control_mode'):
            print(f"控制模式: {self.env.unwrapped.control_mode}")
        
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
        """
        初始化动作空间
        使用7维动作空间(xyz位置+欧拉角+夹爪)
        """
        # 创建与配置一致的7维动作空间(xyz位置+欧拉角+夹爪)
        # 动作格式: [dx, dy, dz, drx, dry, drz, gripper]
        # dx, dy, dz: 末端执行器位置变化
        # drx, dry, drz: 末端执行器欧拉角变化 (xyz顺序)
        # gripper: 夹爪状态 (0-关闭, 1-打开)
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
        
        # 如果启用了Sapien查看器，渲染当前环境状态
        if self.enable_sapien_viewer:
            self.env.render_human()
        
        return processed_obs
    
    def _process_obs(self, obs):
        """
        处理观察数据为所需格式，模拟训练数据的格式
        参考HDF5数据的结构和EnvRunner._process_obs的格式要求
        以及MultiStepWrapper的要求
        """
        processed_obs = {}

        # 1. 处理低维度观察数据 (low_dims)
        # 创建低维状态向量
        low_dim_array = np.zeros(14, dtype=np.float32)

        try:
            # 获取关节位置（前7维）
            qpos = obs['agent']['qpos']
            qpos = qpos.cpu().numpy().flatten()    
            low_dim_array[:7] = qpos[:7]
            
            # 设置夹爪状态（第14维）
            low_dim_array[13] = qpos[7]

            # 提取TCP姿态
            tcp_pose = obs['extra']['tcp_pose']
            tcp_pose = tcp_pose.cpu().numpy().flatten()
            
            # TCP位置 (第7-9维)
            low_dim_array[7:10] = tcp_pose[:3]
            
            # TCP方向 - 将四元数转换为欧拉角 (第10-12维)
            rot = R.from_quat(tcp_pose[3:7])
            euler = rot.as_euler('xyz')
            low_dim_array[10:13] = euler
            
        except Exception as e:
            print(f"处理低维观测数据时出错: {e}")

        # 保存最终的低维数组
        processed_obs['low_dims'] = low_dim_array
        
        # 2. 处理图像数据
        # 这里进行图像处理，确保摄像机图像以正确的格式和键名存在

        for env_camera, target_camera in self.camera_mapping.items():
            # 获取RGB图像
            rgb = obs['sensor_data'][env_camera]['rgb']
            rgb = rgb.cpu().numpy()
            
            # 转换格式并确保是正确的维度 [H, W, C]

            rgb = rgb.squeeze()# 有batch维度，去掉（256, 256, 3)
            # 确保图像是uint8类型，范围0-255
            if rgb.dtype != np.uint8:
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                else:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            
            # 保存图像
            processed_obs[f'{target_camera}_images'] = rgb
            
            # 使用matplotlib显示相机拍摄的图像
            try:
                import matplotlib.pyplot as plt
                # 使用唯一的图像ID，避免多次显示同一窗口
                # plt.figure(f"{target_camera}_camera_view", figsize=(8, 8))
                # plt.clf()  # 清除当前图形
                # plt.imshow(rgb)
                # plt.title(f"{target_camera} 相机图像")
                # plt.axis('off')  # 关闭坐标轴
                # plt.draw()
                # plt.pause(0.001)  # 短暂暂停，让图像能够显示
            except Exception as e:
                print(f"显示图像时出错: {e}")
            
            # 3. 处理掩码数据
            segment = obs['sensor_data'][env_camera]['segmentation']
            
            # 确保是numpy数组
            segment = segment.cpu().numpy()
            # 确保正确的维度
            segment = segment.squeeze()
            # 创建二值掩码：机器人ID通常是特定值
            robot_objects_ids = [1,2,3,4,5,6,7,8,9,10,12,14,18,19]# 14前面是机械臂,要把16，17删了,16是桌子,17是背景的地面
            #robot_ids = [1, 2, 10, 12, 14, 16, 17, 18, 19] 
            mask = np.isin(segment, robot_objects_ids).astype(np.uint8)
            processed_obs[f'{target_camera}_masks'] = mask
            # 使用matplotlib显示掩码图像
            try:
                import matplotlib.pyplot as plt
                # plt.figure(f"{target_camera}_mask_view", figsize=(8, 8))
                # plt.clf()
                # plt.imshow(mask, cmap='gray')
                # plt.title(f"{target_camera} 掩码图像")
                # plt.axis('off')
                # plt.draw()
                # plt.pause(0.001)
            except Exception as e:
                print(f"显示掩码图像时出错: {e}")
    
        return processed_obs
    
    def step(self, action):
        """执行一步动作"""
        try:
            # 如果是多步动作序列，只取第一步
            if isinstance(action, np.ndarray) and len(action.shape) == 2 and action.shape[0] > 1:
                action = action[0]
            
            # 初始化controller_action，避免未定义错误
            controller_action = None
            
            # 检查是否有控制器并尝试使用
            has_controller = hasattr(self.env.unwrapped, 'agent') and hasattr(self.env.unwrapped.agent, 'controller')
            if has_controller:
                # 先尝试使用控制器转换动作
                try:
                    # 将动作转换为控制器需要的字典格式
                    if isinstance(action, np.ndarray) and len(action.shape) == 1 and action.shape[0] >= 7:
                        # 提取动作分量
                        translation = action[:3]
                        rotation_euler = action[3:6]
                        gripper_action = action[6]
                        
                        # 构建动作字典
                        action_dict = {
                            'arm': np.concatenate([translation, rotation_euler]),
                            'gripper': gripper_action
                        }
                        
                        # 转换为Tensor
                        import torch
                        action_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in action_dict.items()}
                        
                        # 使用控制器转换动作
                        controller_action = self.env.unwrapped.agent.controller.from_action_dict(action_dict)
                except Exception as e:
                    print(f"控制器转换出错: {e}")
                    controller_action = None
            
            # 对于pd_ee_delta_pos控制模式，需要转换为[pos(3), gripper(1)]格式
            # 使用位置部分+夹爪
            pos = controller_action[:3]
            gripper = controller_action[6:7]
            # 创建(1,4)形状的动作，忽略欧拉角部分
            formatted_action = np.concatenate([pos, gripper])
            # 添加批处理维度
            formatted_action = formatted_action.reshape(1, -1)
            # 转换为Tensor
            action = torch.tensor(formatted_action, dtype=torch.float32)
            
            # 执行环境step
            result = self.env.step(action)
            
            # 处理gymnasium格式的返回值
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
                
            # 处理观察数据
            processed_obs = self._process_obs(obs)
            processed_obs = self._ensure_obs_keys(processed_obs)
            
            # 保存当前观察
            self.obs_buffer = processed_obs
            
            # 如果启用了Sapien查看器，渲染当前环境状态
            if self.enable_sapien_viewer:
                self.env.render_human()
            
            return processed_obs, reward, done, info
            
        except Exception as e:
            print(f"环境step出错：{e}")
            import traceback
            traceback.print_exc()
            # 创建一个空观察作为fallback
            if self.obs_buffer is not None:
                return self.obs_buffer, 0.0, True, {"error": str(e)}
            else:
                # 如果没有缓存的观察，尝试创建一个基本的观察结构
                dummy_obs = self._ensure_obs_keys({})
                return dummy_obs, 0.0, True, {"error": str(e)}
    
    def render(self):
        """渲染环境"""
        # 如果启用了Sapien查看器，也渲染一下
        if self.enable_sapien_viewer:
            self.env.render_human()
            
        # 获取原始渲染结果
        frame = self.env.render()
        #print(f'frame: {frame}')
        #print(f'frame type: {type(frame)}')
        #print(f'frame shape: {frame.shape}')
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