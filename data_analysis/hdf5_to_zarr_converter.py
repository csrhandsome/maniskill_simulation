import os
import h5py
import numpy as np
import zarr
from pathlib import Path
import glob
import tqdm


def convert_pose_representation(pose):
    """
    转换位姿表示方式,从HDF5中的8维action转换为zarr中需要的7维(6维末端执行器位姿+1维夹爪控制)
    
    Args:
        pose: 原始位姿数据,是8维表示
        
    Returns:
        转换后的7维位姿表示 (x,y,z,rx,ry,rz,gripper)
    """
    # 基于对ManiSkill的代码分析，action格式可能是:
    # 对于PDEEPoseController: 3维位置 + 3维旋转 + 1维夹爪 + 1维额外信息
    # 或其他格式，这里需要根据实际情况调整
    
    # 位置和旋转(6维) + 夹爪状态(1维)
    if len(pose) == 8:
        # 位置(xyz) + 旋转(欧拉角) + 夹爪状态 + 可能的额外信息
        pos_rot = pose[:6]  # 取前6维作为位姿
        gripper = pose[6:7]  # 取第7维作为夹爪
        return np.concatenate([pos_rot, gripper])
    else:
        # 如果已经是7维，直接返回
        return pose


def resize_image(image, target_size=(256, 256)):
    """
    调整图像大小
    
    Args:
        image: 原始图像数据
        target_size: 目标尺寸，默认为(256, 256)
        
    Returns:
        调整大小后的图像
    """
    # 示例实现，使用最简单的调整方法
    # 在实际应用中，您可能需要使用更高级的方法如OpenCV
    from skimage.transform import resize
    
    # 确保输入数据是float类型进行resize操作
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32) / 255.0
        
    # 调整大小
    resized_image = resize(image, (*target_size, 3), anti_aliasing=True)
    
    # 如果原始图像是uint8类型，则将结果转换回uint8
    if image.dtype == np.uint8:
        resized_image = (resized_image * 255).astype(np.uint8)
        
    return resized_image


def hdf5_to_zarr(h5_files, zarr_path, resize=False, target_size=(256, 256), traj_range=None):
    """
    将HDF5文件列表转换为Zarr格式，保留原始轨迹组结构
    
    Args:
        h5_files: HDF5文件路径列表
        zarr_path: 输出Zarr文件的路径
        resize: 是否调整图像大小，默认为False
        target_size: 如果resize=True，调整后的目标尺寸，默认为(256, 256)
        traj_range: 可选，指定要处理的轨迹范围，格式为(start_idx, end_idx)
        task_name: 可选，任务名称，用于创建顶层组
    """
    if not h5_files:
        print("未提供HDF5文件!")
        return
    
    print(f"开始转换 {len(h5_files)} 个HDF5文件")
    
    # 创建Zarr存储
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)
    
    # 创建一个数据组和元数据组
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    
    # 收集所有轨迹的actions和low_dims
    all_actions = []
    all_low_dims = []
    all_front_images = []
    all_wrist_images = []
    all_front_masks = []
    all_wrist_masks = []
    episode_ends = []
    current_length = 0
    
    traj_counter = 0
    
    # 处理每个HDF5文件
    for h5_file in tqdm.tqdm(h5_files, desc="处理HDF5文件"):
        print(f"处理文件: {os.path.basename(h5_file)}")
        
        with h5py.File(h5_file, 'r') as f:
            # 获取轨迹组列表
            trajectory_groups = [key for key in f.keys() if key.startswith('traj_')]
            
            # 如果指定了轨迹范围，只处理指定范围内的轨迹
            if traj_range is not None:
                start_idx, end_idx = traj_range
                trajectory_groups = trajectory_groups[start_idx:end_idx]
                
            for traj_key in tqdm.tqdm(trajectory_groups, desc=f"处理{os.path.basename(h5_file)}中的轨迹", leave=False):
                traj_group = f[traj_key]
                
                # 提取actions
                actions = traj_group['actions'][()]
                # 转换action表示方式
                converted_actions = np.array([convert_pose_representation(action) for action in actions])
                
                # 提取low_dims (关节位置、末端执行器位姿、夹爪状态)
                qpos = traj_group['obs/agent/qpos'][:-1]  # 舍弃最后一个时间步，使得数据长度与actions一致
                tcp_pose = traj_group['obs/extra/tcp_pose'][:-1]  # 舍弃最后一个时间步
                
                # 结合成低维状态向量 (7 joint positions + 6 end-effector pose + 1 gripper status)
                # Panda机器人qpos为9维：前7维是关节位置，后2维是夹爪状态(两个夹爪)
                # 我们只需要第一个夹爪状态，因为两个夹爪通常是镜像的
                low_dims = np.concatenate([
                    qpos[:, :7],  # 前7个关节位置
                    tcp_pose[:, :6],  # 末端执行器位姿 (6维)
                    qpos[:, 7:8],  # 只取第一个夹爪状态
                ], axis=1)
                
                # 提取图像数据
                front_images = traj_group['obs/sensor_data/upper_camera/rgb'][:-1]
                wrist_images = traj_group['obs/sensor_data/phone_camera/rgb'][:-1]
                
                # 提取分割掩码
                front_masks = traj_group['obs/sensor_data/upper_camera/segmentation'][:-1, :, :, 0]  # 取第一个通道
                wrist_masks = traj_group['obs/sensor_data/phone_camera/segmentation'][:-1, :, :, 0]  # 取第一个通道
                
                # 转换为二值掩码：机械臂部分(ID为1,10,12,14,16,17,18,19)为True，其他部分为False
                robot_ids = [1, 2, 10, 12, 14, 16, 17, 18, 19]  # 机械臂相关的ID
                
                # 创建与原掩码形状相同的二值掩码
                binary_front_masks = np.isin(front_masks, robot_ids).astype(np.uint8)
                binary_wrist_masks = np.isin(wrist_masks, robot_ids).astype(np.uint8)
                
                # 如果需要调整图像大小
                if resize:
                    num_frames = front_images.shape[0]
                    # 创建新的调整大小后的数组
                    resized_front_images = np.zeros((num_frames, *target_size, 3), dtype=front_images.dtype)
                    resized_wrist_images = np.zeros((num_frames, *target_size, 3), dtype=wrist_images.dtype)
                    resized_front_masks = np.zeros((num_frames, *target_size), dtype=binary_front_masks.dtype)
                    resized_wrist_masks = np.zeros((num_frames, *target_size), dtype=binary_wrist_masks.dtype)
                    
                    # 逐帧调整大小
                    for i in range(num_frames):
                        if i % 100 == 0 and num_frames > 100:
                            print(f"调整轨迹 {traj_counter} 的第 {i}/{num_frames} 帧")
                            
                        # 调整RGB图像
                        resized_front_images[i] = resize_image(front_images[i], target_size)
                        resized_wrist_images[i] = resize_image(wrist_images[i], target_size)
                        
                        # 调整掩码图像
                        from skimage.transform import resize as sk_resize
                        resized_front_masks[i] = sk_resize(binary_front_masks[i], target_size, order=0, preserve_range=True).astype(binary_front_masks.dtype)
                        resized_wrist_masks[i] = sk_resize(binary_wrist_masks[i], target_size, order=0, preserve_range=True).astype(binary_wrist_masks.dtype)
                    
                    # 使用调整大小后的图像
                    front_images = resized_front_images
                    wrist_images = resized_wrist_images
                    binary_front_masks = resized_front_masks
                    binary_wrist_masks = resized_wrist_masks
                                    
                # 添加到收集的数据中
                all_actions.append(converted_actions)
                all_low_dims.append(low_dims)
                all_front_images.append(front_images)
                all_wrist_images.append(wrist_images)
                all_front_masks.append(binary_front_masks)
                all_wrist_masks.append(binary_wrist_masks)
                
                # 记录每个轨迹的长度
                current_length += len(actions)
                episode_ends.append(current_length)
                
                traj_counter += 1
    
    if traj_counter > 0:
        # 拼接所有数据
        print("拼接所有轨迹数据...")
        all_actions = np.concatenate(all_actions, axis=0)
        all_low_dims = np.concatenate(all_low_dims, axis=0)
        all_front_images = np.concatenate(all_front_images, axis=0)
        all_wrist_images = np.concatenate(all_wrist_images, axis=0)
        all_front_masks = np.concatenate(all_front_masks, axis=0)
        all_wrist_masks = np.concatenate(all_wrist_masks, axis=0)
        
        # 保存到数据组
        data_group.create_dataset('actions', data=all_actions)
        data_group.create_dataset('low_dims', data=all_low_dims)
        data_group.create_dataset('front_camera_images', data=all_front_images)
        data_group.create_dataset('wrist_camera_images', data=all_wrist_images)
        data_group.create_dataset('front_camera_masks', data=all_front_masks)
        data_group.create_dataset('wrist_camera_masks', data=all_wrist_masks)
        
        # 保存轨迹结束点
        meta_group.create_dataset('episode_ends', data=np.array(episode_ends))
    
    print(f"转换完成，保存到: {zarr_path}")
    print(f"总共转换了 {traj_counter} 条轨迹")



if __name__ == "__main__":
    # 指定输入和输出路径
    for task in ["PullCube", "LiftPegUpright", "PegInsertionSide", "PlaceSphere"]:
        hdf5_dir = os.path.join(os.getcwd(), "data", task, "motionplanning")
        train_zarr_output = os.path.join("/home/three/桌面", "data", task, "train_data.zarr")
        val_zarr_output = os.path.join("/home/three/桌面", "data", task, "val_data.zarr")
        
        # 获取HDF5文件列表
        h5_files = sorted(glob.glob(os.path.join(hdf5_dir, "*.h5")))
        
        if not h5_files:
            print(f"在 {hdf5_dir} 中未找到HDF5文件!")
            continue
            
        # 转换训练数据并直接调整图像大小
        print(f"开始转换训练数据并调整图像大小...")
        hdf5_to_zarr(h5_files, train_zarr_output, target_size=(256, 256), traj_range=(0, 100))
        
        # 转换验证数据并直接调整图像大小
        print(f"开始转换验证数据并调整图像大小...")
        hdf5_to_zarr(h5_files, val_zarr_output, target_size=(256, 256), traj_range=(100, 110))
