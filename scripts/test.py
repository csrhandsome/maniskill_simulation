# import numpy as np
# import zarr
# from pathlib import Path

# task = "play_jenga"
# source_dir = Path(f"../cross_embodiment/data/ee_delta_pose/{task}").absolute()

# for episode in list(source_dir.glob("**/*.zarr")):
#     with zarr.open(str(episode), 'r') as f:
#         img_front = f['/images/front_camera'][()]
#         img_wrist = f['/images/wrist_camera'][()]
#         if 'masks' in dict(f).keys():
#             mask = f['/masks/front_camera'][()]
#         else:
#             mask = None
#         qpos = f['/joint_properties/local/joint_positions'][()]
#         pose = f['/joint_properties/global/ee_pose'][()]
#         gripper = f['/joint_properties/global/gripper_open'][()]
#         proprio = np.concatenate([qpos, pose, gripper], axis=1)
#         actions = f['/actions'][()]
#     with zarr.open(str(episode).replace("cross_embodiment", "icon"), 'w') as f:
#         f['/images/front_camera'] = img_front
#         f['/images/wrist_camera'] = img_wrist
#         if mask is not None:
#             f['/masks/front_camera'] = mask
#         f['/proprios'] = proprio
#         f['/actions'] = actions
#     print(f"{str(episode)} is done!")



# import zarr
# import numpy as np
# from pathlib import Path

# dir = Path("data/play_jenga")
# # Train
# episode_lens = []
# front_images = []
# wrist_images = []
# front_masks = []
# low_dims = []
# actions = []
# for episode in list(dir.joinpath("train").glob("*.zarr")):
#     with zarr.open(str(episode), 'r') as f:
#         front_images.append(f['/images/front_camera'][()])
#         wrist_images.append(f['/images/wrist_camera'][()])
#         front_masks.append(f['/masks/front_camera'][()])
#         low_dims.append(f['/proprios'][()])
#         actions.append(f['/actions'][()])
#         episode_len = f['/actions'][()].shape[0]
#         episode_lens.append(episode_len)

# front_images = np.concatenate(front_images)
# wrist_images = np.concatenate(wrist_images)
# front_masks = np.concatenate(front_masks)
# low_dims = np.concatenate(low_dims)
# actions = np.concatenate(actions)    
# cumulative_episode_lens = np.cumsum(episode_lens)
# with zarr.open(str(dir).replace("data", "data_copy") + "/train_data.zarr", 'w') as f:
#     f['/data/low_dims'] = low_dims
#     f['/data/actions'] = actions
#     f['/data/front_camera_images'] = front_images
#     f['/data/wrist_camera_images'] = wrist_images
#     f['/data/front_camera_masks'] = front_masks
#     f['/meta/episode_ends'] = cumulative_episode_lens

# ### Val
# episode_lens = []
# front_images = []
# wrist_images = []
# low_dims = []
# actions = []
# for episode in list(dir.joinpath("val").glob("*.zarr")):
#     with zarr.open(str(episode), 'r') as f:
#         front_images.append(f['/images/front_camera'][()])
#         wrist_images.append(f['/images/wrist_camera'][()])
#         low_dims.append(f['/proprios'][()])
#         actions.append(f['/actions'][()])
#         episode_len = f['/actions'][()].shape[0]
#         episode_lens.append(episode_len)

# front_images = np.concatenate(front_images)
# wrist_images = np.concatenate(wrist_images)
# low_dims = np.concatenate(low_dims)
# actions = np.concatenate(actions)    
# cumulative_episode_lens = np.cumsum(episode_lens)
# with zarr.open(str(dir).replace("data", "data_copy") + "/val_data.zarr", 'w') as f:
#     f['/data/low_dims'] = low_dims
#     f['/data/actions'] = actions
#     f['/data/front_camera_images'] = front_images
#     f['/data/wrist_camera_images'] = wrist_images
#     f['/meta/episode_ends'] = cumulative_episode_lens

import math

# Define the three numbers
num1 = 11 * 2 / 100
num2 = 10 * 2 / 100
num3 = 12 * 2 / 100

# Compute the mean
mean = (num1 + num2 + num3) / 3

# Compute the variance
variance = ((num1 - mean)**2 + (num2 - mean)**2 + (num3 - mean)**2) / 3

# Compute the standard deviation
std_deviation = math.sqrt(variance)

print(f"The mean is: {mean}")
print(f"The standard deviation is: {std_deviation}")