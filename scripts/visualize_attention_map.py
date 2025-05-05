import torch
import av
import hydra
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


task = "open_box"
algo = "icon_diffusion_unet"
camera = "front_camera"
device = torch.device('cuda')

# Load ViT model
checkpoint = f"checkpoints/{task}/{algo}.pth"
with hydra.initialize_config_dir(
        config_dir=str(Path(__file__).parent.parent.joinpath("icon/configs")),
        version_base="1.2" 
    ):
        overrides = [
            f'task={task}',
            f'algo={algo}',
        ]
        cfg = hydra.compose(config_name="config", overrides=overrides)
        policy = hydra.utils.instantiate(cfg.algo.policy)
        policy.to(device)
        policy.load_state_dicts(torch.load(checkpoint, map_location=device))

model = policy.obs_encoder.image_encoder.backbones[camera]
model.eval()

# Hook to capture attention weights
attn_weights = {}

def hook_fn(module, input, output):
    # Get the input tensor (B, N, C) where N = number of tokens (patches + [CLS])
    x = input[0]
    B, N, C = x.shape  # B = batch size, N = number of tokens, C = embedding dimension
    
    # Get the query, key, and value tensors from the module (ViT self-attention)
    qkv = module.qkv(x)  # Shape: [B, N, 3 * C]
    
    # Reshape qkv into [B, N, 3, num_heads, head_dim], where head_dim = C // num_heads
    qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: [3, B, num_heads, N, head_dim]

    # Extract query, key, and value
    q, k, v = qkv[0], qkv[1], qkv[2]  # Shapes: [B, num_heads, N, head_dim]
    
    # Compute attention scores: (q @ k^T) / sqrt(head_dim)
    attn = (q @ k.transpose(-2, -1)) * module.scale  # Shape: [B, num_heads, N, N]
    attn = attn.softmax(dim=-1)  # Apply softmax across the last dimension (N tokens)

    # Save the attention map
    attn_weights['attn'] = attn.detach()



handle = model.blocks[-1].attn.register_forward_hook(hook_fn)

# Image preprocessing
transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((224, 224)),
    T.ToTensor(),
    # T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Load image
frame_id = 90
video_path = f"/home/wangjl/project/cross_embodiment/data/ee_pose/{task}/train/episode_003/videos/{camera}.mp4"
img = None
with av.open(str(video_path)) as container:
    for i, frame in enumerate(container.decode(video=0)):
        if i == frame_id:
            img = frame.to_ndarray(format='rgb24')
            break
img = Image.fromarray(img)
img = transform(img)
input_tensor = img.unsqueeze(0).to(device)  # [1, 3, 224, 224]

# Forward pass
with torch.no_grad():
    _ = model(input_tensor)

# Extract CLS→patch attention
attn = attn_weights['attn']  # shape: [1, heads, tokens, tokens]
cls_attn = attn[0, :, 0, 1:]  # [heads, num_patches]
avg_attn = cls_attn.mean(dim=0)  # [num_patches]

# Reshape to 2D grid
num_patches = avg_attn.shape[0]
side = int(num_patches ** 0.5)
attn_map = avg_attn.reshape(side, side).cpu().numpy()

# Upsample attention map to image size
attn_map_resized = np.array(Image.fromarray(attn_map).resize((224, 224), resample=Image.BILINEAR))

# Normalize attention map to [0, 1] for overlay transparency
attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min())

# Make the original image a numpy array for manipulation
img_np = np.moveaxis(np.array(img), 0, -1)

# Overlay the attention map on the image (adjust alpha for transparency)
overlay = img_np * (1 - 0.5) + plt.cm.jet(attn_map_resized)[:, :, :3] * 0.5  # 0.5 is the alpha

plt.imshow(img_np)
plt.axis('off')
plt.savefig(f"image_{task}.svg", bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.show()

plt.imshow(overlay)
plt.axis('off')
plt.savefig(f"attention_{algo}_{task}.svg", bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.show()