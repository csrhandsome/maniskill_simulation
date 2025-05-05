# Intra Contrast (ICon)
This is the official PyTorch implementation of *Intra Contrast*.

## 🔧 Installation
First create a conda environment and install all dependencies.
```bash
conda create -n icon_env python=3.9
conda activate icon_env
pip install -e .
```
If you meet the following error:
```
libGL error: MESA-LOADER: failed to open swrast: /lib/x86_64-linux-gnu/libLLVM-12.so.1: undefined symbol: ffi_type_sint32, version LIBFFI_BASE_7.0
```
Downgrade `libffi` version ([reference](https://github.com/ContinuumIO/anaconda-issues/issues/13205)):
```bash
conda install -c conda-forge libffi=3.4.2
```

### 💻 Simulation
We tested our algorithm across 5 tasks from 1 simulation benchmarks. To reproduce our simulation results, you need to install every simulation framework following the instructions in the original codebase.  
- [RLBench](https://github.com/stepjam/RLBench?tab=readme-ov-file#install)

### 🤖 Real Robot
We haven't conducted any real-world robot experiments. 

## 📋 Usage Instructions
### 🎮 Collecting Data in Simulation
Human demonstrations for imitation learning are usually collected via teleopration. Most of the simulation benchmarks utilized in this project provide users with tools to collect demonstrations in simulation, which can be found in their documentation or project websites. For RLBench, just run the following command:
```bash
python scripts/collect_episodes.py --save_dir ~/data/close_door --train_samples 50 --val_samples 10
```
This will create 50 training samples and 10 validation samples under directory `~/data/close_door`.
### 🎬 Generating Segmentation Masks for Videos
We use an off-the-shelf image (also video) segmentation model, [Segment Anything 2 (SAM2)](https://arxiv.org/abs/2408.00714), to generate segmentation masks for robots in the scenes. We recommend users to use the [online tool](https://ai.meta.com/sam2/) provided by Meta, as it is more convenient for users to adjust segmentation masks in real time. Remember not to change the names of MP4 files and to place videos of masks under directory `episode_???/masks`. If you want to remove the watermarks on videos, run:
```bash
python scripts/postprocess_masks.py -ed ~/data/close_door
```

### ⏳ Running for Epochs
Now it's time to have a try! Run the following command to start training a new policy with seed 1 on GPU:
```bash
python scripts/train.py --config-name=clear.yaml task=close_door seed=1 device=cuda dataset_dir='data/close_door'
```
This will load configuration from `cross_embodiment/configs/workspaces/clear.yaml` and create a directory `outputs/$WORKSPACE_NAME/$TASK_NAME/YYYY-MM-DD/HH-MM-SS` where configuration files, logging files, and checkpoints are written to. For more details of model and training configuration, find them under `cross_embodiment/configs/workspaces`.

### 📐 Evaluating Pre-trained Checkpoints in Simulation
You can evaluate task success rate by running the following command: 
```bash
python scripts/eval.py -w clear -e rlbench -c @CHECKPOINT_PATH -s 100
```
This will rollout the pre-trained policy in the RLBench environment. If the robot successfully completes the task, or the running iteration exceeds the maximum rollout steps (100 in this situation), the program will automatically terminate. 