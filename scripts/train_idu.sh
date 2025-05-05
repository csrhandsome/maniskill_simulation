#!/bin/bash

python scripts/train.py task=pull_cube algo=icon_diffusion_unet
python scripts/train.py task=place_sphere algo=icon_diffusion_unet
python scripts/train.py task=lift_peg_upright algo=icon_diffusion_unet
python scripts/train.py task=peg_insertion_side algo=icon_diffusion_unet
python scripts/train.py task=pull_cube algo=icon_diffusion_unet train.seed=0
python scripts/train.py task=place_sphere algo=icon_diffusion_unet train.seed=0
python scripts/train.py task=lift_peg_upright algo=icon_diffusion_unet train.seed=0
python scripts/train.py task=peg_insertion_side algo=icon_diffusion_unet train.seed=0
python scripts/train.py task=pull_cube algo=icon_diffusion_unet train.seed=100
python scripts/train.py task=place_sphere algo=icon_diffusion_unet train.seed=100
python scripts/train.py task=lift_peg_upright algo=icon_diffusion_unet train.seed=100
python scripts/train.py task=peg_insertion_side algo=icon_diffusion_unet train.seed=100