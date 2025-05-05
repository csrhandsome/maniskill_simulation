#!/bin/bash

python scripts/train.py task=pull_cube algo=diffusion_transformer
python scripts/train.py task=place_sphere algo=diffusion_transformer
python scripts/train.py task=lift_peg_upright algo=diffusion_transformer
python scripts/train.py task=peg_insertion_side algo=diffusion_transformer
python scripts/train.py task=pull_cube algo=diffusion_transformer train.seed=0
python scripts/train.py task=place_sphere algo=diffusion_transformer train.seed=0
python scripts/train.py task=lift_peg_upright algo=diffusion_transformer train.seed=0
python scripts/train.py task=peg_insertion_side algo=diffusion_transformer train.seed=0
python scripts/train.py task=pull_cube algo=diffusion_transformer train.seed=100
python scripts/train.py task=place_sphere algo=diffusion_transformer train.seed=100
python scripts/train.py task=lift_peg_upright algo=diffusion_transformer train.seed=100
python scripts/train.py task=peg_insertion_side algo=diffusion_transformer train.seed=100