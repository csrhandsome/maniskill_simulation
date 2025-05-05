#!/bin/bash

python scripts/eval_sim_robot.py -t lift_cube -a diffusion_transformer -nt 50
python scripts/eval_sim_robot.py -t open_door -a diffusion_transformer -nt 50
python scripts/eval_sim_robot.py -t pick_place_cereal -a diffusion_transformer -nt 50
python scripts/eval_sim_robot.py -t stack_cube -a diffusion_transformer -nt 50