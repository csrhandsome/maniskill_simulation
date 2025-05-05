#!/bin/bash

python scripts/eval_sim_robot.py -t lift_cube -a crossway_diffusion_unet -nt 50
python scripts/eval_sim_robot.py -t open_door -a crossway_diffusion_unet -nt 50
python scripts/eval_sim_robot.py -t stack_cube -a crossway_diffusion_unet -nt 50