python -m mani_skill.examples.motionplanning.panda.run -e "PullCube-v1" \
    -n 110 \
    --obs-mode "rgb+segmentation" \
    --render-mode "sensors" \
    --record-dir "data" \
    --save-video \
    --sim-backend "auto" \
    --traj-name "motionplaning_simulation" \
    --shader "default" &
python -m mani_skill.examples.motionplanning.panda.run -e "PegInsertionSide-v1" \
    -n 110 \
    --obs-mode "rgb+segmentation" \
    --render-mode "sensors" \
    --record-dir "data" \
    --save-video \
    --sim-backend "auto" \
    --traj-name "motionplaning_simulation" \
    --shader "default" &
python -m mani_skill.examples.motionplanning.panda.run -e "LiftPegUpright-v1" \
    -n 110 \
    --obs-mode "rgb+segmentation" \
    --render-mode "sensors" \
    --record-dir "data" \
    --save-video \
    --sim-backend "auto" \
    --traj-name "motionplaning_simulation" \
    --shader "default" &
python -m mani_skill.examples.motionplanning.panda.run -e "PlaceSphere-v1" \
    -n 110 \
    --obs-mode "rgb+segmentation" \
    --render-mode "sensors" \
    --record-dir "data" \
    --save-video \
    --sim-backend "auto" \
    --traj-name "motionplaning_simulation" \
    --shader "default" 