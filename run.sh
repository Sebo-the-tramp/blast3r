#!/bin/bash

source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/experiment_paths.sh" || exit 1

# MIP360
# RUN BASE BLAST3R
# "$BLAST3R_PYTHON" scripts/run_offline.py --images  "$BLAST3R_BICYCLE_IMAGES" --output "$BLAST3R_OUTPUT_ROOT/mip360/bycicle/blast3r/test_00" --mono_cam

# VISUALIZE OUTPUT
# "$BLAST3R_PYTHON" scripts/visualize.py --scene  "$BLAST3R_OUTPUT_ROOT/mip360/bycicle/blast3r/test_00/images_2.pth"

# RUN VGGT
"$BLAST3R_PYTHON" scripts/run_offline.py --images  "$BLAST3R_BICYCLE_IMAGES" --output "$BLAST3R_OUTPUT_ROOT/mip360/bycicle/vggt/test_00" --mono_cam --depth_backend vggt --ckpt_depth "$BLAST3R_VGGT_CHECKPOINT"

# VISUALIZE OUTPUT
# "$BLAST3R_PYTHON" scripts/visualize.py --scene  "$BLAST3R_OUTPUT_ROOT/mip360/bycicle/vggt/test_00/images_2.pth"

# RUN VGGT + REFINED HEAD
# "$BLAST3R_PYTHON" scripts/run_offline.py --images  "$BLAST3R_BICYCLE_IMAGES" --output "$BLAST3R_OUTPUT_ROOT/mip360/bycicle/vggt_refined_257/test_00" --mono_cam --depth_backend vggt --ckpt_depth "$BLAST3R_VGGT_CHECKPOINT" --vggt_refine_head

# VISUALIZE OUTPUT
# "$BLAST3R_PYTHON" scripts/visualize.py --scene  "$BLAST3R_OUTPUT_ROOT/mip360/bycicle/vggt_refined_257/test_00/images_2.pth"


# ETH3D

# RUN BASE BLAST3R
# "$BLAST3R_PYTHON" scripts/run_offline.py --images "$BLAST3R_COURTYARD_IMAGES" --output "$BLAST3R_OUTPUT_ROOT/eth3d/courtyard/blast3r/test_00" --mono_cam

# VISUALIZE OUTPUT
# "$BLAST3R_PYTHON" scripts/visualize.py --scene  "$BLAST3R_OUTPUT_ROOT/eth3d/courtyard/blast3r/test_00/$(basename -- "$BLAST3R_COURTYARD_IMAGES").pth"

# RUN VGGT
# "$BLAST3R_PYTHON" scripts/run_offline.py --images  "$BLAST3R_COURTYARD_IMAGES" --output "$BLAST3R_OUTPUT_ROOT/eth3d/courtyard/vggt/test_00" --mono_cam --depth_backend vggt --ckpt_depth "$BLAST3R_VGGT_CHECKPOINT"

# VISUALIZE OUTPUT
# "$BLAST3R_PYTHON" scripts/visualize.py --scene  "$BLAST3R_OUTPUT_ROOT/eth3d/courtyard/vggt/test_00/$(basename -- "$BLAST3R_COURTYARD_IMAGES").pth"
