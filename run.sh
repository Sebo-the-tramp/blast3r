#!/bin/bash

# MIP360
# RUN BASE BLAST3R
# python scripts/run_offline.py --images  /home/se7882ca/Documents/scsv/datasets/mip360/bicycle/images_2 --output out/mip360/bycicle/blast3r/test_00 --mono_cam

# VISUALIZE OUTPUT
# python scripts/visualize.py --scene  out/mip360/bycicle/blast3r/test_00/images_2.pth

# RUN VGGT
# python scripts/run_offline.py --images  /home/se7882ca/Documents/scsv/datasets/mip360/bicycle/images_2 --output out/mip360/bycicle/vggt/test_00 --mono_cam --depth_backend vggt --ckpt_depth /fast-storage/huggingface/hub/models--facebook--VGGT-Omega/snapshots/1041e80fc0e911235d3426b0a3d9a81075111a53/vggt_omega_1b_512.pt

# VISUALIZE OUTPUT
# python scripts/visualize.py --scene  out/mip360/bycicle/vggt/test_00/images_2.pth

# RUN VGGT + REFINED HEAD
python scripts/run_offline.py --images  /home/se7882ca/Documents/scsv/datasets/mip360/bicycle/images_2 --output out/mip360/bycicle/vggt_refined_257/test_00 --mono_cam --depth_backend vggt --ckpt_depth /fast-storage/huggingface/hub/models--facebook--VGGT-Omega/snapshots/1041e80fc0e911235d3426b0a3d9a81075111a53/vggt_omega_1b_512.pt --vggt_refine_head

# VISUALIZE OUTPUT
# python scripts/visualize.py --scene  out/mip360/bycicle/vggt_refined_257/test_00/images_2.pth


# ETH3D

# RUN BASE BLAST3R
# python scripts/run_offline.py --images /home/se7882ca/Documents/scsv/datasets/eth3d/multi_view_training_dslr_undistorted/courtyard/images/dslr_images_undistorted --output out/eth3d/courtyard/blast3r/test_00 --mono_cam

# VISUALIZE OUTPUT
# python scripts/visualize.py --scene  out/eth3d/courtyard/blast3r/test_00/dslr_images_undistorted.pth

# RUN VGGT
# python scripts/run_offline.py --images  /home/se7882ca/Documents/scsv/datasets/eth3d/multi_view_training_dslr_undistorted/courtyard/images/dslr_images_undistorted --output out/eth3d/courtyard/vggt/test_00 --mono_cam --depth_backend vggt --ckpt_depth /fast-storage/huggingface/hub/models--facebook--VGGT-Omega/snapshots/1041e80fc0e911235d3426b0a3d9a81075111a53/vggt_omega_1b_512.pt

# VISUALIZE OUTPUT
# python scripts/visualize.py --scene  out/eth3d/courtyard/vggt/test_00/dslr_images_undistorted.pth