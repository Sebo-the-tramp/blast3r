#!/bin/bash

## PREPARATION ##

# scripts/preprocessing/prepare_eth3d.py \
#   --eth3d_root /home/se7882ca/Documents/scsv/datasets/eth3d/multi_view_training_dslr_undistorted \
#   --out outputs/evaluation/eth3d/gt\


## INFERENCE ##

# MAX_IMAGES=-1 # -1 for all images, 10 for testing
MAX_IMAGES=10 # -1 for all images, 10 for testing

# # Reconstruct all scenes using blast3r NO BA
# scripts/run_offline.py \
#   --images outputs/evaluation/eth3d/gt/*/ \
#   --output outputs/evaluation/eth3d/blast3r_no_BA \
#   --depth_backend blast3r \
#   --max_iters 0 \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing \

# # Reconstruct all scenes using blast3r with 1 BA iteration
# scripts/run_offline.py \
#   --images outputs/evaluation/eth3d/gt/*/ \
#   --output outputs/evaluation/eth3d/blast3r_with_BA_1 \
#   --depth_backend blast3r \
#   --max_iters 1 \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing \

# Reconstruct all scenes using blast3r but not optimizing DEPTH
# scripts/run_offline.py \
#   --images outputs/evaluation/eth3d/gt/*/ \
#   --output outputs/evaluation/eth3d/blast3r_no_Z_optim \
#   --depth_backend blast3r \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing \
#   --weight_z 0

# # Reconstruct all scenes using blast3r
# scripts/run_offline.py \
#   --images outputs/evaluation/eth3d/gt/*/ \
#   --output outputs/evaluation/eth3d/blast3r_nz1_${MAX_IMAGES}_max_imgs \
#   --depth_backend blast3r \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing \
#   -nz 1 \
#   --max_frames $MAX_IMAGES

# # Reconstruct all scenes using blast3r 15 depths
# scripts/run_offline.py \
#   --images outputs/evaluation/eth3d/gt/*/ \
#   --output outputs/evaluation/eth3d/blast3r_nz16 \
#   --depth_backend blast3r \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing \
#   -nz 1

# # Reconstruct all scenes using VGGT no refinement:
# scripts/run_offline.py \
#   --images outputs/evaluation/eth3d/gt/*/ \
#   --output outputs/evaluation/eth3d/vggt_no_refinement_${MAX_IMAGES}_max_imgs \
#   --depth_backend vggt \
#   --ckpt_depth /fast-storage/huggingface/hub/models--facebook--VGGT-Omega/snapshots/1041e80fc0e911235d3426b0a3d9a81075111a53/vggt_omega_1b_512.pt \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing \
#   --max_frames $MAX_IMAGES

# # Reconstruct all scenes using VGGT PCA-15:
# scripts/run_offline.py \
#   --images outputs/evaluation/eth3d/gt/*/ \
#   --output outputs/evaluation/eth3d/vggt_pca15_all \
#   --depth_backend vggt \
#   --ckpt_depth /fast-storage/huggingface/hub/models--facebook--VGGT-Omega/snapshots/1041e80fc0e911235d3426b0a3d9a81075111a53/vggt_omega_1b_512.pt \
#   --vggt_refine_head \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing

# # Reconstruct all scenes using VGGT PCA-15:
# scripts/run_offline.py \
#   --images outputs/evaluation/eth3d/gt/*/ \
#   --output outputs/evaluation/eth3d/vggt_pca15_all_${MAX_IMAGES}_max_imgs \
#   --depth_backend vggt \
#   --ckpt_depth /fast-storage/huggingface/hub/models--facebook--VGGT-Omega/snapshots/1041e80fc0e911235d3426b0a3d9a81075111a53/vggt_omega_1b_512.pt \
#   --vggt_refine_head \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing \
#   --max_frames $MAX_IMAGES

## EVAL ##

# # Evaluate blast3r no BA (row 5 of Table 22)
# echo "Evaluating blast3r no BA"
# .venv/bin/python scripts/eval_offline.py \
#   --gt_dir outputs/evaluation/eth3d/gt \
#   --preds_dir outputs/evaluation/eth3d/blast3r_no_BA
# echo "-----------------------------------------"

# # Evaluate blast3r no BA (row 5 of Table 22)
# echo "Evaluating blast3r no BA"
# .venv/bin/python scripts/eval_offline.py \
#   --gt_dir outputs/evaluation/eth3d/gt \
#   --preds_dir outputs/evaluation/eth3d/blast3r_with_BA_1
# echo "-----------------------------------------"

# # Evaluate blast3r no Z optimization (row 6 of Table 22)
# echo "Evaluating blast3r no Z optimization"
# .venv/bin/python scripts/eval_offline.py \
#   --gt_dir outputs/evaluation/eth3d/gt \
#   --preds_dir outputs/evaluation/eth3d/blast3r_no_Z_optim
# echo "-----------------------------------------"

# # Evaluate blast3r nz1
# echo "Evaluating blast3r nz1"
# .venv/bin/python scripts/eval_offline.py \
#   --gt_dir outputs/evaluation/eth3d/gt \
#   --preds_dir outputs/evaluation/eth3d/blast3r_nz1
# echo "-----------------------------------------"


# Evaluate blast3r nz1 with MAX_IMAGES=10 (row 7 of Table 22)
echo "Evaluating blast3r nz1 with MAX_IMAGES=10"
.venv/bin/python scripts/eval_offline.py \
  --gt_dir outputs/evaluation/eth3d/gt \
  --preds_dir outputs/evaluation/eth3d/blast3r_nz1_${MAX_IMAGES}_max_imgs \
  --allow_subset
echo "-----------------------------------------"

# # Evaluate blast3r nz16
# echo "Evaluating blast3r nz16"
# .venv/bin/python scripts/eval_offline.py \
#   --gt_dir outputs/evaluation/eth3d/gt \
#   --preds_dir outputs/evaluation/eth3d/blast3r_nz16
# echo "-----------------------------------------"

# # Evaluate VGGT no-refinement
# echo "Evaluating VGGT no-refinement"
# .venv/bin/python scripts/eval_offline.py \
#   --gt_dir outputs/evaluation/eth3d/gt \
#   --preds_dir outputs/evaluation/eth3d/vggt_no_refinement
# echo "-----------------------------------------"

# Evaluate VGGT PCA_15 MAX_IMAGES=10
echo "Evaluating VGGT no-refinement with MAX_IMAGES=10"
.venv/bin/python scripts/eval_offline.py \
  --gt_dir outputs/evaluation/eth3d/gt \
  --preds_dir outputs/evaluation/eth3d/vggt_no_refinement_${MAX_IMAGES}_max_imgs \
  --allow_subset
echo "-----------------------------------------"

# # Evaluate VGGT PCA_15
# echo "Evaluating VGGT PCA_15"
# .venv/bin/python scripts/eval_offline.py \
#   --gt_dir outputs/evaluation/eth3d/gt \
#   --preds_dir outputs/evaluation/eth3d/vggt_pca15_all
# echo "-----------------------------------------"

# Evaluate VGGT PCA_15 MAX_IMAGES=10
echo "Evaluating VGGT PCA_15 with MAX_IMAGES=10"
.venv/bin/python scripts/eval_offline.py \
  --gt_dir outputs/evaluation/eth3d/gt \
  --preds_dir outputs/evaluation/eth3d/vggt_pca15_all_${MAX_IMAGES}_max_imgs \
  --allow_subset
echo "-----------------------------------------"
