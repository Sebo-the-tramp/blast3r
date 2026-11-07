#!/bin/bash

source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/experiment_paths.sh" || exit 1

## PREPARATION ##

# "$BLAST3R_PYTHON" scripts/preprocessing/prepare_eth3d.py \
#   --eth3d_root "$BLAST3R_ETH3D_ROOT" \
#   --out "$BLAST3R_ETH3D_GT"\


## INFERENCE ##

# MAX_IMAGES=-1 # -1 for all images, 10 for testing
MAX_IMAGES=10 # -1 for all images, 10 for testing

# # Reconstruct all scenes using blast3r NO BA
# "$BLAST3R_PYTHON" scripts/run_offline.py \
#   --images "$BLAST3R_ETH3D_GT"/*/ \
#   --output "$BLAST3R_EVAL_ROOT/blast3r_no_BA" \
#   --depth_backend blast3r \
#   --max_iters 0 \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing \

# # Reconstruct all scenes using blast3r with 1 BA iteration
# "$BLAST3R_PYTHON" scripts/run_offline.py \
#   --images "$BLAST3R_ETH3D_GT"/*/ \
#   --output "$BLAST3R_EVAL_ROOT/blast3r_with_BA_1" \
#   --depth_backend blast3r \
#   --max_iters 1 \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing \

# Reconstruct all scenes using blast3r but not optimizing DEPTH
# "$BLAST3R_PYTHON" scripts/run_offline.py \
#   --images "$BLAST3R_ETH3D_GT"/*/ \
#   --output "$BLAST3R_EVAL_ROOT/blast3r_no_Z_optim" \
#   --depth_backend blast3r \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing \
#   --weight_z 0

# # Reconstruct all scenes using blast3r
# "$BLAST3R_PYTHON" scripts/run_offline.py \
#   --images "$BLAST3R_ETH3D_GT"/*/ \
#   --output "$BLAST3R_EVAL_ROOT/blast3r_nz1_${MAX_IMAGES}_max_imgs" \
#   --depth_backend blast3r \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing \
#   -nz 1 \
#   --max_frames $MAX_IMAGES

# # Reconstruct all scenes using blast3r 15 depths
# "$BLAST3R_PYTHON" scripts/run_offline.py \
#   --images "$BLAST3R_ETH3D_GT"/*/ \
#   --output "$BLAST3R_EVAL_ROOT/blast3r_nz16" \
#   --depth_backend blast3r \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing \
#   -nz 1

# # Reconstruct all scenes using VGGT no refinement:
# "$BLAST3R_PYTHON" scripts/run_offline.py \
#   --images "$BLAST3R_ETH3D_GT"/*/ \
#   --output "$BLAST3R_EVAL_ROOT/vggt_no_refinement_${MAX_IMAGES}_max_imgs" \
#   --depth_backend vggt \
#   --ckpt_depth "$BLAST3R_VGGT_CHECKPOINT" \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing \
#   --max_frames $MAX_IMAGES

# # Reconstruct all scenes using VGGT PCA-15:
# "$BLAST3R_PYTHON" scripts/run_offline.py \
#   --images "$BLAST3R_ETH3D_GT"/*/ \
#   --output "$BLAST3R_EVAL_ROOT/vggt_pca15_all" \
#   --depth_backend vggt \
#   --ckpt_depth "$BLAST3R_VGGT_CHECKPOINT" \
#   --vggt_refine_head \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing

# # Reconstruct all scenes using VGGT PCA-15:
# "$BLAST3R_PYTHON" scripts/run_offline.py \
#   --images "$BLAST3R_ETH3D_GT"/*/ \
#   --output "$BLAST3R_EVAL_ROOT/vggt_pca15_all_${MAX_IMAGES}_max_imgs" \
#   --depth_backend vggt \
#   --ckpt_depth "$BLAST3R_VGGT_CHECKPOINT" \
#   --vggt_refine_head \
#   --mono_cam \
#   --seed 0 \
#   --skip_existing \
#   --max_frames $MAX_IMAGES

## EVAL ##

# # Evaluate blast3r no BA (row 5 of Table 22)
# echo "Evaluating blast3r no BA"
# "$BLAST3R_PYTHON" scripts/eval_offline.py \
#   --gt_dir "$BLAST3R_ETH3D_GT" \
#   --preds_dir "$BLAST3R_EVAL_ROOT/blast3r_no_BA"
# echo "-----------------------------------------"

# # Evaluate blast3r no BA (row 5 of Table 22)
# echo "Evaluating blast3r no BA"
# "$BLAST3R_PYTHON" scripts/eval_offline.py \
#   --gt_dir "$BLAST3R_ETH3D_GT" \
#   --preds_dir "$BLAST3R_EVAL_ROOT/blast3r_with_BA_1"
# echo "-----------------------------------------"

# # Evaluate blast3r no Z optimization (row 6 of Table 22)
# echo "Evaluating blast3r no Z optimization"
# "$BLAST3R_PYTHON" scripts/eval_offline.py \
#   --gt_dir "$BLAST3R_ETH3D_GT" \
#   --preds_dir "$BLAST3R_EVAL_ROOT/blast3r_no_Z_optim"
# echo "-----------------------------------------"

# # Evaluate blast3r nz1
# echo "Evaluating blast3r nz1"
# "$BLAST3R_PYTHON" scripts/eval_offline.py \
#   --gt_dir "$BLAST3R_ETH3D_GT" \
#   --preds_dir "$BLAST3R_EVAL_ROOT/blast3r_nz1"
# echo "-----------------------------------------"


# Evaluate blast3r nz1 with MAX_IMAGES=10 (row 7 of Table 22)
echo "Evaluating blast3r nz1 with MAX_IMAGES=10"
"$BLAST3R_PYTHON" scripts/eval_offline.py \
  --gt_dir "$BLAST3R_ETH3D_GT" \
  --preds_dir "$BLAST3R_EVAL_ROOT/blast3r_nz1_${MAX_IMAGES}_max_imgs" \
  --allow_subset
echo "-----------------------------------------"

# # Evaluate blast3r nz16
# echo "Evaluating blast3r nz16"
# "$BLAST3R_PYTHON" scripts/eval_offline.py \
#   --gt_dir "$BLAST3R_ETH3D_GT" \
#   --preds_dir "$BLAST3R_EVAL_ROOT/blast3r_nz16"
# echo "-----------------------------------------"

# # Evaluate VGGT no-refinement
# echo "Evaluating VGGT no-refinement"
# "$BLAST3R_PYTHON" scripts/eval_offline.py \
#   --gt_dir "$BLAST3R_ETH3D_GT" \
#   --preds_dir "$BLAST3R_EVAL_ROOT/vggt_no_refinement"
# echo "-----------------------------------------"

# Evaluate VGGT PCA_15 MAX_IMAGES=10
echo "Evaluating VGGT no-refinement with MAX_IMAGES=10"
"$BLAST3R_PYTHON" scripts/eval_offline.py \
  --gt_dir "$BLAST3R_ETH3D_GT" \
  --preds_dir "$BLAST3R_EVAL_ROOT/vggt_no_refinement_${MAX_IMAGES}_max_imgs" \
  --allow_subset
echo "-----------------------------------------"

# # Evaluate VGGT PCA_15
# echo "Evaluating VGGT PCA_15"
# "$BLAST3R_PYTHON" scripts/eval_offline.py \
#   --gt_dir "$BLAST3R_ETH3D_GT" \
#   --preds_dir "$BLAST3R_EVAL_ROOT/vggt_pca15_all"
# echo "-----------------------------------------"

# Evaluate VGGT PCA_15 MAX_IMAGES=10
echo "Evaluating VGGT PCA_15 with MAX_IMAGES=10"
"$BLAST3R_PYTHON" scripts/eval_offline.py \
  --gt_dir "$BLAST3R_ETH3D_GT" \
  --preds_dir "$BLAST3R_EVAL_ROOT/vggt_pca15_all_${MAX_IMAGES}_max_imgs" \
  --allow_subset
echo "-----------------------------------------"
