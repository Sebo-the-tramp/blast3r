# 1. Prepare all scenes (symlinks images and writes reference poses):
scripts/preprocessing/prepare_eth3d.py \
  --eth3d_root /home/se7882ca/Documents/scsv/datasets/eth3d/multi_view_training_dslr_undistorted \
  --out outputs/evaluation/eth3d/gt

# Reconstruct all scenes using blast3r
scripts/run_offline.py \
  --images outputs/evaluation/eth3d/gt/*/ \
  --output outputs/evaluation/eth3d/vggt_pca15_all \
  --depth_backend blast3r \  
  --mono_cam \
  --seed 0 \
  --skip_existing

# Evaluate blast3r
.venv/bin/python scripts/eval_offline.py \
  --gt_dir outputs/evaluation/eth3d/gt \
  --preds_dir outputs/evaluation/eth3d/blast3r_pca15_all

# Reconstruct all scenes using VGGT PCA-15:

scripts/run_offline.py \
  --images outputs/evaluation/eth3d/gt/*/ \
  --output outputs/evaluation/eth3d/vggt_pca15_all \
  --depth_backend vggt \
  --ckpt_depth /fast-storage/huggingface/hub/models--facebook--VGGT-Omega/snapshots/1041e80fc0e911235d3426b0a3d9a81075111a53/vggt_omega_1b_512.pt \
  --vggt_refine_head \
  --mono_cam \    
  --seed 0 \
  --skip_existing

# Evaluate VGGT

.venv/bin/python scripts/eval_offline.py \
  --gt_dir outputs/evaluation/eth3d/gt \
  --preds_dir outputs/evaluation/eth3d/vggt_pca15_all

# For the VGGT scale-only baseline, repeat step 2 without --vggt_refine_head, changing the output to outputs/evaluation/eth3d/vggt_scale_all. Then evaluate that directory.