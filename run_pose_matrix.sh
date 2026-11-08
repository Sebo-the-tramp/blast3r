#!/bin/bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/experiment_paths.sh"
exec "$BLAST3R_PYTHON" scripts/run_pose_matrix.py \
    --images_root "$BLAST3R_ETH3D_10" \
    --ckpt_vggt "$BLAST3R_VGGT_CHECKPOINT" \
    --ckpt_match "$BLAST3R_MATCH_CHECKPOINT" \
    --ckpt_depth "$BLAST3R_DEPTH_CHECKPOINT" \
    --cache_root "$BLAST3R_CACHE_ROOT" \
    --output "$BLAST3R_MATRIX_OUTPUT" "$@"
