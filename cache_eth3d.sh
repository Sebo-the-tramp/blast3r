#!/bin/bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/experiment_paths.sh"
"$BLAST3R_PYTHON" scripts/cache_eth3d.py \
    --eth3d_root "$BLAST3R_ETH3D_ROOT" \
    --out "$BLAST3R_ETH3D_10" \
    --ckpt_depth "$BLAST3R_VGGT_CHECKPOINT" \
    --cache_root "$BLAST3R_CACHE_ROOT" \
    --max_frames 10 --seed 0 --size 512 "$@"
