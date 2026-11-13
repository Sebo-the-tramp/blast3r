#!/usr/bin/env bash
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

# Evaluate seed 0 of every enabled dense ETH3D offline ablation. Predictions
# must already exist and must have been saved with --export_mode full.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$HERE/../../.venv/bin/python"
ETH3D="data/eth3d"
ETH3D_SCANS="data/eth3d_scans"
EXPERIMENT_ROOT="out/eth3d/ablation"
SEED=0

# Keep this list in sync with RUNS in eth3d_offline_ablation_mean_5.sh.
RUNS=(
    no_ba
    ba_2d
    ba_fixed_depth
    ba_alpha_beta
    ba_full
    vggt_alpha_beta  # enable after changing that run to --export_mode full
    # vggt_rank1
    # vggt_rank4
    # vggt_rank8
    # vggt_rho95
    # vggt_rank15
)

for RUN in "${RUNS[@]}"; do
    RUN_OUTPUT="$EXPERIMENT_ROOT/$RUN/seed_$SEED"
    if [[ ! -d "$RUN_OUTPUT" ]]; then
        echo "Missing prediction directory: $RUN_OUTPUT" >&2
        exit 1
    fi

    echo
    echo "Evaluating $RUN (seed $SEED)"
    "$PYTHON" "$HERE/../eval_offline.py" \
        --gt_dir "$ETH3D" \
        --preds_dir "$RUN_OUTPUT" \
        --eval_pointcloud \
        --eth3d_scans_dir "$ETH3D_SCANS"
done
