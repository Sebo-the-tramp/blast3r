#!/usr/bin/env bash
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

# Compare depth-based initialization, classical 2D BA, and the full two-stage
# BLASt3R schedule over ETH3D retrieval seeds. All methods reuse the same
# cached depth predictions, tracks and initial poses for a given seed.
set -euo pipefail

ETH3D="data/eth3d_small"
EXPERIMENT_ROOT="out/eth3d/ablation"
NO_BA_ROOT="$EXPERIMENT_ROOT/no_ba"
BA_2D_ROOT="$EXPERIMENT_ROOT/ba_2d"
BA_FIXED_DEPTH_ROOT="$EXPERIMENT_ROOT/ba_fixed_depth"
BA_ALPHA_BETA_ROOT="$EXPERIMENT_ROOT/ba_alpha_beta"
BA_FULL_ROOT="$EXPERIMENT_ROOT/ba_full"

CKPT_MATCH="naver/blast3r-matcher"
CKPT_DEPTH="naver/blast3r-depth"

# Enable or disable individual ablations without editing the loop below.
RUN_NO_BA=true
RUN_BA_2D=true
RUN_BA_FIXED_DEPTH=true
RUN_BA_ALPHA_BETA=true
RUN_BA_FULL=true

for SEED in 0 1 2 3 4; do
    if [[ "$RUN_NO_BA" == true ]]; then
        # Depth-based spanning-tree initialization, with no BA updates.
        RUN_OUTPUT="$NO_BA_ROOT/seed_$SEED"
        python "$(dirname "$0")/../run_offline.py" \
            --images "$ETH3D"/*/ \
            --output "$RUN_OUTPUT" \
            --ckpt_match "$CKPT_MATCH" \
            --ckpt_depth "$CKPT_DEPTH" \
            --size 512 \
            --mono_cam \
            --max_iters 0 \
            --seed "$SEED" \
            --export_mode pose \
            --use_cache

        python "$(dirname "$0")/../eval_offline.py" \
            --gt_dir "$ETH3D" \
            --preds_dir "$RUN_OUTPUT"
    fi

    if [[ "$RUN_BA_2D" == true ]]; then
        # Classical sparse BA: K, poses and sparse points use 2D residuals only.
        RUN_OUTPUT="$BA_2D_ROOT/seed_$SEED"
        python "$(dirname "$0")/../run_offline.py" \
            --images "$ETH3D"/*/ \
            --output "$RUN_OUTPUT" \
            --ckpt_match "$CKPT_MATCH" \
            --ckpt_depth "$CKPT_DEPTH" \
            --size 512 \
            --max_iters 100 \
            --weight_z 0 \
            --n_zcfs 0 \
            --mono_cam \
            --optim_K True \
            --optim_P True \
            --optim_Z False \
            --seed "$SEED" \
            --export_mode pose \
            --use_cache

        python "$(dirname "$0")/../eval_offline.py" \
            --gt_dir "$ETH3D" \
            --preds_dir "$RUN_OUTPUT"
    fi

    if [[ "$RUN_BA_FIXED_DEPTH" == true ]]; then
        # Depth-regularized BA with the initialized alpha/beta held fixed.
        RUN_OUTPUT="$BA_FIXED_DEPTH_ROOT/seed_$SEED"
        python "$(dirname "$0")/../run_offline.py" \
            --images "$ETH3D"/*/ \
            --output "$RUN_OUTPUT" \
            --ckpt_match "$CKPT_MATCH" \
            --ckpt_depth "$CKPT_DEPTH" \
            --size 512 \
            --max_iters 100 \
            --weight_z 1e-4 \
            --n_zcfs 1 \
            --mono_cam \
            --optim_K True \
            --optim_P True \
            --optim_Z False \
            --seed "$SEED" \
            --export_mode pose \
            --use_cache

        python "$(dirname "$0")/../eval_offline.py" \
            --gt_dir "$ETH3D" \
            --preds_dir "$RUN_OUTPUT"
    fi

    if [[ "$RUN_BA_ALPHA_BETA" == true ]]; then
        # Depth-regularized BA with only alpha and beta optimized.
        RUN_OUTPUT="$BA_ALPHA_BETA_ROOT/seed_$SEED"
        python "$(dirname "$0")/../run_offline.py" \
            --images "$ETH3D"/*/ \
            --output "$RUN_OUTPUT" \
            --ckpt_match "$CKPT_MATCH" \
            --ckpt_depth "$CKPT_DEPTH" \
            --size 512 \
            --max_iters 100 \
            --weight_z 1e-4 \
            --n_zcfs 2 \
            --mono_cam \
            --optim_K True \
            --optim_P True \
            --optim_Z True \
            --seed "$SEED" \
            --export_mode pose \
            --use_cache

        python "$(dirname "$0")/../eval_offline.py" \
            --gt_dir "$ETH3D" \
            --preds_dir "$RUN_OUTPUT"
    fi

    if [[ "$RUN_BA_FULL" == true ]]; then
        # Full schedule: recover cameras with a weak depth term, then fix cameras
        # and stitch all depth components using a stronger depth term.
        RUN_OUTPUT="$BA_FULL_ROOT/seed_$SEED"
        python "$(dirname "$0")/../run_offline.py" \
            --images "$ETH3D"/*/ \
            --output "$RUN_OUTPUT" \
            --ckpt_match "$CKPT_MATCH" \
            --ckpt_depth "$CKPT_DEPTH" \
            --size 512 \
            --max_iters 100 \
            --weight_z 1e-4 \
            --n_zcfs 17 \
            --mono_cam \
            --optim_K True \
            --optim_P True \
            --optim_Z True \
            --seed "$SEED" \
            --export_mode pose \
            --use_cache

        python "$(dirname "$0")/../eval_offline.py" \
            --gt_dir "$ETH3D" \
            --preds_dir "$RUN_OUTPUT"
    fi
done

if [[ "$RUN_NO_BA" == true ]]; then
    python "$(dirname "$0")/../aggregate_offline_runs.py" \
        --runs_dir "$NO_BA_ROOT" \
        --expected_runs 0
fi

if [[ "$RUN_BA_2D" == true ]]; then
    python "$(dirname "$0")/../aggregate_offline_runs.py" \
        --runs_dir "$BA_2D_ROOT" \
        --expected_runs 0
fi

if [[ "$RUN_BA_FIXED_DEPTH" == true ]]; then
    python "$(dirname "$0")/../aggregate_offline_runs.py" \
        --runs_dir "$BA_FIXED_DEPTH_ROOT" \
        --expected_runs 0
fi

if [[ "$RUN_BA_ALPHA_BETA" == true ]]; then
    python "$(dirname "$0")/../aggregate_offline_runs.py" \
        --runs_dir "$BA_ALPHA_BETA_ROOT" \
        --expected_runs 0
fi

if [[ "$RUN_BA_FULL" == true ]]; then
    python "$(dirname "$0")/../aggregate_offline_runs.py" \
        --runs_dir "$BA_FULL_ROOT" \
        --expected_runs 0
fi

# Render one LaTeX row per configuration in the manifest, including mean time.
# This requires the four corresponding RUN_* switches above to remain enabled.
if [[ "$RUN_BA_2D" == true && "$RUN_BA_FIXED_DEPTH" == true && \
      "$RUN_BA_ALPHA_BETA" == true && "$RUN_BA_FULL" == true ]]; then
    python "$(dirname "$0")/../aggregate_offline_runs.py" \
        --runs_dir "$BA_2D_ROOT" \
        --expected_runs 0 \
        --latex_manifest "$(dirname "$0")/eth3d_offline_ablation_mean_5.json" \
        --latex_output "$EXPERIMENT_ROOT/ablation_table.ltx"
fi
