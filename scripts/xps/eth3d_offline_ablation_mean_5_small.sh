#!/usr/bin/env bash
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

# Compare depth-based initialization, classical 2D BA, and the full two-stage
# BLASt3R schedule over ETH3D retrieval seeds. All methods reuse the same
# cached depth predictions, tracks and initial poses for a given seed.
# Pass an optional output root; defaults to out/eth3d/ablation.
# Use --eval-only to evaluate existing predictions without running inference.
set -euo pipefail

ETH3D="data/eth3d_small"
EXPERIMENT_ROOT="out/eth3d/ablation_small"
EVAL_ONLY=false
while (($#)); do
    case "$1" in
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        --root)
            [[ $# -ge 2 ]] || { echo "--root requires a directory" >&2; exit 2; }
            EXPERIMENT_ROOT="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *)
            EXPERIMENT_ROOT="$1"
            shift
            ;;
    esac
done
NO_BA_ROOT="$EXPERIMENT_ROOT/no_ba"
BA_2D_ROOT="$EXPERIMENT_ROOT/ba_2d"
BA_FIXED_DEPTH_ROOT="$EXPERIMENT_ROOT/ba_fixed_depth"
BA_ALPHA_BETA_ROOT="$EXPERIMENT_ROOT/ba_alpha_beta"
BA_FULL_ROOT="$EXPERIMENT_ROOT/ba_full"
VGGT_ALPHA_BETA_ROOT="$EXPERIMENT_ROOT/vggt_alpha_beta"
VGGT_RANK15_ROOT="$EXPERIMENT_ROOT/vggt_rank15"

CKPT_MATCH="naver/blast3r-matcher"
CKPT_DEPTH="naver/blast3r-depth"
CKPT_VGGT_OMEGA="${CKPT_VGGT_OMEGA:-/fast-storage/huggingface/hub/models--facebook--VGGT-Omega/snapshots/1041e80fc0e911235d3426b0a3d9a81075111a53/vggt_omega_1b_512.pt}"

# Enable configurations by leaving their names uncommented. Comment out any
# configuration you want to skip.
RUNS=(
    # no_ba
    # ba_2d
    # ba_fixed_depth
    # ba_alpha_beta
    # ba_full
    # vggt_alpha_beta
    vggt_rank15
)

run_enabled() {
    [[ " ${RUNS[*]} " == *" $1 "* ]]
}

for SEED in 0; do
    if run_enabled no_ba; then
        # Depth-based spanning-tree initialization, with no BA updates.
        RUN_OUTPUT="$NO_BA_ROOT/seed_$SEED"
        if [[ "$EVAL_ONLY" != true ]]; then
            python "$(dirname "$0")/../run_offline.py" \
            --images "$ETH3D"/*/ \
            --output "$RUN_OUTPUT" \
            --ckpt_match "$CKPT_MATCH" \
            --ckpt_depth "$CKPT_DEPTH" \
            --size 512 \
            --mono_cam \
            --max_iters 0 \
            --seed "$SEED" \
            --export_mode full \
                --use_cache
        fi

        python "$(dirname "$0")/../eval_offline.py" \
            --gt_dir "$ETH3D" \
            --preds_dir "$RUN_OUTPUT"
    fi

    if run_enabled ba_2d; then
        # Classical sparse BA: K, poses and sparse points use 2D residuals only.
        RUN_OUTPUT="$BA_2D_ROOT/seed_$SEED"
        if [[ "$EVAL_ONLY" != true ]]; then
            python "$(dirname "$0")/../run_offline.py" \
            --images "$ETH3D"/*/ \
            --output "$RUN_OUTPUT" \
            --ckpt_match "$CKPT_MATCH" \
            --ckpt_depth "$CKPT_DEPTH" \
            --size 512 \
            --max_iters 100 \
            --weight_z 0 \
            --n_zcfs 2 \
            --mono_cam \
            --optim_K True \
            --optim_P True \
            --optim_Z False \
            --seed "$SEED" \
            --export_mode full \
                --use_cache
        fi

        python "$(dirname "$0")/../eval_offline.py" \
            --gt_dir "$ETH3D" \
            --preds_dir "$RUN_OUTPUT"
    fi

    if run_enabled ba_fixed_depth; then
        # Depth-regularized BA with the initialized alpha/beta held fixed.
        RUN_OUTPUT="$BA_FIXED_DEPTH_ROOT/seed_$SEED"
        if [[ "$EVAL_ONLY" != true ]]; then
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
            --export_mode full \
                --use_cache
        fi

        python "$(dirname "$0")/../eval_offline.py" \
            --gt_dir "$ETH3D" \
            --preds_dir "$RUN_OUTPUT"
    fi

    if run_enabled ba_alpha_beta; then
        # Depth-regularized BA with only alpha and beta optimized.
        RUN_OUTPUT="$BA_ALPHA_BETA_ROOT/seed_$SEED"
        if [[ "$EVAL_ONLY" != true ]]; then
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
            --export_mode full \
                --use_cache
        fi

        python "$(dirname "$0")/../eval_offline.py" \
            --gt_dir "$ETH3D" \
            --preds_dir "$RUN_OUTPUT"
    fi

    if run_enabled ba_full; then
        # Full schedule: recover cameras with a weak depth term, then fix cameras
        # and stitch all depth components using a stronger depth term.
        RUN_OUTPUT="$BA_FULL_ROOT/seed_$SEED"
        if [[ "$EVAL_ONLY" != true ]]; then
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
            --export_mode full \
                --use_cache
        fi

        python "$(dirname "$0")/../eval_offline.py" \
            --gt_dir "$ETH3D" \
            --preds_dir "$RUN_OUTPUT"
    fi

    if run_enabled vggt_alpha_beta; then
        # Monocular VGGT-Omega depth (one independent forward pass per image)
        # with affine alpha/beta BA. Deliberately bypasses the offline cache.
        RUN_OUTPUT="$VGGT_ALPHA_BETA_ROOT/seed_$SEED"
        if [[ "$EVAL_ONLY" != true ]]; then
            python "$(dirname "$0")/../run_offline.py" \
            --images "$ETH3D"/*/ \
            --output "$RUN_OUTPUT" \
            --ckpt_match "$CKPT_MATCH" \
            --ckpt_depth "$CKPT_VGGT_OMEGA" \
            --depth_backend vggt_omega \
            --size 512 \
            --max_iters 100 \
            --weight_z 1e-4 \
            --n_zcfs 2 \
            --mono_cam \
            --optim_K True \
            --optim_P True \
            --optim_Z True \
            --seed "$SEED" \
            --export_mode full \
            --use_cache 
        fi

        python "$(dirname "$0")/../eval_offline.py" \
            --gt_dir "$ETH3D" \
            --preds_dir "$RUN_OUTPUT"
    fi

    if run_enabled vggt_rank15; then
        RUN_OUTPUT="$VGGT_RANK15_ROOT/seed_$SEED"
        if [[ "$EVAL_ONLY" != true ]]; then
            python "$(dirname "$0")/../run_offline.py" \
            --images "$ETH3D"/*/ \
            --output "$RUN_OUTPUT" \
            --ckpt_match "$CKPT_MATCH" \
            --ckpt_depth "$CKPT_VGGT_OMEGA" \
            --depth_backend vggt_omega \
            --vggt_feature_rank 15 \
            --vggt_whitening 1 \
            --size 512 \
            --max_iters 100 \
            --weight_z 1e-4 \
            --n_zcfs 17 \
            --mono_cam \
            --optim_K True \
            --optim_P True \
            --optim_Z True \
            --seed "$SEED" \
            --export_mode full \
            --use_cache
        fi

        python "$(dirname "$0")/../eval_offline.py" \
            --gt_dir "$ETH3D" \
            --preds_dir "$RUN_OUTPUT"
    fi
done

if run_enabled no_ba; then
    python "$(dirname "$0")/../aggregate_offline_runs.py" \
        --runs_dir "$NO_BA_ROOT" \
        --expected_runs 0
fi

if run_enabled ba_2d; then
    python "$(dirname "$0")/../aggregate_offline_runs.py" \
        --runs_dir "$BA_2D_ROOT" \
        --expected_runs 0
fi

if run_enabled ba_fixed_depth; then
    python "$(dirname "$0")/../aggregate_offline_runs.py" \
        --runs_dir "$BA_FIXED_DEPTH_ROOT" \
        --expected_runs 0
fi

if run_enabled ba_alpha_beta; then
    python "$(dirname "$0")/../aggregate_offline_runs.py" \
        --runs_dir "$BA_ALPHA_BETA_ROOT" \
        --expected_runs 0
fi

if run_enabled ba_full; then
    python "$(dirname "$0")/../aggregate_offline_runs.py" \
        --runs_dir "$BA_FULL_ROOT" \
        --expected_runs 0
fi

if run_enabled vggt_alpha_beta; then
    python "$(dirname "$0")/../aggregate_offline_runs.py" \
        --runs_dir "$VGGT_ALPHA_BETA_ROOT" \
        --expected_runs 0
fi

if run_enabled vggt_rank15; then
    python "$(dirname "$0")/../aggregate_offline_runs.py" \
        --runs_dir "$VGGT_RANK15_ROOT" \
        --expected_runs 0
fi

# Render one LaTeX row per configuration in the manifest, including mean time.
# The optional VGGT row stays blank until it has evaluated runs.
if run_enabled ba_2d && run_enabled ba_fixed_depth && \
   run_enabled ba_alpha_beta && run_enabled ba_full; then
    python "$(dirname "$0")/../aggregate_offline_runs.py" \
        --runs_dir "$BA_2D_ROOT" \
        --expected_runs 0 \
        --latex_manifest "$(dirname "$0")/eth3d_offline_ablation_mean_5.json" \
        --latex_runs_root "$EXPERIMENT_ROOT" \
        --latex_output "$EXPERIMENT_ROOT/ablation_table.ltx"
fi
