#!/usr/bin/env bash
# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Prepare ScanNet++, run native BLASt3R inference, and evaluate pose + dense F1.
# Usage:
#   bash scripts/xps/scannet_offline_ablation_mean_5.sh
#   bash scripts/xps/scannet_offline_ablation_mean_5.sh --eval-only
#   bash scripts/xps/scannet_offline_ablation_mean_5.sh --prepare-only
#
# Select scenes/runs with SCENE_LIST="5f99900f09" and RUNS="vggt_rank1".
# Select a retrieval seed with SEED=0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PREPARE="$ROOT/scripts/preprocessing/prepare_scannetpp_multiview.py"
INFERENCE="$ROOT/scripts/run_offline.py"
EVALUATOR="$ROOT/src/blast3r/eval/evaluate_scannetpp_multiview.py"
AGGREGATOR="$ROOT/scripts/aggregate_scannetpp_runs.py"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/scannet++/data_download/scannetpp/data}"
PREPARED_ROOT="${PREPARED_ROOT:-$ROOT/out/scannetpp/prepared}"
RUNS_ROOT="${RUNS_ROOT:-$ROOT/out/scannetpp/ablation}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/.cache/blast3r/scannetpp}"
SEED="${SEED:-0}"
MAX_IMAGES="${MAX_IMAGES:-150}"
CANDIDATE_IMAGES="${CANDIDATE_IMAGES:-150}"
SAMPLING_SEED="${SAMPLING_SEED:-0}"
# Separate selections from old results and from the retrieval seed SEED.
SUBSET_TAG="pool_${CANDIDATE_IMAGES}_n_${MAX_IMAGES}_sample_${SAMPLING_SEED}"
PREPARED_ROOT="$PREPARED_ROOT/$SUBSET_TAG"
RUNS_ROOT="$RUNS_ROOT/$SUBSET_TAG"
POINT_STRIDE="${POINT_STRIDE:-4}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0}"

CKPT_MATCH="${CKPT_MATCH:-naver/blast3r-matcher}"
CKPT_BLAST3R_DEPTH="${CKPT_BLAST3R_DEPTH:-naver/blast3r-depth}"
CKPT_VGGT="${CKPT_VGGT:-$ROOT/checkpoints/vggt_omega_1b_512.pt}"
if [[ ! -f "$CKPT_VGGT" ]]; then
    CKPT_VGGT="${CKPT_VGGT_OMEGA:-/fast-storage/huggingface/hub/models--facebook--VGGT-Omega/snapshots/1041e80fc0e911235d3426b0a3d9a81075111a53/vggt_omega_1b_512.pt}"
fi
if [[ ! -f "$CKPT_VGGT" ]]; then
    echo "VGGT-Omega checkpoint not found; set CKPT_VGGT=/path/to/vggt_omega_1b_512.pt" >&2
    exit 1
fi

DEFAULT_SCENES=(
    09c1414f1b 1ada7a0617 21d970d8de 286b55a2bf 38d58a7a31
    3e8bba0176 40aec5fffa 578511c8a9 5f99900f09 7831862f02
    7bc286c1b6 9071e139d9 acd95847c5 bcd2436daf bde1e479ad
    c4c04e6d6c c5439f4607 cc5237fd77 f3d64c30f8 fb5a96b1a2
)
read -r -a SCENES <<< "${SCENE_LIST:-${DEFAULT_SCENES[*]}}"

DEFAULT_RUNS=(
    no_ba
    ba_2d
    ba_fixed_depth
    ba_alpha_beta
    # ba_full
    vggt_alpha_beta
    vggt_rank1
    vggt_rank4
    vggt_rank8
    vggt_rho95
    # vggt_rank15
)
read -r -a ENABLED_RUNS <<< "${RUNS:-${DEFAULT_RUNS[*]}}"
TOLERANCES=(0.01 0.02 0.05 0.1 0.2 0.5)

MODE=all
case "${1:-}" in
    "") ;;
    --eval-only) MODE=eval ;;
    --prepare-only) MODE=prepare ;;
    *) echo "Usage: $0 [--eval-only|--prepare-only]" >&2; exit 2 ;;
esac

if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi

prepare_scenes() {
    for scene in "${SCENES[@]}"; do
        "$PYTHON" "$PREPARE" \
            --scene "$scene" \
            --dataset-root "$DATA_ROOT" \
            --output-dir "$PREPARED_ROOT/$scene" \
            --max-images "$MAX_IMAGES" \
            --candidate-images "$CANDIDATE_IMAGES" \
            --sampling-seed "$SAMPLING_SEED"
    done
}

run_inference() {
    local arm="$1"
    local output="$RUNS_ROOT/$arm/seed_$SEED"
    local backend=blast3r
    local depth_checkpoint="$CKPT_BLAST3R_DEPTH"
    local extra=()
    local image_dirs=()
    local scene
    for scene in "${SCENES[@]}"; do
        image_dirs+=("$PREPARED_ROOT/$scene/")
    done
    case "$arm" in
        no_ba)
            extra=(--max_iters 0)
            ;;
        ba_2d)
            extra=(--max_iters 100 --weight_z 0 --n_zcfs 2
                   --optim_K True --optim_P True --optim_Z False)
            ;;
        ba_fixed_depth)
            extra=(--max_iters 100 --weight_z 1e-4 --n_zcfs 1
                   --optim_K True --optim_P True --optim_Z False)
            ;;
        ba_alpha_beta)
            extra=(--max_iters 100 --weight_z 1e-4 --n_zcfs 2
                   --optim_K True --optim_P True --optim_Z True)
            ;;
        ba_full)
            extra=(--max_iters 100 --weight_z 1e-4 --n_zcfs 17
                   --optim_K True --optim_P True --optim_Z True)
            ;;
        vggt_alpha_beta)
            backend=vggt_omega
            depth_checkpoint="$CKPT_VGGT"
            extra=(--max_iters 100 --weight_z 1e-4 --n_zcfs 2
                   --optim_K True --optim_P True --optim_Z True
                   --vggt_feature_rank 0)
            ;;
        vggt_rank1)
            backend=vggt_omega
            depth_checkpoint="$CKPT_VGGT"
            extra=(--max_iters 100 --weight_z 1e-4 --n_zcfs 17
                   --optim_K True --optim_P True --optim_Z True
                   --vggt_feature_rank 15 --vggt_effective_rank 1 --vggt_whitening 1)
            ;;
        vggt_rank4)
            backend=vggt_omega
            depth_checkpoint="$CKPT_VGGT"
            extra=(--max_iters 100 --weight_z 1e-4 --n_zcfs 17
                   --optim_K True --optim_P True --optim_Z True
                   --vggt_feature_rank 15 --vggt_effective_rank 4 --vggt_whitening 1)
            ;;
        vggt_rank8)
            backend=vggt_omega
            depth_checkpoint="$CKPT_VGGT"
            extra=(--max_iters 100 --weight_z 1e-4 --n_zcfs 17
                   --optim_K True --optim_P True --optim_Z True
                   --vggt_feature_rank 15 --vggt_effective_rank 8 --vggt_whitening 1)
            ;;
        vggt_rho95)
            backend=vggt_omega
            depth_checkpoint="$CKPT_VGGT"
            extra=(--max_iters 100 --weight_z 1e-4 --n_zcfs 17
                   --optim_K True --optim_P True --optim_Z True
                   --vggt_feature_rank 15 --vggt_feature_rho 0.95 --vggt_whitening 1)
            ;;
        vggt_rank15)
            backend=vggt_omega
            depth_checkpoint="$CKPT_VGGT"
            extra=(--max_iters 100 --weight_z 1e-4 --n_zcfs 17
                   --optim_K True --optim_P True --optim_Z True
                   --vggt_feature_rank 15 --vggt_whitening 1)
            ;;
        *)
            echo "Unknown run: $arm" >&2
            return 2
            ;;
    esac

    echo "Running inference: $arm, seed $SEED"
    "$PYTHON" "$INFERENCE" \
        --images "${image_dirs[@]}" \
        --output "$output" \
        --ckpt_match "$CKPT_MATCH" \
        --ckpt_depth "$depth_checkpoint" \
        --depth_backend "$backend" \
        --size 512 \
        --mono_cam \
        --seed "$SEED" \
        --export_mode full \
        --use_cache \
        --cache_dir "$CACHE_ROOT" \
        --skip_existing \
        "${extra[@]}"
}

evaluate_run() {
    local arm="$1"
    local output="$RUNS_ROOT/$arm/seed_$SEED"
    for scene in "${SCENES[@]}"; do
        local prediction="$output/$scene.pth"
        if [[ ! -f "$prediction" ]]; then
            echo "Skipping missing prediction: $prediction" >&2
            continue
        fi
        "$PYTHON" "$EVALUATOR" \
            --scene "$scene" \
            --dataset-root "$DATA_ROOT" \
            --prepared-scene "$PREPARED_ROOT/$scene" \
            --prediction "$prediction" \
            --point-stride "$POINT_STRIDE" \
            --confidence-threshold "$CONFIDENCE_THRESHOLD" \
            --tolerances "${TOLERANCES[@]}"
    done
}

if [[ "$MODE" != eval ]]; then
    prepare_scenes
fi
if [[ "$MODE" == prepare ]]; then
    exit 0
fi

for arm in "${ENABLED_RUNS[@]}"; do
    if [[ "$MODE" == all ]]; then
        run_inference "$arm"
    fi
    evaluate_run "$arm"
done

if [[ -z "${SCENE_LIST:-}" ]]; then
    "$PYTHON" "$AGGREGATOR" --runs-root "$RUNS_ROOT" --expected-scenes "${#SCENES[@]}"
else
    echo "Skipping automatic CSV aggregation for a SCENE_LIST subset."
fi
