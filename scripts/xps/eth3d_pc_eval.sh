#!/usr/bin/env bash
# Evaluate ETH3D point clouds for selected VGGT feature-rank ablations.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
EXPERIMENT_ROOT="${1:-$REPO_ROOT/out/eth3d/ablation_version_2}"
SEED="${SEED:-0}"
JOBS="${JOBS:-1}"
GT_DIR="${GT_DIR:-$REPO_ROOT/data/eth3d}"
SCANS_DIR="${SCANS_DIR:-$REPO_ROOT/data/eth3d_scans}"

# Pass run names as arguments to select a subset. With no arguments, evaluate
# all listed runs. Override this default list by editing this array.
DEFAULT_RUNS=(
    no_ba
    ba_2d
    ba_fixed_depth
    ba_alpha_beta
    ba_full
    
    # vggt_alpha_beta
    # vggt_rank1
    # vggt_rank4 
    # vggt_rank8 
    # vggt_rho95 
    # vggt_rank15
)
RUNS=("${@:-${DEFAULT_RUNS[@]}}")

if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found or not executable: $PYTHON" >&2
    echo "Set PYTHON=/path/to/python to override it." >&2
    exit 1
fi
if (( JOBS < 1 )); then
    echo "JOBS must be at least 1 (got $JOBS)" >&2
    exit 1
fi

evaluate() {
    local run="$1"
    local predictions="$EXPERIMENT_ROOT/$run/seed_$SEED"
    if [[ ! -d "$predictions" ]]; then
        echo "Missing prediction directory: $predictions" >&2
        return 1
    fi
    echo "Evaluating $run (seed $SEED)"
    "$PYTHON" "$HERE/../eval_offline.py" \
        --gt_dir "$GT_DIR" \
        --preds_dir "$predictions" \
        --eval_pointcloud \
        --eth3d_scans_dir "$SCANS_DIR"
}

if (( ${#RUNS[@]} == 0 )); then
    echo "No runs selected." >&2
    exit 2
fi

for run in "${RUNS[@]}"; do
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
    evaluate "$run" &
done
wait
