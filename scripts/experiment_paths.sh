# shellcheck shell=bash
# Shared paths for run.sh and eval.sh. Source this file; paths are relative to
# the BLASt3R checkout unless absolute. See docs/BENCHMARKS.md for overrides.
BLAST3R_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BLAST3R_ROOT" || return 1

# Local configuration uses ordinary shell assignments (export is optional).
if [[ -n "${BLAST3R_CONFIG:-}" ]]; then
    if [[ ! -f "$BLAST3R_CONFIG" ]]; then
        echo "Missing BLAST3R_CONFIG: $BLAST3R_CONFIG" >&2
        return 1
    fi
    # shellcheck source=/dev/null
    source "$BLAST3R_CONFIG" || return 1
elif [[ -f "$BLAST3R_ROOT/experiments.local.sh" ]]; then
    # shellcheck source=/dev/null
    source "$BLAST3R_ROOT/experiments.local.sh" || return 1
fi

: "${BLAST3R_DATA_ROOT:=/proj/vlarsson/datasets}"
: "${BLAST3R_CHECKPOINT_ROOT:=/proj/vlarsson/checkpoints}"
: "${BLAST3R_VGGT_CHECKPOINT:=$BLAST3R_CHECKPOINT_ROOT/vggt_omega_1b_512.pt}"
: "${BLAST3R_MIP360_ROOT:=$BLAST3R_DATA_ROOT/mip360}"
: "${BLAST3R_ETH3D_ROOT:=$BLAST3R_DATA_ROOT/eth3d-eval/eth3d_processed}"
: "${BLAST3R_BICYCLE_IMAGES:=$BLAST3R_MIP360_ROOT/bicycle/images_2}"
if [[ -d "$BLAST3R_ETH3D_ROOT/courtyard/undistorted_images" ]]; then
    : "${BLAST3R_COURTYARD_IMAGES:=$BLAST3R_ETH3D_ROOT/courtyard/undistorted_images}"
else
    : "${BLAST3R_COURTYARD_IMAGES:=$BLAST3R_ETH3D_ROOT/courtyard/images/dslr_images_undistorted}"
fi
: "${BLAST3R_OUTPUT_ROOT:=$BLAST3R_ROOT/out}"
: "${BLAST3R_EVAL_ROOT:=$BLAST3R_ROOT/outputs/evaluation/eth3d}"
: "${BLAST3R_ETH3D_GT:=$BLAST3R_EVAL_ROOT/gt}"
if [[ -x "$BLAST3R_ROOT/.venv/bin/python" ]]; then
    : "${BLAST3R_PYTHON:=$BLAST3R_ROOT/.venv/bin/python}"
else
    : "${BLAST3R_PYTHON:=python}"
fi
