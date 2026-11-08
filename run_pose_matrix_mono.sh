#!/bin/bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/experiment_paths.sh"
exec bash "$BLAST3R_ROOT/run_pose_matrix.sh" \
    --vggt_inference mono --output "${BLAST3R_MATRIX_OUTPUT}_mono" "$@"
