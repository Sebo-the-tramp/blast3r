#!/usr/bin/env bash

ETH3D="data/eth3d"
OUTPUT="out/eth3d/all"
CKPT_MATCH=${3:-naver/blast3r-matcher}
CKPT_DEPTH=${4:-naver/blast3r-depth}

# python "$(dirname "$0")/../run_offline.py" \
#     --images "$ETH3D"/*/ \
#     --output "$OUTPUT" \
#     --ckpt_match "$CKPT_MATCH" \
#     --ckpt_depth "$CKPT_DEPTH" \
#     --size 512 \
#     --mono_cam \
#     --export_mode full

# python "$(dirname "$0")/../eval_offline.py" --gt_dir "$ETH3D" --preds_dir "$OUTPUT"


.venv/bin/python scripts/eval_offline.py \
  --gt_dir data/eth3d \
  --preds_dir out/eth3d/all \
  --eval_pointcloud \
  --eth3d_scans_dir data/eth3d_scans