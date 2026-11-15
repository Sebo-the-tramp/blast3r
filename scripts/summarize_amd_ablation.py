#!/usr/bin/env python3
"""Table of the AerialMegaDepth ablation (jobs/job_blast3r_amd_ablation.sh), one row per config.

    PYTHONPATH=src .venv/bin/python scripts/summarize_amd_ablation.py [out/aerialmegadepth/ablation]

Pose AUC@1 / AUC@5 (relpose.csv AVG), dense F1 at 0.5 / 1 / 2 / 5 m (amd_pointcloud.csv AVG,
missing sets score zero) and mean total reconstruction time per set, averaged over seed_* runs.
Writes <root>/summary.txt and prints it.
"""
import sys
from pathlib import Path

import pandas as pd
import torch

ROWS = (('No BA', 'no_ba'), ('2D', 'ba_2d'), ('3D (fixed depth)', 'ba_fixed_depth'),
        ('3D (alpha,beta free)', 'ba_alpha_beta'), ('3D (full depth)', 'ba_full'),
        ('r=0', 'vggt_alpha_beta'), ('r=1', 'vggt_rank1'), ('r=4', 'vggt_rank4'),
        ('r=8', 'vggt_rank8'), ('rho=0.95', 'vggt_rho95'), ('r=16', 'vggt_rank15'),
        ('r=32', 'vggt_rank32'), ('r=64', 'vggt_rank64'), ('r=full (256)', 'vggt_rank256'))
F1 = ('f1_50cm', 'f1_100cm', 'f1_200cm', 'f1_500cm')


def row(run_dir):
    values = []
    for seed in sorted(run_dir.glob('seed_*')):
        pose_path, cloud_path = seed / 'relpose.csv', seed / 'amd_pointcloud.csv'
        if not pose_path.is_file() or not cloud_path.is_file():
            continue
        pose = pd.read_csv(pose_path, index_col='scene')
        cloud = pd.read_csv(cloud_path, index_col='scene')
        scenes = pose.index.drop('AVG')
        times = []
        for scene in scenes:
            prediction = seed / f'{scene}.pth'
            times.append(float(torch.load(prediction, map_location='cpu', weights_only=False)
                               ['timing']['total']['total_time']) if prediction.is_file() else 0.0)
        values.append([pose.loc['AVG', 'Auc_1'], pose.loc['AVG', 'Auc_5'],
                       *cloud.reindex(scenes)[list(F1)].fillna(0).mean(), sum(times) / len(times)])
    return pd.DataFrame(values).mean().tolist() if values else None


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'out/aerialmegadepth/ablation')
    columns = ['Pose AUC@1', 'AUC@5', 'F1@0.5m', 'F1@1m', 'F1@2m', 'F1@5m', 'Time(s)']
    table = pd.DataFrame({label: row(root / method) or [float('nan')] * len(columns)
                          for label, method in ROWS}, index=columns).T
    text = ('AerialMegaDepth, 8 sets x 150 images (0015/0022 n150 d0-d3), seed mean\n'
            + table.to_markdown(floatfmt='.2f') + '\n')
    (root / 'summary.txt').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
