#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Fill the final multi-dataset table from per-seed evaluation outputs.

The current ETH3D layout is ``<runs-root>/eth3d/ablation/<method>/seed_N``.
ScanNet++ uses the same layout when its results become available; missing
datasets or metrics are left as ``--`` in the table. Hypersim is omitted from
the default final table.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch


DATASETS = (
    {'name': 'ETH3D', 'directory': 'eth3d', 'f1_file': 'eth3d_pointcloud.csv'},
    {'name': 'ScanNet++', 'directory': 'scannetpp', 'f1_file': 'depth.csv'},
)

# These rows correspond to the two ablation groups in final_table.ltx.
BA_ROWS = (
    ('No', 'no_ba'),
    ('2D', 'ba_2d'),
    ('3D (fixed depth)', 'ba_fixed_depth'),
    (r'3D ($\alpha_{n,1},\beta_n$ free)', 'ba_alpha_beta'),
    ('3D (full depth)', 'ba_full'),
)
OURS_ROWS = (
    (r'$r=0$', 'vggt_alpha_beta'),
    (r'$r=1$', 'vggt_rank1'),
    (r'$r=4$', 'vggt_rank4'),
    (r'$r=8$', 'vggt_rank8'),
    (r'$\rho=0.95$', 'vggt_rho95'),
    (r'$r=16$', 'vggt_rank15'),
)

POSE_COLUMN = 'Racc_1'
F1_COLUMNS = ('f1_1cm', 'f1_2cm', 'f1_5cm', 'f1_20cm')


def csv_paths(run_dir, filename):
    return sorted(run_dir.glob(f'seed_*/{filename}'))


def average_pose(run_dir):
    """Return mean per-run 5-degree pose accuracy, or None if unavailable."""
    values = []
    for path in csv_paths(run_dir, 'relpose.csv'):
        table = pd.read_csv(path, index_col='scene')
        if 'AVG' not in table.index or POSE_COLUMN not in table.columns:
            raise ValueError(f'{path} must contain AVG and {POSE_COLUMN}')
        values.append(float(table.loc['AVG', POSE_COLUMN]))
    return sum(values) / len(values) if values else None


def average_f1(run_dir, filename):
    """Average F1 over all pose scenes and available seeds; absent scenes score zero."""
    values = []
    for path in csv_paths(run_dir, filename):
        pose_path = path.with_name('relpose.csv')
        if not pose_path.is_file():
            continue
        pose = pd.read_csv(pose_path, index_col='scene')
        scenes = pose.index.drop('AVG', errors='ignore')
        table = pd.read_csv(path, index_col='scene')
        missing = set(F1_COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(f'{path} is missing F1 columns: {sorted(missing)}')
        # Use the pose evaluation's scene list so point-cloud failures count as zero.
        per_scene = table.reindex(scenes)[list(F1_COLUMNS)].fillna(0)
        values.append(per_scene.mean(axis=0))
    if not values:
        return None
    return pd.DataFrame(values).mean(axis=0).to_dict()


def average_time(run_dir):
    """Average total reconstruction time over scenes and runs, with failures at zero."""
    per_run = []
    for pose_path in csv_paths(run_dir, 'relpose.csv'):
        pose = pd.read_csv(pose_path, index_col='scene')
        scenes = pose.index.drop('AVG', errors='ignore')
        times = []
        for scene in scenes:
            prediction = pose_path.parent / f'{scene}.pth'
            if not prediction.is_file():
                times.append(0.0)
                continue
            result = torch.load(prediction, map_location='cpu', weights_only=False)
            try:
                times.append(float(result['timing']['total']['total_time']))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{prediction} has no valid total timing') from exc
        if times:
            per_run.append(sum(times) / len(times))
    return sum(per_run) / len(per_run) if per_run else None


def collect_row_metrics(runs_root, method):
    datasets = []
    for dataset in DATASETS:
        run_dir = runs_root / dataset['directory'] / 'ablation_version_2' / method
        pose = average_pose(run_dir)
        f1 = average_f1(run_dir, dataset['f1_file'])
        timing = average_time(run_dir)
        datasets.append([pose, *(f1.get(column) if f1 else None for column in F1_COLUMNS), timing])
    return datasets


def format_metric(value, best=False):
    if value is None or pd.isna(value):
        return '--'
    rendered = f'{value:.2f}'
    return rf'\textbf{{{rendered}}}' if best else rendered


def render_rows(row_specs, all_metrics, comparison_rows):
    # Determine the best score separately for each dataset and metric column.
    maxima = []
    for dataset_idx in range(len(DATASETS)):
        dataset_maxima = []
        for metric_idx in range(6):
            candidates = [metrics[dataset_idx][metric_idx]
                          for _, metrics in comparison_rows
                          if metrics[dataset_idx][metric_idx] is not None]
            if not candidates:
                dataset_maxima.append(None)
            elif metric_idx == 5:  # Time is better when smaller.
                dataset_maxima.append(min(candidates))
            else:
                dataset_maxima.append(max(candidates))
        maxima.append(dataset_maxima)

    rendered_rows = []
    for label, metrics in all_metrics:
        cells = [label]
        for dataset_idx, dataset_metrics in enumerate(metrics):
            for metric_idx, value in enumerate(dataset_metrics):
                cells.append(format_metric(
                    value, value is not None and value == maxima[dataset_idx][metric_idx]
                ))
        rendered_rows.append(' & '.join(cells) + r' \\')
    return '\n'.join(rendered_rows)


def fill_template(template_path, output_path, runs_root):
    template = Path(template_path).read_text()
    required = ('{{BA_ROWS}}', '{{OURS_ROWS}}')
    for placeholder in required:
        if placeholder not in template:
            raise ValueError(f'{template_path} is missing {placeholder}')

    row_metrics = {}
    for _, method in (*BA_ROWS, *OURS_ROWS):
        row_metrics[method] = collect_row_metrics(runs_root, method)

    ba_metrics = [(label, row_metrics[method]) for label, method in BA_ROWS]
    ours_metrics = [(label, row_metrics[method]) for label, method in OURS_ROWS]
    comparison_rows = ba_metrics + ours_metrics
    ba = render_rows(BA_ROWS, ba_metrics, comparison_rows)
    ours = render_rows(OURS_ROWS, ours_metrics, comparison_rows)
    output = template.replace('{{BA_ROWS}}', ba).replace('{{OURS_ROWS}}', ours)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output + '\n')
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs_root', default='out',
                        help='root containing <dataset>/ablation/<method>/seed_N folders')
    parser.add_argument('--template',
                        default=str(Path(__file__).resolve().parents[1] / 'templates/final_table.ltx'),
                        help='final LaTeX table template')
    parser.add_argument('--output', default=None,
                        help='output path (default: <runs_root>/final_table.ltx)')
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output = fill_template(args.template, args.output or runs_root / 'final_table.ltx', runs_root)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
