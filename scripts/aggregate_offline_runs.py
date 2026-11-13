#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Aggregate per-scene offline metrics over repeated inference runs."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from blast3r.eval.tables import LABELS


def load_runs(runs_dir, pattern, expected_runs):
    paths = sorted(Path(runs_dir).glob(pattern))
    if expected_runs and len(paths) != expected_runs:
        raise ValueError(
            f'expected {expected_runs} run CSVs matching {pattern!r} under '
            f'{runs_dir}, found {len(paths)}: {[str(path) for path in paths]}'
        )
    if not paths:
        raise ValueError(f'no run CSVs matching {pattern!r} under {runs_dir}')

    runs = {}
    expected_scenes = None
    for path in paths:
        run_name = path.parent.name
        table = pd.read_csv(path, index_col='scene')
        if 'AVG' not in table.index:
            raise ValueError(f'{path} has no AVG row')

        scene_table = table.drop(index='AVG')
        # Evaluation records failed/missing scenes explicitly. The paper table
        # convention assigns those scenes zero rather than dropping a run.
        if 'status' in scene_table:
            failed = ~scene_table['status'].eq('ok')
            metric_columns = scene_table.select_dtypes(include='number').columns
            scene_table.loc[failed, metric_columns] = 0
            table = pd.concat([scene_table, table.loc[['AVG']]])

        scenes = scene_table.index.tolist()
        if expected_scenes is None:
            expected_scenes = scenes
        elif scenes != expected_scenes:
            raise ValueError(f'{path} does not contain the same ordered scenes as the other runs')

        runs[run_name] = table

    return paths, runs, expected_scenes


def aggregate(runs):
    combined = pd.concat(runs, names=['run', 'scene'])
    numeric = combined.select_dtypes(include='number').columns.tolist()
    if not numeric:
        raise ValueError('the run CSVs contain no numeric metrics')

    grouped = combined[numeric].groupby(level='scene', sort=False)
    means = grouped.mean().add_suffix('_mean')
    stds = grouped.std(ddof=1).add_suffix('_std')

    columns = []
    for metric in numeric:
        columns.extend((f'{metric}_mean', f'{metric}_std'))
    return means.join(stds)[columns], numeric


def aggregate_timings(paths, scenes):
    rows = []
    for csv_path in paths:
        run_name = csv_path.parent.name
        for scene in scenes:
            prediction = csv_path.parent / f'{scene}.pth'
            if not prediction.exists():
                # A failed scene still contributes zero time to the mean.
                rows.append({'run': run_name, 'scene': scene, 'total_time_s': 0.0})
                continue

            result = torch.load(prediction, map_location='cpu', weights_only=False)
            try:
                total_time = float(result['timing']['total']['total_time'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{prediction} has no valid total timing') from exc
            rows.append({'run': run_name, 'scene': scene, 'total_time_s': total_time})

    timings = pd.DataFrame(rows).set_index(['run', 'scene'])['total_time_s']
    by_scene = timings.groupby(level='scene', sort=False)
    summary = pd.DataFrame({
        'time_sum_s': by_scene.sum(),
        'time_mean_s': by_scene.mean(),
        'time_std_s': by_scene.std(ddof=1),
    })

    # Match the metric AVG convention: average scenes within each run first,
    # then report mean and sample standard deviation across runs. For the sum
    # column, AVG is the average cumulative time per scene over all runs.
    per_run_avg = timings.groupby(level='run').mean()
    summary.loc['AVG'] = [
        summary['time_sum_s'].mean(),
        per_run_avg.mean(),
        per_run_avg.std(ddof=1),
    ]
    return summary


def render_ablation_table(template_path, manifest_path, output_path, expected_runs,
                          runs_root=None):
    """Fill the ablation table from a JSON list of labeled run directories."""
    template = Path(template_path).read_text()
    manifest = json.loads(Path(manifest_path).read_text())
    if not isinstance(manifest, list) or not manifest:
        raise ValueError('LaTeX manifest must be a non-empty JSON list')

    rows = []
    required = {'label', 'runs_dir', 'ba', 'blast3r_depth', 'vggt_feat_depth'}
    for item in manifest:
        missing = required - item.keys()
        if missing:
            raise ValueError(f'manifest row is missing fields: {sorted(missing)}')
        runs_dir = Path(runs_root) / item['runs_dir'] if runs_root else Path(item['runs_dir'])
        pattern = item.get('pattern', 'seed_*/relpose.csv')
        if item.get('optional') and not list(runs_dir.glob(pattern)):
            metrics = ['', '', '', '']
        else:
            paths, runs, scenes = load_runs(runs_dir, pattern,
                                            item.get('expected_runs', expected_runs))
            summary, _ = aggregate(runs)
            timing = aggregate_timings(paths, scenes)
            avg = summary.join(timing).loc['AVG']
            metrics = [f"{avg['Auc_1_mean']:.2f}", f"{avg['Auc_5_mean']:.2f}",
                       f"{avg['Auc_30_mean']:.2f}", f"{avg['time_mean_s']:.2f}"]
        cells = [item['label'], item['ba'], item['blast3r_depth'], item['vggt_feat_depth'],
                 *metrics]
        rows.append(' & '.join(map(str, cells)) + ' \\\\')

    if '{{ABLATION_ROWS}}' not in template:
        raise ValueError(f'{template_path} has no {{{{ABLATION_ROWS}}}} placeholder')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.replace('{{ABLATION_ROWS}}', '\n'.join(rows)) + '\n')
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs_dir', required=True,
                        help='directory containing one seed_<N> subdirectory per run')
    parser.add_argument('--pattern', default='seed_*/relpose.csv',
                        help='run CSV glob relative to --runs_dir')
    parser.add_argument('--expected_runs', type=int, default=0,
                        help='required run count; 0 detects the number of matching runs (default)')
    parser.add_argument('--output', default=None,
                        help='output CSV (default: <runs_dir>/relpose_mean_std.csv)')
    parser.add_argument('--latex_manifest', default=None,
                        help='JSON list mapping ablation table rows to run directories')
    parser.add_argument('--latex_template',
                        default=str(Path(__file__).resolve().parents[1] / 'templates/ablation_table.ltx'),
                        help='ablation LaTeX template')
    parser.add_argument('--latex_output', default=None,
                        help='write a filled LaTeX table (requires --latex_manifest)')
    parser.add_argument('--latex_runs_root', default=None,
                        help='base directory for relative runs_dir entries in the manifest')
    args = parser.parse_args()

    paths, runs, scenes = load_runs(args.runs_dir, args.pattern, args.expected_runs)
    summary, metrics = aggregate(runs)
    timing_summary = aggregate_timings(paths, scenes)
    summary = summary.join(timing_summary)
    output = Path(args.output or Path(args.runs_dir) / 'relpose_mean_std.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, float_format='%.6g')

    overall = summary.loc['AVG']
    shown = pd.DataFrame({
        'Mean': [overall[f'{metric}_mean'] for metric in metrics],
        'Std': [overall[f'{metric}_std'] for metric in metrics],
    }, index=[LABELS.get(metric, metric) for metric in metrics])

    print(f'Aggregated {len(paths)} runs:')
    for path in paths:
        print(f'  {path}')
    print('\nOverall ETH3D metrics across runs')
    print(shown.to_markdown(floatfmt='.4f'))
    print('\nPer-scene reconstruction time across runs (seconds)')
    print(timing_summary.rename(columns={
        'time_sum_s': 'Sum',
        'time_mean_s': 'Mean',
        'time_std_s': 'Std',
    }).to_markdown(floatfmt='.3f'))
    print(f'\nWrote {output}')
    if args.latex_manifest:
        latex_path = render_ablation_table(
            args.latex_template, args.latex_manifest,
            args.latex_output or str(Path(args.runs_dir) / 'ablation_table.ltx'),
            args.expected_runs, args.latex_runs_root,
        )
        print(f'Wrote {latex_path}')
    elif args.latex_output:
        parser.error('--latex_output requires --latex_manifest')


if __name__ == '__main__':
    main()

"""
RUN THIS

python scripts/aggregate_offline_runs.py \
  --runs_dir out/eth3d/ablation/ba_2d \
  --expected_runs 5 \
  --latex_manifest scripts/xps/eth3d_offline_ablation_mean_5.json \
  --latex_runs_root out/eth3d/ablation \
  --latex_output out/eth3d/ablation/ablation_table.ltx


  OR THIS for pointclouds
.venv/bin/python scripts/eval_offline.py \
  --gt_dir data/eth3d \
  --preds_dir out/eth3d/ablation/vggt_rank15/seed_0 \
  --eval_pointcloud \
  --eth3d_scans_dir data/eth3d_scans \
  --eth3d_evaluator tools/eth3d_multi_view_evaluation/build/ETH3DMultiViewEvaluation

.venv/bin/python scripts/eval_offline.py \
  --gt_dir data/eth3d \
  --preds_dir out/eth3d/ablation_version_2/vggt_rank1/seed_0 \
  --eval_pointcloud \
  --eth3d_scans_dir data/eth3d_scans \
  --eth3d_evaluator tools/eth3d_multi_view_evaluation/build/ETH3DMultiViewEvaluation

"""