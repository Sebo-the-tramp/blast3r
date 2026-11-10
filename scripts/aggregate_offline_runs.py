#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Aggregate per-scene offline metrics over repeated inference runs."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from blast3r.eval.tables import LABELS


def load_runs(runs_dir, pattern, expected_runs):
    paths = sorted(Path(runs_dir).glob(pattern))
    if len(paths) != expected_runs:
        raise ValueError(
            f'expected {expected_runs} run CSVs matching {pattern!r} under '
            f'{runs_dir}, found {len(paths)}: {[str(path) for path in paths]}'
        )

    runs = {}
    expected_scenes = None
    for path in paths:
        run_name = path.parent.name
        table = pd.read_csv(path, index_col='scene')
        if 'AVG' not in table.index:
            raise ValueError(f'{path} has no AVG row')

        scene_table = table.drop(index='AVG')
        if 'status' in scene_table and not scene_table['status'].eq('ok').all():
            bad = scene_table.index[~scene_table['status'].eq('ok')].tolist()
            raise ValueError(f'{path} has missing or failed scenes: {bad}')

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
                raise ValueError(f'missing prediction needed for timing: {prediction}')

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs_dir', required=True,
                        help='directory containing one seed_<N> subdirectory per run')
    parser.add_argument('--pattern', default='seed_*/relpose.csv',
                        help='run CSV glob relative to --runs_dir')
    parser.add_argument('--expected_runs', type=int, default=5,
                        help='required number of complete runs')
    parser.add_argument('--output', default=None,
                        help='output CSV (default: <runs_dir>/relpose_mean_std.csv)')
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


if __name__ == '__main__':
    main()
