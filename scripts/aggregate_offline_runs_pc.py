#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Aggregate ETH3D point-cloud F1 results and render the ablation LaTeX table."""

import argparse
import json
from pathlib import Path

import pandas as pd


F1_COLUMNS = ('f1_1cm', 'f1_2cm', 'f1_5cm', 'f1_10cm')


def aggregate_pointcloud(runs_dir, pattern, expected_runs):
    paths = sorted(Path(runs_dir).glob(pattern))
    if not paths:
        raise ValueError(f'no point-cloud CSVs matching {pattern!r} under {runs_dir}')
    if expected_runs and len(paths) != expected_runs:
        raise ValueError(
            f'expected {expected_runs} point-cloud CSVs matching {pattern!r} under '
            f'{runs_dir}, found {len(paths)}: {[str(path) for path in paths]}'
        )

    rows = []
    for path in paths:
        table = pd.read_csv(path, index_col='scene')
        missing = set(F1_COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(f'{path} is missing F1 columns: {sorted(missing)}')
        if 'AVG' not in table.index:
            raise ValueError(f'{path} has no AVG row')
        rows.append(table.loc['AVG', list(F1_COLUMNS)].astype(float))

    # The CSV AVG row is the full-scene mean for each seed. Average those rows
    # across seeds; with one seed this returns its reported metrics unchanged.
    return pd.DataFrame(rows).mean(axis=0)


def render_table(template_path, manifest_path, output_path, runs_root, expected_runs):
    template = Path(template_path).read_text()
    manifest = json.loads(Path(manifest_path).read_text())
    if not isinstance(manifest, list) or not manifest:
        raise ValueError('manifest must be a non-empty JSON list')

    entries = []
    required = {'label', 'runs_dir', 'ba', 'blast3r_depth', 'vggt_feat_depth'}
    for item in manifest:
        missing = required - item.keys()
        if missing:
            raise ValueError(f'manifest row is missing fields: {sorted(missing)}')
        run_dir = Path(runs_root) / item['runs_dir']
        pattern = item.get('pattern', 'seed_*/eth3d_pointcloud.csv')
        paths = sorted(run_dir.glob(pattern))
        if item.get('optional') and not paths:
            values = [None, None, None, None]
        else:
            summary = aggregate_pointcloud(
                run_dir, pattern, item.get('expected_runs', expected_runs)
            )
            values = [float(summary[column]) for column in F1_COLUMNS]
        entries.append((item, values))

    maxima = [
        max((values[i] for _, values in entries if values[i] is not None), default=None)
        for i in range(len(F1_COLUMNS))
    ]
    rows = []
    for item, values in entries:
        if item.get('separator_before'):
            rows.append(r'\midrule')
        rendered = []
        for i, value in enumerate(values):
            if value is None:
                rendered.append('')
            elif value == maxima[i]:
                rendered.append(r'\textbf{' + f'{value:.2f}' + '}')
            else:
                rendered.append(f'{value:.2f}')
        cells = [item['label'], item['ba'], item['blast3r_depth'],
                 item['vggt_feat_depth'], *rendered]
        rows.append(' & '.join(map(str, cells)) + ' \\\\')

    if '{{POINTCLOUD_ROWS}}' not in template:
        raise ValueError(f'{template_path} has no {{{{POINTCLOUD_ROWS}}}} placeholder')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.replace('{{POINTCLOUD_ROWS}}', '\n'.join(rows)) + '\n')
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs_root', default='out/eth3d/ablation',
                        help='base directory containing the configuration subdirectories')
    parser.add_argument('--manifest', default='scripts/xps/eth3d_pointcloud_ablation.json',
                        help='JSON manifest describing table rows')
    parser.add_argument('--template',
                        default=str(Path(__file__).resolve().parents[1] / 'templates/ablation_table_pc.ltx'),
                        help='point-cloud LaTeX template')
    parser.add_argument('--output', default=None,
                        help='output LaTeX file (default: <runs_root>/ablation_table_pc.ltx)')
    parser.add_argument('--expected_runs', type=int, default=0,
                        help='required CSV count per row; 0 detects the count (default)')
    args = parser.parse_args()

    output = render_table(
        args.template, args.manifest,
        args.output or str(Path(args.runs_root) / 'ablation_table_pc.ltx'),
        args.runs_root, args.expected_runs,
    )
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
