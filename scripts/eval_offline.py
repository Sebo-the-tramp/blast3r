#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Relative-pose evaluation of offline reconstructions.

Ground truth is one directory per scene, holding a `gt.pkl` with `poses`
(Nx4x4 camera-to-world) and `filenames`, as written by the preprocessing
scripts. Predictions are one `<scene>.pth` per scene, as written by
`run_offline.py`.
"""
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from blast3r.eval.datasets import has_scene_gt, read_scene_gt_poses
from blast3r.eval.io import read_scene_predictions
from blast3r.eval.metrics.eth3d import (DEFAULT_TOLERANCES, evaluate_prediction,
                                       parse_tolerances, resolve_evaluator)
from blast3r.eval.metrics.relpose import compute_pose_errors
from blast3r.eval.tables import build_table, show, to_latex, write_csv

REPORT = [
    'Auc_1',
    'Auc_3',
    'Auc_5',
    'Auc_30',
    'Racc_5',
    'Tacc_5',
    'n_images',
    'n_gt_images',
    'n_pairs',
]


def eval_scene(gt_poses, predictions):
    missing = [name for name in gt_poses if name not in predictions]
    assert not missing, f'{len(missing)} ground-truth images have no prediction, e.g. {missing[:3]}'

    gt = torch.stack([torch.as_tensor(gt_poses[n]).float() for n in gt_poses])
    pred = torch.stack([torch.as_tensor(predictions[n]['cam2w']).float() for n in gt_poses])
    return compute_pose_errors(gt, pred)


def evaluate(gt_dir, preds_dir):
    scenes = sorted(p for p in Path(gt_dir).iterdir() if has_scene_gt(p))
    assert scenes, f'no scenes with ground truth found under {gt_dir}'

    rows, missing = [], []
    for scene in tqdm(scenes, desc='Evaluating scenes'):
        pred_path = Path(preds_dir) / f'{scene.name}.pth'
        if not pred_path.exists():
            missing.append(scene.name)
            # a scene with no reconstruction scores zero, as in the paper
            rows.append({'scene': scene.name, 'status': 'missing',
                         **{k: 0.0 for k in REPORT if k != 'n_pairs'}})
            continue

        errs = eval_scene(read_scene_gt_poses(scene), read_scene_predictions(pred_path))
        rows.append({'scene': scene.name, 'status': 'ok', **errs})

    return rows, missing


def evaluate_pointclouds(gt_dir, preds_dir, scans_dir, evaluator_path, tolerances,
                         cloud_dir, stride=1, max_depth=float('inf'), conf_threshold=None):
    """Evaluate dense predictions while also preserving the regular pose evaluation."""
    evaluator = resolve_evaluator(evaluator_path)
    scenes = sorted(p for p in Path(gt_dir).iterdir() if has_scene_gt(p))
    rows = []
    for scene in tqdm(scenes, desc='Evaluating ETH3D point clouds'):
        pred_path = Path(preds_dir) / f'{scene.name}.pth'
        if not pred_path.exists():
            continue
        ground_truth_project = Path(scans_dir) / scene.name / 'dslr_scan_eval' / 'scan_alignment.mlp'
        if not ground_truth_project.is_file():
            raise FileNotFoundError(
                f'ETH3D scan alignment is missing: {ground_truth_project}. Extract the '
                '*_dslr_scan_eval archive and pass its scene root with --eth3d_scans_dir.')
        scene_cloud_dir = Path(cloud_dir) / scene.name
        metrics, scale, output = evaluate_prediction(
            read_scene_predictions(pred_path), read_scene_gt_poses(scene), evaluator,
            ground_truth_project, scene_cloud_dir / 'point_cloud.ply', tolerances,
            stride, max_depth, conf_threshold)
        (scene_cloud_dir / 'evaluator.log').write_text(output)
        row = {'scene': scene.name, 'point_count': metrics.point_count,
               'alignment_scale': scale}
        for tolerance, accuracy, completeness, f1_score in zip(
                metrics.tolerances_m, metrics.accuracy, metrics.completeness, metrics.f1_score):
            label = f'{100 * tolerance:g}cm'
            row[f'accuracy_{label}'] = 100 * accuracy
            row[f'completeness_{label}'] = 100 * completeness
            row[f'f1_{label}'] = 100 * f1_score
        rows.append(row)
    if not rows:
        raise RuntimeError('no predictions found for ETH3D point-cloud evaluation')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gt_dir', required=True, help='ground-truth root, one directory per scene')
    parser.add_argument('--preds_dir', required=True, help='directory of <scene>.pth predictions')
    parser.add_argument('--csv', default=None, help='where to write the per-scene CSV '
                                                    '(default: <preds_dir>/relpose.csv)')
    parser.add_argument('--latex', action='store_true', help='also print a LaTeX table')
    parser.add_argument('--eval_pointcloud', action='store_true',
                        help='also run official ETH3D dense point-cloud evaluation')
    parser.add_argument('--eth3d_scans_dir', default=None,
                        help='ETH3D scene root containing <scene>/dslr_scan_eval '
                             '(default: --gt_dir)')
    parser.add_argument('--eth3d_evaluator', default=None,
                        help='path to ETH3DMultiViewEvaluation (or set ETH3D_EVALUATOR)')
    parser.add_argument('--eth3d_tolerances', default=','.join(map(str, DEFAULT_TOLERANCES)),
                        help='comma-separated point-cloud tolerances in metres')
    parser.add_argument('--point_stride', type=int, default=1,
                        help='sample every Nth depth pixel for point-cloud evaluation')
    parser.add_argument('--max_depth', type=float, default=float('inf'),
                        help='discard aligned depths beyond this many metres')
    parser.add_argument('--conf_threshold', type=float, default=None,
                        help='discard dense points below this saved confidence')
    parser.add_argument('--pointcloud_dir', default=None,
                        help='point clouds and evaluator logs (default: <preds_dir>/eth3d_pointclouds)')
    args = parser.parse_args()

    rows, missing = evaluate(args.gt_dir, args.preds_dir)
    df = build_table(rows)

    if missing:
        print(f'\nWarning: {len(missing)} of {len(rows)} scenes had no prediction and '
              f'were scored as zero: {", ".join(missing)}')

    show(df, 'Relative pose accuracy (%)', columns=REPORT)
    write_csv(df, args.csv or str(Path(args.preds_dir) / 'relpose.csv'))
    if args.latex:
        print('\n' + to_latex(df, columns=['Auc_30']))

    if args.eval_pointcloud:
        pointcloud_rows = evaluate_pointclouds(
            args.gt_dir, args.preds_dir, args.eth3d_scans_dir or args.gt_dir,
            args.eth3d_evaluator, parse_tolerances(args.eth3d_tolerances),
            args.pointcloud_dir or str(Path(args.preds_dir) / 'eth3d_pointclouds'),
            args.point_stride, args.max_depth, args.conf_threshold)
        pointcloud_df = build_table(pointcloud_rows)
        show(pointcloud_df, 'ETH3D point-cloud accuracy / completeness / F1 (%)')
        write_csv(pointcloud_df, str(Path(args.preds_dir) / 'eth3d_pointcloud.csv'))


if __name__ == '__main__':
    main()
