#!/usr/bin/env python3
"""Relative pose and dense-cloud F1 of offline reconstructions on AerialMegaDepth sets.

Mirrors src/blast3r/eval/evaluate_scannetpp_multiview.py with the AerialMegaDepth ground truth
of sc3r's scripts/setup_aerialmegadepth_subsets.py: per-image GT cameras
(`<set>/gt_sparse/cameras.txt`, SIMPLE_PINHOLE / PINHOLE) and GT MVS depth
(`<set>/depth/<stem>.npy`, zero where invalid). Per set:

* Pose: blast3r's compute_pose_errors against `gt.pkl` (as eval_offline.py).
* Cloud: the prediction is put in the GT frame by a Sim(3) fit on camera centres. The GT
  cloud lifts the GT depth, sampled (nearest) at the prediction grid's pixel centres, with the
  GT camera scaled to that grid; the predicted cloud keeps only pixels with valid GT depth.
  F1 is symmetric nearest-neighbour at metric tolerances. Images are assumed to be resized
  without cropping (1920x1080 -> 512x288 at --size 512), which is checked.

Writes `<preds_dir>/relpose.csv` and `<preds_dir>/amd_pointcloud.csv` (AVG row included).
Missing predictions score zero.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from blast3r.eval.datasets import has_scene_gt, read_scene_gt_poses
from blast3r.eval.io import read_scene_predictions
from blast3r.eval.metrics.eth3d import estimate_similarity
from blast3r.eval.metrics.relpose import compute_pose_errors
from blast3r.eval.scannetpp import cloud_scores, lift_depth_array_to_world
from blast3r.eval.tables import build_table, show, write_csv

POSE_COLUMNS = ['Auc_1', 'Auc_3', 'Auc_5', 'Auc_30', 'Racc_5', 'Tacc_5', 'n_pairs']


def read_gt_cameras(path):
    """{name: (f_x, f_y, c_x, c_y, width, height)} from the set's COLMAP text model."""
    cameras = {}
    for line in (path / 'cameras.txt').read_text().splitlines():
        fields = line.split()
        if line.startswith('#') or not fields:
            continue
        model, width, height, params = fields[1], int(fields[2]), int(fields[3]), list(map(float, fields[4:]))
        if model == 'SIMPLE_PINHOLE':
            f, cx, cy = params
            cameras[fields[0]] = (f, f, cx, cy, width, height)
        elif model == 'PINHOLE':
            cameras[fields[0]] = (*params, width, height)
        else:
            raise ValueError(f'unsupported GT camera model {model}')
    by_name = {}
    for line in (path / 'images.txt').read_text().splitlines():
        fields = line.split()
        if not line.startswith('#') and len(fields) == 10:
            by_name[fields[9]] = cameras[fields[8]]
    return by_name


def depth_on_grid(depth, shape):
    rows = ((np.arange(shape[0]) + 0.5) * depth.shape[0] / shape[0]).astype(int)
    columns = ((np.arange(shape[1]) + 0.5) * depth.shape[1] / shape[1]).astype(int)
    sampled = depth[np.ix_(rows, columns)]
    return np.where(np.isfinite(sampled) & (sampled > 0), sampled, 0).astype(np.float32)


def intrinsics_on_grid(camera, shape):
    fx, fy, cx, cy, width, height = camera
    sx, sy = shape[1] / width, shape[0] / height
    if abs(sx - sy) > 1e-3 * sx:
        raise ValueError(f'prediction grid {shape} is not an uncropped resize of {width}x{height}')
    return np.array([[fx * sx, 0, cx * sx - 0.5], [0, fy * sy, cy * sy - 0.5], [0, 0, 1]])


def evaluate_set(gt_scene, subset, prediction_path, stride, tolerances):
    gt_poses = read_scene_gt_poses(gt_scene)
    predictions = read_scene_predictions(prediction_path)
    names = list(gt_poses)
    missing = [n for n in names if n not in predictions]
    assert not missing, f'{len(missing)} GT images have no prediction, e.g. {missing[:3]}'

    gt = torch.stack([torch.as_tensor(gt_poses[n]).float() for n in names])
    pred = torch.stack([torch.as_tensor(predictions[n]['cam2w']).float() for n in names])
    pose = compute_pose_errors(gt, pred)

    pred_centres = np.stack([np.asarray(predictions[n]['cam2w'])[:3, 3] for n in names])
    gt_centres = np.stack([np.asarray(gt_poses[n])[:3, 3] for n in names])
    scale, rotation, translation = estimate_similarity(pred_centres, gt_centres)

    cameras = read_gt_cameras(subset / 'gt_sparse')
    gt_parts, pred_parts = [], []
    for name in names:
        view = predictions[name]
        depth = np.asarray(view['depth'], dtype=np.float32)
        gt_depth = depth_on_grid(np.load(subset / 'depth' / f'{Path(name).stem}.npy'), depth.shape)
        gt_valid = gt_depth > 0
        gt_points, _, _ = lift_depth_array_to_world(
            gt_depth, gt_valid, intrinsics_on_grid(cameras[name], depth.shape),
            np.asarray(gt_poses[name], dtype=np.float64), stride=stride, max_depth=float('inf'))
        gt_parts.append(gt_points)

        c2w = np.asarray(view['cam2w'], dtype=np.float64)
        aligned = np.eye(4)
        aligned[:3, :3] = rotation @ c2w[:3, :3]
        aligned[:3, 3] = scale * rotation @ c2w[:3, 3] + translation
        valid = gt_valid & np.isfinite(depth) & (depth > 0)
        points, _, _ = lift_depth_array_to_world(
            depth * scale, valid, np.asarray(view['K'], dtype=np.float64), aligned,
            stride=stride, max_depth=float('inf'))
        pred_parts.append(points)

    points, gt_points = np.concatenate(pred_parts), np.concatenate(gt_parts)
    scores = cloud_scores(points, gt_points, tolerances)
    cloud = {'point_count': len(points), 'gt_point_count': len(gt_points), 'sim3_scale': scale}
    for tolerance, acc, comp, f1 in zip(tolerances, scores['accuracy'],
                                        scores['completeness'], scores['f1_score']):
        label = f'{100 * tolerance:g}cm'
        cloud[f'accuracy_{label}'] = 100 * acc
        cloud[f'completeness_{label}'] = 100 * comp
        cloud[f'f1_{label}'] = 100 * f1
    return pose, cloud


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gt_dir', required=True, help='prepared sets (gt.pkl per set)')
    parser.add_argument('--subsets_dir', required=True, help='sets with gt_sparse/ and depth/')
    parser.add_argument('--preds_dir', required=True)
    parser.add_argument('--point_stride', type=int, default=4)
    parser.add_argument('--tolerances', type=float, nargs='+', default=[0.5, 1, 2, 5, 10, 20])
    args = parser.parse_args()

    pose_rows, cloud_rows = [], []
    for gt_scene in sorted(p for p in Path(args.gt_dir).iterdir() if has_scene_gt(p)):
        prediction = Path(args.preds_dir) / f'{gt_scene.name}.pth'
        if not prediction.is_file():
            print(f'Missing prediction, scored as zero: {prediction}')
            pose_rows.append({'scene': gt_scene.name, 'status': 'missing',
                              **{k: 0.0 for k in POSE_COLUMNS if k != 'n_pairs'}})
            cloud_rows.append({'scene': gt_scene.name, **{
                f'f1_{100 * t:g}cm': 0.0 for t in args.tolerances}})
            continue
        pose, cloud = evaluate_set(gt_scene, Path(args.subsets_dir) / gt_scene.name, prediction,
                                   args.point_stride, args.tolerances)
        pose_rows.append({'scene': gt_scene.name, 'status': 'ok', **pose})
        cloud_rows.append({'scene': gt_scene.name, **cloud})
        f1 = ' '.join(f'{cloud[f"f1_{100 * t:g}cm"]:.1f}' for t in args.tolerances)
        print(f'{gt_scene.name}: AUC@1 {pose["Auc_1"]:.2f} AUC@5 {pose["Auc_5"]:.2f} F1 [{f1}]',
              flush=True)

    pose_df, cloud_df = build_table(pose_rows), build_table(cloud_rows)
    show(pose_df, 'Relative pose accuracy (%)', columns=POSE_COLUMNS)
    show(cloud_df, 'Dense-cloud F1 (%)',
         columns=[f'f1_{100 * t:g}cm' for t in args.tolerances])
    write_csv(pose_df, str(Path(args.preds_dir) / 'relpose.csv'))
    write_csv(cloud_df, str(Path(args.preds_dir) / 'amd_pointcloud.csv'))


if __name__ == '__main__':
    main()
