# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Dense point-cloud evaluation with the official ETH3D evaluator."""
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blast3r.utils.geometry import depthmap_to_pts3d


DEFAULT_TOLERANCES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_RESULT_LINE = re.compile(
    r"^(Tolerances|Completenesses|Accuracies|F1-scores):\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Eth3dMetrics:
    point_count: int
    tolerances_m: tuple[float, ...]
    accuracy: tuple[float, ...]
    completeness: tuple[float, ...]
    f1_score: tuple[float, ...]


def parse_tolerances(value):
    values = ([float(item.strip()) for item in value.split(',')]
              if isinstance(value, str) else [float(item) for item in value])
    if not values or any(not np.isfinite(item) or item <= 0 for item in values):
        raise ValueError('ETH3D tolerances must be positive finite values')
    return tuple(sorted(set(values)))


def resolve_evaluator(explicit_path=None):
    """Find the official ``ETH3DMultiViewEvaluation`` executable."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    if os.environ.get('ETH3D_EVALUATOR'):
        candidates.append(Path(os.environ['ETH3D_EVALUATOR']).expanduser())
    if shutil.which('ETH3DMultiViewEvaluation'):
        candidates.append(Path(shutil.which('ETH3DMultiViewEvaluation')))
    candidates.append(
        _REPO_ROOT / 'tools' / 'eth3d_multi_view_evaluation' / 'build'
        / 'ETH3DMultiViewEvaluation')
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(
        'ETH3DMultiViewEvaluation was not found; pass --eth3d_evaluator, set '
        'ETH3D_EVALUATOR, put it on PATH, or build the vendored source under '
        'tools/eth3d_multi_view_evaluation/build')


def estimate_similarity(source, target):
    """Return ``scale, rotation, translation`` mapping source points to target.

    This is the Umeyama least-squares similarity fit. It is used to put a
    monocular reconstruction into the metric ETH3D ground-truth frame.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f'expected matching [N, 3] arrays, got {source.shape} and {target.shape}')
    if len(source) < 3:
        raise ValueError('at least three cameras are required for similarity alignment')

    source_mean, target_mean = source.mean(0), target.mean(0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = np.mean(np.sum(source_centered ** 2, axis=1))
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError('predicted camera centres are degenerate')

    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    signs = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        signs[-1] = -1
    rotation = u @ np.diag(signs) @ vt
    scale = float(np.dot(singular_values, signs) / source_variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f'invalid similarity scale: {scale}')
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def collect_aligned_points(predictions, gt_poses, *, stride=1, max_depth=np.inf,
                           conf_threshold=None):
    """Lift saved BLASt3R depths and Sim(3)-align them to the ETH3D frame."""
    if stride < 1:
        raise ValueError('point stride must be positive')
    if max_depth <= 0:
        raise ValueError('max depth must be positive')
    names = list(gt_poses)
    missing = [name for name in names if name not in predictions]
    if missing:
        raise ValueError(f'{len(missing)} ground-truth images have no prediction')
    without_depth = [name for name in names if 'depth' not in predictions[name]]
    if without_depth:
        raise ValueError(
            f'{len(without_depth)} predictions have no depth (use --export_mode full), '
            f'e.g. {without_depth[:3]}')

    pred_centres = np.stack([np.asarray(predictions[name]['cam2w'])[:3, 3] for name in names])
    gt_centres = np.stack([np.asarray(gt_poses[name])[:3, 3] for name in names])
    scale, rotation, translation = estimate_similarity(pred_centres, gt_centres)

    point_parts = []
    for name in names:
        view = predictions[name]
        depth = np.asarray(view['depth'], dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f'{name}: expected a 2D depth map, got {depth.shape}')
        points, sub = depthmap_to_pts3d(
            depth, np.asarray(view['K']), np.asarray(view['cam2w']), stride=stride)
        sampled_depth = depth[sub, sub]
        valid = np.isfinite(points).all(axis=-1) & np.isfinite(sampled_depth)
        valid &= sampled_depth > 0
        # Depth is in the predicted similarity frame; scale converts it to metres.
        valid &= sampled_depth * scale <= max_depth
        if conf_threshold is not None:
            if 'conf' not in view:
                raise ValueError(f'{name}: confidence threshold requested but no confidence was saved')
            valid &= np.asarray(view['conf'])[sub, sub] >= conf_threshold
        points = points[valid]
        points = scale * (points @ rotation.T) + translation
        point_parts.append(points.astype(np.float32, copy=False))

    if not point_parts or not any(len(part) for part in point_parts):
        raise ValueError('no valid points remained after filtering')
    return np.concatenate(point_parts), scale


def write_binary_point_cloud(path, points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f'expected finite [N, 3] points, got {points.shape}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        'ply\nformat binary_little_endian 1.0\ncomment generated by BLASt3R\n'
        f'element vertex {len(points)}\nproperty float x\nproperty float y\n'
        'property float z\nend_header\n'
    ).encode('ascii')
    with path.open('wb') as output:
        output.write(header)
        np.asarray(points, dtype='<f4').tofile(output)


def evaluate_cloud(evaluator, ground_truth_project, point_cloud_path, tolerances):
    command = [
        str(evaluator), '--reconstruction_ply_path', str(point_cloud_path),
        '--ground_truth_mlp_path', str(ground_truth_project),
        '--tolerances', ','.join(f'{value:g}' for value in tolerances),
    ]
    try:
        completed = subprocess.run(command, check=True, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'official ETH3D evaluation failed:\n{error.stdout}') from error
    values = {
        name: tuple(float(token) for token in content.split())
        for name, content in _RESULT_LINE.findall(completed.stdout)
    }
    required = {'Tolerances', 'Completenesses', 'Accuracies', 'F1-scores'}
    missing = required - values.keys()
    if missing:
        raise RuntimeError(
            f'could not parse ETH3D evaluator output; missing {sorted(missing)}\n'
            + completed.stdout)
    count = len(values['Tolerances'])
    if count == 0 or any(len(values[name]) != count for name in required):
        raise RuntimeError('ETH3D evaluator returned inconsistent metric lengths')
    return Eth3dMetrics(
        point_count=0,  # supplied by evaluate_prediction below
        tolerances_m=values['Tolerances'], accuracy=values['Accuracies'],
        completeness=values['Completenesses'], f1_score=values['F1-scores']), completed.stdout


def evaluate_prediction(predictions, gt_poses, evaluator, ground_truth_project, cloud_path,
                        tolerances=DEFAULT_TOLERANCES, stride=1, max_depth=np.inf,
                        conf_threshold=None):
    points, alignment_scale = collect_aligned_points(
        predictions, gt_poses, stride=stride, max_depth=max_depth,
        conf_threshold=conf_threshold)
    write_binary_point_cloud(cloud_path, points)
    metrics, output = evaluate_cloud(evaluator, ground_truth_project, cloud_path, tolerances)
    metrics = Eth3dMetrics(len(points), metrics.tolerances_m, metrics.accuracy,
                           metrics.completeness, metrics.f1_score)
    return metrics, alignment_scale, output
