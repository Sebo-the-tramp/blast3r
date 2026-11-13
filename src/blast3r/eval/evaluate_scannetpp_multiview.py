#!/usr/bin/env python3
"""Evaluate one native BLASt3R prediction on prepared ScanNet++ frames.

Reports the repository's relative-pose metrics and symmetric point-cloud F1
against ScanNet++ laser-mesh rendered depths. The prediction must be a ``.pth``
file written by ``scripts/run_offline.py --export_mode full``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from blast3r.eval.datasets import read_scene_gt_poses  # noqa: E402
from blast3r.eval.io import read_scene_predictions  # noqa: E402
from blast3r.eval.metrics.eth3d import estimate_similarity  # noqa: E402
from blast3r.eval.metrics.relpose import compute_pose_errors  # noqa: E402
from blast3r.eval.scannetpp import (  # noqa: E402
    cloud_scores, lift_depth_array_to_world, read_exr_depth,
)


def depth_on_grid(path, shape):
    """Nearest-sample a rendered depth map at output-grid pixel centres."""
    if not path.is_file():
        return np.zeros(shape, np.float32)
    depth = read_exr_depth(path)
    rows = ((np.arange(shape[0]) + 0.5) * depth.shape[0] / shape[0]).astype(int)
    columns = ((np.arange(shape[1]) + 0.5) * depth.shape[1] / shape[1]).astype(int)
    sampled = depth[np.ix_(rows, columns)]
    return np.where(np.isfinite(sampled) & (sampled > 0), sampled, 0).astype(np.float32)


def intrinsics_on_grid(transforms, shape):
    scale_x, scale_y = shape[1] / transforms["w"], shape[0] / transforms["h"]
    return np.array([
        [transforms["fl_x"] * scale_x, 0, transforms["cx"] * scale_x - 0.5],
        [0, transforms["fl_y"] * scale_y, transforms["cy"] * scale_y - 0.5],
        [0, 0, 1],
    ], dtype=np.float64)


def evaluate(scene, dataset_root, prepared_scene, prediction_path, stride, tolerances,
             confidence_threshold):
    started = time.perf_counter()
    gt_poses = read_scene_gt_poses(prepared_scene)
    predictions = read_scene_predictions(prediction_path)
    if set(predictions) != set(gt_poses):
        missing = sorted(set(gt_poses) - set(predictions))
        extra = sorted(set(predictions) - set(gt_poses))
        raise ValueError(f"Prediction/GT image mismatch; missing={missing[:3]}, extra={extra[:3]}")
    names = list(gt_poses)
    without_depth = [name for name in names if "depth" not in predictions[name]]
    if without_depth:
        raise ValueError(f"Prediction has no depth for {without_depth[:3]}; use --export_mode full")

    gt = torch.stack([torch.as_tensor(gt_poses[name]).float() for name in names])
    pred = torch.stack([torch.as_tensor(predictions[name]["cam2w"]).float() for name in names])
    pose_metrics = compute_pose_errors(gt, pred)

    pred_centres = np.stack([np.asarray(predictions[name]["cam2w"])[:3, 3] for name in names])
    gt_centres = np.stack([np.asarray(gt_poses[name])[:3, 3] for name in names])
    scale, rotation, translation = estimate_similarity(pred_centres, gt_centres)

    transforms_path = dataset_root / scene / "dslr/nerfstudio/transforms_undistorted.json"
    transforms = json.loads(transforms_path.read_text())
    depth_dir = dataset_root / scene / "rendered_depth"
    predicted_parts, gt_parts, images_with_depth = [], [], 0
    for name in names:
        view = predictions[name]
        depth = np.asarray(view["depth"], dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"{name}: expected 2D depth, got {depth.shape}")
        rendered = depth_on_grid(depth_dir / f"{name}.exr", depth.shape)
        gt_valid = rendered > 0
        images_with_depth += int(gt_valid.any())

        gt_points, _, _ = lift_depth_array_to_world(
            rendered, gt_valid, intrinsics_on_grid(transforms, depth.shape), gt_poses[name],
            stride=stride, max_depth=float("inf"))
        gt_parts.append(gt_points)

        valid = gt_valid & np.isfinite(depth) & (depth > 0)
        if confidence_threshold is not None:
            if "conf" not in view:
                raise ValueError(f"{name}: confidence threshold requested but confidence is absent")
            valid &= np.asarray(view["conf"]) >= confidence_threshold
        aligned_c2w = np.eye(4)
        predicted_c2w = np.asarray(view["cam2w"])
        aligned_c2w[:3, :3] = rotation @ predicted_c2w[:3, :3]
        aligned_c2w[:3, 3] = scale * rotation @ predicted_c2w[:3, 3] + translation
        predicted_points, _, _ = lift_depth_array_to_world(
            depth * scale, valid, np.asarray(view["K"]), aligned_c2w,
            stride=stride, max_depth=float("inf"))
        predicted_parts.append(predicted_points)

    gt_points = np.concatenate(gt_parts)
    predicted_points = np.concatenate(predicted_parts)
    if not len(gt_points):
        raise ValueError("No selected image has valid rendered depth")
    scores = cloud_scores(predicted_points, gt_points, tolerances)
    return {
        "scene": scene,
        "prediction": str(prediction_path),
        "image_count": len(names),
        "images_with_rendered_depth": images_with_depth,
        "pose": {key: float(value) if np.isscalar(value) else value
                 for key, value in pose_metrics.items()},
        "cloud": {
            "point_count": len(predicted_points), "gt_point_count": len(gt_points),
            "tolerances_m": list(tolerances), **scores,
        },
        "sim3_scale": float(scale),
        "point_stride": stride,
        "confidence_threshold": confidence_threshold,
        "evaluation_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dataset-root", type=Path,
                        default=ROOT / "data/scannet++/data_download/scannetpp/data")
    parser.add_argument("--prepared-scene", type=Path, required=True,
                        help="Prepared scene directory containing gt.pkl")
    parser.add_argument("--prediction", type=Path, required=True,
                        help="Native BLASt3R <scene>.pth prediction")
    parser.add_argument("--output", type=Path,
                        help="Default: <prediction stem>_scannetpp_metrics.json")
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--tolerances", type=float, nargs="+",
                        default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    args = parser.parse_args()
    if args.point_stride < 1:
        parser.error("--point-stride must be positive")
    if not args.prediction.is_file() or not (args.prepared_scene / "gt.pkl").is_file():
        parser.error("Missing prediction or prepared gt.pkl")
    transforms = args.dataset_root / args.scene / "dslr/nerfstudio/transforms_undistorted.json"
    if not transforms.is_file() or not (args.dataset_root / args.scene / "rendered_depth").is_dir():
        parser.error(f"Incomplete ScanNet++ scene under {args.dataset_root / args.scene}")

    result = evaluate(args.scene, args.dataset_root, args.prepared_scene, args.prediction,
                      args.point_stride, args.tolerances, args.confidence_threshold)
    output = args.output or args.prediction.with_name(
        args.prediction.stem + "_scannetpp_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    auc5 = result["pose"]["Auc_5"]
    f1 = " ".join(f"{100 * value:.1f}" for value in result["cloud"]["f1_score"])
    print(f"{args.scene}: AUC@5 {auc5:.2f}; cloud F1 [{f1}]")
    print(f"Metrics: {output}")


if __name__ == "__main__":
    main()
