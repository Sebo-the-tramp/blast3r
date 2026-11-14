#!/usr/bin/env python3
"""Convert per-scene ScanNet++ JSON metrics into the standard run CSVs."""

import argparse
import json
from pathlib import Path

import pandas as pd


def tolerance_label(value):
    centimetres = 100 * float(value)
    return f"{centimetres:g}cm"


def aggregate_seed(seed_dir):
    prediction_paths = sorted(seed_dir.glob("*.pth"))
    if not prediction_paths:
        raise ValueError(f"No predictions found in {seed_dir}")

    pose_rows, depth_rows = [], []
    for prediction in prediction_paths:
        scene = prediction.stem
        metrics_path = seed_dir / f"{scene}_scannetpp_metrics.json"
        if not metrics_path.is_file():
            raise ValueError(f"Missing evaluation for {prediction}: {metrics_path}")
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("scene") != scene:
            raise ValueError(f"Scene mismatch in {metrics_path}")

        pose_rows.append({"scene": scene, "status": "ok", **metrics["pose"]})
        cloud = metrics["cloud"]
        depth_row = {
            "scene": scene,
            "point_count": cloud["point_count"],
            "gt_point_count": cloud["gt_point_count"],
        }
        for tolerance, accuracy, completeness, f1 in zip(
                cloud["tolerances_m"], cloud["accuracy"],
                cloud["completeness"], cloud["f1_score"]):
            label = tolerance_label(tolerance)
            depth_row[f"accuracy_{label}"] = 100 * accuracy
            depth_row[f"completeness_{label}"] = 100 * completeness
            depth_row[f"f1_{label}"] = 100 * f1
        depth_rows.append(depth_row)

    pose = pd.DataFrame(pose_rows).set_index("scene")
    depth = pd.DataFrame(depth_rows).set_index("scene").astype(float)
    pose.loc["AVG", pose.select_dtypes("number").columns] = pose.mean(numeric_only=True)
    pose.loc["AVG", "status"] = "ok"
    depth.loc["AVG"] = depth.mean(numeric_only=True)
    pose.to_csv(seed_dir / "relpose.csv", float_format="%.6g")
    depth.to_csv(seed_dir / "depth.csv", float_format="%.6g")
    return len(prediction_paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("out/scannetpp/ablation"))
    parser.add_argument("--expected-scenes", type=int, default=20)
    args = parser.parse_args()

    seed_dirs = sorted(path for path in args.runs_root.glob("*/seed_*") if path.is_dir())
    if not seed_dirs:
        parser.error(f"No <method>/seed_* directories under {args.runs_root}")
    for seed_dir in seed_dirs:
        count = aggregate_seed(seed_dir)
        if args.expected_scenes and count != args.expected_scenes:
            raise ValueError(f"Expected {args.expected_scenes} scenes in {seed_dir}, found {count}")
        print(f"{seed_dir}: {count} scenes -> relpose.csv, depth.csv")


if __name__ == "__main__":
    main()
