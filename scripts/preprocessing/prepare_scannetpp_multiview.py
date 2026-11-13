#!/usr/bin/env python3
"""Prepare sampled ScanNet++ scenes for native BLASt3R offline evaluation.

Each output scene contains image symlinks and ``gt.pkl``, the same layout used
by the repository's other offline benchmarks. No model inference is imported or
run by this script.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dataset-root", type=Path,
                        default=root / "data/scannet++/data_download/scannetpp/data")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output scene directory, e.g. out/scannetpp/prepared/<scene>")
    parser.add_argument("--max-images", type=int, default=150)
    parser.add_argument("--include-bad", action="store_true")
    args = parser.parse_args()

    dslr = args.dataset_root / args.scene / "dslr"
    image_dir = dslr / "resized_undistorted_images"
    transforms_path = dslr / "nerfstudio/transforms_undistorted.json"
    if not image_dir.is_dir() or not transforms_path.is_file():
        parser.error(f"Missing DSLR images or transforms under {dslr}")
    if args.max_images < 2:
        parser.error("--max-images must be at least 2")

    transforms = json.loads(transforms_path.read_text())
    frames = transforms["frames"] + transforms.get("test_frames", [])
    frames = sorted((frame for frame in frames
                     if args.include_bad or not frame.get("is_bad", False)),
                    key=lambda frame: frame["file_path"])
    indices = np.linspace(0, len(frames) - 1, min(args.max_images, len(frames)), dtype=int)
    selected = [frames[index] for index in indices]
    sources = [(image_dir / frame["file_path"]).resolve() for frame in selected]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        parser.error(f"Missing selected images, e.g. {missing[:3]}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    names = [frame["file_path"] for frame in selected]
    expected = set(names) | {"gt.pkl", "inputs.json"}
    unexpected = {path.name for path in output.iterdir()} - expected
    if unexpected:
        parser.error(f"Output contains unexpected files: {sorted(unexpected)[:3]}")

    poses = []
    for frame, source in zip(selected, sources):
        target = output / frame["file_path"]
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                parser.error(f"Existing link points elsewhere: {target}")
        else:
            target.symlink_to(source)
        camera_to_world = np.asarray(frame["transform_matrix"], dtype=np.float64).copy()
        camera_to_world[:3, 1:3] *= -1  # OpenGL to OpenCV camera axes
        poses.append(camera_to_world.astype(np.float32))

    with (output / "gt.pkl").open("wb") as stream:
        pickle.dump({"filenames": names, "poses": poses}, stream)
    manifest = {
        "scene": args.scene,
        "dataset_root": str(args.dataset_root.resolve()),
        "max_images": args.max_images,
        "include_bad": args.include_bad,
        "names": names,
    }
    (output / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(selected)} images in {output}")


if __name__ == "__main__":
    main()
