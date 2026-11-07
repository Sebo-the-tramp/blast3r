#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Prepare the ETH3D multi-view benchmark for offline evaluation.

    python scripts/preprocessing/prepare_eth3d.py --eth3d_root DIR --out DIR

`--eth3d_root` is the official ETH3D high-resolution multi-view download, one
directory per scene, each holding `dslr_calibration_undistorted/` and
`images/dslr_images_undistorted/`. Writes one directory per scene, its images
symlinked and its ground-truth poses in `gt.pkl`. Also accepts processed scenes
with `undistorted_images/` and camera-from-world `poses/*.npy`.
"""
import argparse

import numpy as np
from pathlib import Path

from blast3r.eval.colmap import read_images as read_colmap_images
from blast3r.eval.datasets import colmap_pose
from blast3r.eval.prepare import write_scene

# The 13 scenes with public ground truth, as used for the published results.
ETH3D_SCENES = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'meadow',
                'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']

CALIBRATION = 'dslr_calibration_undistorted'
IMAGES = 'images'


def read_scene(scene_dir):
    """Return source images and camera-to-world GT, for either ETH3D layout."""
    scene_dir = Path(scene_dir)
    if (scene_dir / 'undistorted_images').is_dir():
        from blast3r.image import IMAGE_SUFFIXES
        paths = sorted(p for p in (scene_dir / 'undistorted_images').iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        poses = []
        for path in paths:
            world_to_camera = np.load(scene_dir / 'poses' / f'{path.stem}.npy', allow_pickle=False)
            if (world_to_camera.shape != (4, 4) or not np.isfinite(world_to_camera).all()
                    or not np.allclose(world_to_camera[3], [0, 0, 0, 1])):
                raise ValueError(f'Invalid world-to-camera pose for {path}')
            poses.append(np.linalg.inv(world_to_camera))
    else:
        images = read_colmap_images(scene_dir / CALIBRATION)
        names = sorted(images)
        paths = [scene_dir / IMAGES / name for name in names]
        poses = [colmap_pose(*images[name]) for name in names]
    if not paths:
        raise ValueError(f'No images found in {scene_dir}')
    if len({path.name for path in paths}) != len(paths):
        raise ValueError(f'Duplicate image basenames in {scene_dir}')
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} images are missing, e.g. {missing[0]}')
    return paths, poses


def prepare_scene(scene_dir, out_dir):
    paths, poses = read_scene(scene_dir)
    write_scene(out_dir, paths, poses)
    return len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eth3d_root', required=True, help='the official ETH3D download')
    parser.add_argument('--out', required=True, help='where to write the prepared scenes')
    parser.add_argument('--scenes', nargs='+', default=ETH3D_SCENES,
                        help='scenes to prepare (default: the 13 published scenes)')
    args = parser.parse_args()

    for scene in args.scenes:
        scene_dir = Path(args.eth3d_root) / scene
        assert scene_dir.is_dir(), f'no such scene directory: {scene_dir}'
        n_images = prepare_scene(scene_dir, Path(args.out) / scene)
        print(f'{scene}: {n_images} images')

    print(f'\nPrepared {len(args.scenes)} scenes in {args.out}')


if __name__ == '__main__':
    main()
