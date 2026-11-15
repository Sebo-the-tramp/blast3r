#!/usr/bin/env python3
"""Prepare ETH3D from the preprocessed Berzelius layout for offline evaluation.

    python scripts/preprocessing/prepare_eth3d_processed.py \
        --eth3d_root /proj/vlarsson/datasets/eth3d-eval/eth3d_processed --out data/eth3d

`--eth3d_root` holds one directory per scene with `undistorted_images/*.jpg`
and `poses/*.npy` (4x4 world-to-camera, in the scan frame). Writes the same
layout as prepare_eth3d.py: symlinked images plus `gt.pkl` with camera-to-world
poses. The scan GT for point-cloud evaluation stays in
`<eth3d_root>/<scene>/dslr_scan_eval`, so pass `--eth3d_scans_dir <eth3d_root>`.
"""
import argparse
from pathlib import Path

import numpy as np

from blast3r.eval.prepare import write_scene

ETH3D_SCENES = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'meadow',
                'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']


def prepare_scene(scene_dir, out_dir, image_dir, image_root=None):
    """Poses always come from `<scene_dir>/poses`. With `image_root`, images are
    taken from `<image_root>/<scene>/images/<stem>.JPG` instead (e.g. the official
    full-resolution undistorted DSLR images), matched by file stem."""
    stems = sorted(p.stem for p in (scene_dir / 'poses').glob('*.npy'))
    assert stems, f'no poses under {scene_dir / "poses"}'
    if image_root is None:
        images = [scene_dir / image_dir / f'{stem}.jpg' for stem in stems]
    else:
        images = [Path(image_root) / scene_dir.name / 'images' / f'{stem}.JPG' for stem in stems]
    missing = [p for p in images if not p.is_file()]
    assert not missing, f'{len(missing)} images missing, e.g. {missing[0]}'
    poses = []
    for stem in stems:
        w2c = np.load(scene_dir / 'poses' / f'{stem}.npy').astype(np.float64)
        poses.append(np.linalg.inv(w2c))
    write_scene(out_dir, images, poses)
    return len(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eth3d_root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--image_dir', default='undistorted_images')
    parser.add_argument('--image_root', default=None,
                        help='take full-resolution images from <image_root>/<scene>/images/*.JPG, '
                             'e.g. /proj/vlarsson/users/x_jonas/data/eth3d_dfsfm')
    parser.add_argument('--scenes', nargs='+', default=ETH3D_SCENES)
    args = parser.parse_args()

    for scene in args.scenes:
        scene_dir = Path(args.eth3d_root) / scene
        assert (scene_dir / 'dslr_scan_eval' / 'scan_alignment.mlp').is_file(), scene_dir
        n = prepare_scene(scene_dir, Path(args.out) / scene, args.image_dir, args.image_root)
        print(f'{scene}: {n} images')


if __name__ == '__main__':
    main()
