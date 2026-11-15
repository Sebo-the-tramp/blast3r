#!/usr/bin/env python3
"""Prepare AerialMegaDepth image sets for offline BLASt3R evaluation.

    python scripts/preprocessing/prepare_aerialmegadepth.py \
        --subsets_dir /proj/vlarsson/users/x_luddi/sc3r/out/aerialmegadepth_val/subsets \
        --out data/aerialmegadepth

`--subsets_dir` holds the sets written by sc3r's scripts/setup_aerialmegadepth_subsets.py
(`<set>/images/`, `<set>/gt_sparse/` COLMAP text with one camera per image, and
`<set>/depth/<stem>.npy`). Writes one directory per set with symlinked images and `gt.pkl`
(camera-to-world poses), the layout `eval_offline.py` and `eval_aerialmegadepth.py` read.
The GT cameras and depth stay in `--subsets_dir`.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from blast3r.eval.prepare import write_scene

DEFAULT_SETS = [f'{scene}_n150_d{draw}' for scene in ('0015', '0022') for draw in range(4)]


def read_gt_images(path):
    """{name: 4x4 camera-to-world} from a COLMAP images.txt.

    The subsets write one pose line per image followed by an empty POINTS2D line, so pose lines
    are identified by their field count rather than by alternating lines.
    """
    poses = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if line.startswith('#') or len(fields) != 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, fields[1:8])
        w2c = np.eye(4)
        w2c[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c[:3, 3] = (tx, ty, tz)
        poses[fields[9]] = np.linalg.inv(w2c)
    return poses


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subsets_dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--sets', nargs='+', default=DEFAULT_SETS)
    args = parser.parse_args()

    for name in args.sets:
        source = Path(args.subsets_dir) / name
        poses = read_gt_images(source / 'gt_sparse' / 'images.txt')
        images = sorted((source / 'images').iterdir())
        assert len(images) == len(poses), f'{name}: {len(images)} images, {len(poses)} GT poses'
        write_scene(Path(args.out) / name, images, [poses[image.name] for image in images])
        print(f'{name}: {len(images)} images')


if __name__ == '__main__':
    main()
