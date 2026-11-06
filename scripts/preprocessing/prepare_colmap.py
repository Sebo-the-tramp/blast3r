#!/usr/bin/env python3
"""Prepare one COLMAP-reference scene for eval_offline.py.

Supports Mip-NeRF 360's sparse/0 model and images_2 directory, among others.
COLMAP poses are reference estimates, not independent depth ground truth.
"""
import argparse
from pathlib import Path

from blast3r.eval.colmap import read_images
from blast3r.eval.datasets import colmap_pose
from blast3r.eval.prepare import write_scene


def prepare_scene(model_dir, image_dir, out_dir):
    references = read_images(model_dir)
    if len(references) < 2:
        raise ValueError('At least two registered images are required')
    names = sorted(references)
    paths = [Path(image_dir) / name for name in names]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} registered images are missing: {missing[:3]}')
    if len({p.name for p in paths}) != len(paths):
        raise ValueError('Duplicate image basenames cannot be represented by the evaluator')
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to replace an existing prepared scene: {out}')
    write_scene(out, paths, [colmap_pose(*references[name]) for name in names])
    return len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='directory containing images.bin or images.txt')
    parser.add_argument('--images', required=True, help='image root corresponding to COLMAP filenames')
    parser.add_argument('--out', required=True, help='prepared scene directory, e.g. outputs/gt/bicycle')
    args = parser.parse_args()
    count = prepare_scene(args.model, args.images, args.out)
    print(f'Prepared {count} registered images in {args.out}')


if __name__ == '__main__':
    main()
