#!/usr/bin/env python3
"""Freeze an ETH3D subset and warm image, VGGT-Omega and PCA-15 caches."""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from preprocessing.prepare_eth3d import ETH3D_SCENES, read_scene
from blast3r.inference.loading import load_depth_model, sample_scene_items
from blast3r.inference.multidepth import compute_pts3d_multidepths_vggt, vggt_cache_file
from blast3r.inference.scene_cache import save_cached_views


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_subset(source, output, max_frames=10, seed=0):
    paths, poses = read_scene(source)
    scene = Path(source).name
    selected = sample_scene_items(paths, scene, max_frames=max_frames, seed=seed)
    if len(selected) != max_frames:
        raise ValueError(f'{scene}: requested {max_frames} images, found only {len(selected)}')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entries = [{'name': path.name, 'source': str(path.resolve()), 'sha256': sha256(path)}
               for path in selected]
    manifest = {'version': 1, 'scene': scene, 'seed': seed, 'max_frames': max_frames,
                'sampling': 'blast3r.sample_scene_items', 'images': entries}
    manifest_path = output / 'selection.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        # Moving the source dataset does not change a frozen selection.
        def identity(value):
            return (value['version'], value['scene'], value['seed'], value['max_frames'],
                    [(entry['name'], entry['sha256']) for entry in value['images']])
        if identity(existing) != identity(manifest):
            raise ValueError(f'Subset changed at {output}; choose a new output directory')
    from blast3r.image import IMAGE_SUFFIXES
    extra = [path for path in output.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
             and path.name not in {entry['name'] for entry in entries}]
    if extra:
        raise ValueError(f'Unexpected images in frozen subset: {extra}')
    for path, entry in zip(selected, entries):
        destination = output / path.name
        if destination.exists():
            if sha256(destination) != entry['sha256']:
                raise ValueError(f'Cached image changed: {destination}')
        else:
            with tempfile.NamedTemporaryFile(dir=output, suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
            try:
                shutil.copy2(path, temporary)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    pose_by_path = dict(zip(paths, poses))
    gt = {'filenames': [path.name for path in selected],
          'poses': [np.asarray(pose_by_path[path], dtype=np.float32) for path in selected]}
    atomic_bytes(output / 'gt.pkl', pickle.dumps(gt))
    atomic_bytes(manifest_path, (json.dumps(manifest, indent=2) + '\n').encode())
    return [output / path.name for path in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eth3d_root', required=True)
    parser.add_argument('--out', required=True, help='frozen scene directories with images and GT')
    parser.add_argument('--ckpt_depth', required=True)
    parser.add_argument('--cache_root', default=os.environ.get('BLAST3R_CACHE_ROOT', '.cache'))
    parser.add_argument('--max_frames', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--force_ar', type=float, default=None)
    parser.add_argument('--scenes', nargs='+', default=ETH3D_SCENES)
    parser.add_argument('--prepare_only', action='store_true', help='skip VGGT inference')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error('--max_frames must be positive')
    if len(args.scenes) != len(set(args.scenes)) or any(Path(s).name != s for s in args.scenes):
        parser.error('--scenes must contain unique scene directory names')
    torch.set_num_threads(min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))))
    prepared = []
    for scene in args.scenes:
        output = Path(args.out) / scene
        images = prepare_subset(Path(args.eth3d_root) / scene, output, args.max_frames, args.seed)
        # Source camera calibration and GT poses are deliberately absent from views.
        save_cached_views(output, images, size=args.size, force_ar=args.force_ar)
        prepared.append((scene, output, images))
        print(f'{scene}: froze {len(images)} images and cached preprocessing', flush=True)
    if args.prepare_only:
        return
    model = load_depth_model(args.ckpt_depth, device=args.device, backend='vggt')
    for scene, output, images in prepared:
        views = save_cached_views(output, images, size=args.size, force_ar=args.force_ar)
        conf, pts3d, basis, depth_mode = compute_pts3d_multidepths_vggt(
            model, views, refine_head=True, cache_dir=Path(args.cache_root) / 'vggt_monodepth')
        if len(basis) != args.max_frames:
            raise RuntimeError(f'Incomplete VGGT cache for {scene}')
        index = {'version': 1, 'checkpoint_identity': model.cache_identity,
                 'size': args.size, 'force_ar': args.force_ar, 'images': []}
        for i, image in enumerate(images):
            cache_file = vggt_cache_file(Path(args.cache_root) / 'vggt_monodepth',
                                         model.cache_identity, views['img'][i], views['true_shape'][i])
            index['images'].append({'name': image.name,
                                    'cache_file': os.path.relpath(cache_file, output)})
        atomic_bytes(output / 'vggt_cache.json', (json.dumps(index, indent=2) + '\n').encode())
        print(f'{scene}: VGGT features, depth, confidence, intrinsics and PCA-15 ready', flush=True)
        del views, conf, pts3d, basis
    print(f'Cache complete: {len(prepared)} scenes x {args.max_frames} images at {args.out}', flush=True)


if __name__ == '__main__':
    main()
