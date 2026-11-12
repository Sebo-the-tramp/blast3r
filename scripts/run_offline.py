#!/usr/bin/env python3
# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Offline reconstruction of an image collection.

    python scripts/run_offline.py --images DIR --output DIR --mono_cam

Defaults are the values used for the published results.
"""
import hashlib
import json
from pathlib import Path

import torch

from blast3r.inference.args import base_parser
from blast3r.inference.loading import load_dataset, load_depth_model, load_match_model
from blast3r.inference.pipeline import run_inference
from blast3r.inference.results import prepare_results
from blast3r.inference.timing import timer

OFFLINE_CACHE_FORMAT = 1
DEPTH_CACHE_SCHEMA = 1


def arg_parser():
    parser = base_parser('BLASt3R offline reconstruction')
    parser.add_argument('--depth_backend', choices=('blast3r', 'vggt_omega'), default='blast3r',
                        help='model used for depth and focal initialization')
    parser.add_argument('--use_cache', action='store_true',
                        help='reuse shared depths and seed-specific pre-BA initialization')
    parser.add_argument('--cache_dir', default='.cache/blast3r/offline',
                        help='offline cache root (used only with --use_cache)')
    return parser


def _image_fingerprints(views):
    fingerprints = []
    for instance in views.get('instance', []):
        path = Path(str(instance)).resolve()
        stat = path.stat()
        fingerprints.append({
            'path': str(path),
            'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
        })
    return fingerprints


def _cache_file(cache_dir, kind, scene, metadata, seed=None):
    encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    root = Path(cache_dir) / kind
    if seed is not None:
        root /= f'seed_{seed}'
    return root / str(scene) / f'{digest}.pth'


def _cache_config(views, scene, seed, ckpt_depth, ckpt_match, size, force_ar,
                  depth_backend,
                  device, dtype, options, cache_dir):
    depth_metadata = {
        'format': OFFLINE_CACHE_FORMAT,
        # Cache the normalized depth representation consumed by the pipeline,
        # not backend-private model outputs. Backends use separate namespaces.
        'schema': DEPTH_CACHE_SCHEMA,
        'backend': depth_backend,
        'scene': str(scene),
        'images': _image_fingerprints(views),
        'ckpt_depth': str(ckpt_depth),
        'size': size,
        'force_ar': force_ar,
        'device': str(device),
        'dtype': str(dtype),
    }
    if depth_backend == 'vggt_omega':
        depth_metadata.update({
            'backend_data_schema': 1,
            'feature_name': 'dense_head_fused_pre_projection',
            'feature_dtype': 'float16',
            'feature_stride': 4,
        })
    init_metadata = {
        'format': OFFLINE_CACHE_FORMAT,
        'depth': depth_metadata,
        'ckpt_match': str(ckpt_match),
        'seed': seed,
        'mono_cam': options.get('mono_cam', False),
        'kpt_spacing': options.get('kpt_spacing', 8),
        'remove_far_kpts': options.get('remove_far_kpts', True),
        'retrieval_mode': options.get('retrieval_mode', 'coreset_fps_30'),
        'num_retrieved_images': options.get('num_retrieved_images', 40),
        'optim_memory': options.get('optim_memory', False),
        # Initialization only distinguishes whether K/P may ever be optimized;
        # later per-phase schedules are BA settings and reuse the same cache.
        'optim_K': any(options.get('optim_K', [True])),
        'optim_P': any(options.get('optim_P', [True])),
    }
    return ({
        'depth': _cache_file(cache_dir, f'depth/{depth_backend}', scene, depth_metadata),
        'init': _cache_file(cache_dir, 'init', scene, init_metadata, seed=seed),
    }, {
        'depth': depth_metadata,
        'init': init_metadata,
    })


def run(images, ckpt_depth, ckpt_match, size, force_ar, output, export_mode, name=None,
        skip_existing=False, skip_failures=False, seed=0, device='cuda', dtype=torch.float32,
        use_cache=False, cache_dir='.cache/blast3r/offline', depth_backend='blast3r',
        **options):
    dataset = load_dataset(images, size=size, force_ar=force_ar, name=name)
    depther = load_depth_model(ckpt_depth, device=device, dtype=dtype, backend=depth_backend)
    matcher = load_match_model(ckpt_match, device=device, dtype=dtype)

    for i, views in enumerate(dataset):
        scene = views['label'][0] if 'label' in views else None

        filename = f'{scene}.pth' if scene is not None else f'result_{i:03d}.pth'
        out_file = output if output.endswith('.pth') else str(Path(output) / filename)
        if Path(out_file).exists() and skip_existing:
            continue
        # fail before reconstructing rather than after
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)

        print(f'Processing scene #{i+1}')
        # reset per scene so the coreset draw does not depend on the position in the batch
        torch.manual_seed(seed)
        cache_paths = cache_metadata = None
        if use_cache:
            cache_paths, cache_metadata = _cache_config(
                views, scene or filename, seed, ckpt_depth, ckpt_match, size, force_ar,
                depth_backend,
                device, dtype, options, cache_dir)
        try:
            # release the previous scene's buffers before sizing this one
            torch.cuda.empty_cache()
            timer.reset()
            with timer.time('total'):
                out = run_inference(views, depther, matcher, **options, device=device,
                                    cache_paths=cache_paths, cache_metadata=cache_metadata)
            timer_out = timer.summary()
        except Exception as e:
            if not skip_failures:
                raise
            print(f'Error processing scene #{i+1}: {e}')
            continue

        img_names = [Path(inst).name for inst in views['instance']]
        out = prepare_results(out, img_names, extra_data={'timing': timer_out}, export_mode=export_mode)
        torch.save(out, out_file)


if __name__ == '__main__':
    run(**vars(arg_parser().parse_args()))
