"""Reusable image preprocessing with explicit input and preprocessing identities."""
import tempfile
from pathlib import Path

import numpy as np
import torch

from blast3r.utils.device import stack, todevice


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def image_identity(images):
    return [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in map(Path, images)]


def load_cached_views(directory, images, size=224, force_ar=None, patch_size=16,
                      with_ij_grid=False, return_transform=False):
    path = Path(directory) / 'preprocessed.pth'
    if not path.is_file() or with_ij_grid or return_transform:
        return None
    cached = torch.load(path, map_location='cpu', weights_only=True)
    expected = {'version': 1, 'size': size, 'force_ar': force_ar, 'patch_size': patch_size,
                'images': image_identity(images)}
    if cached['identity'] != expected:
        return None
    views = cached['views']
    views['instance'] = [str(p) for p in images]
    print(f'>> Loaded cached preprocessing for {len(images)} images from {path}')
    return views


def save_cached_views(directory, images, size=512, force_ar=None):
    from blast3r.image import load_images
    views = load_cached_views(directory, images, size=size, force_ar=force_ar)
    if views is not None:
        return views
    frames = load_images(images, size=size, force_ar=force_ar)
    # Only tensors and Python primitives: readable with weights_only=True.
    views = {key: todevice(stack([frame[key] for frame in frames]), 'cpu')
             for key in frames[0] if key != 'instance'}
    def primitives(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, list):
            return [primitives(item) for item in value]
        return value
    views = {key: primitives(value) for key, value in views.items()}
    views['instance'] = [str(p) for p in images]
    atomic_torch_save({'identity': {'version': 1, 'size': size, 'force_ar': force_ar,
                                   'patch_size': 16, 'images': image_identity(images)},
                       'views': views}, Path(directory) / 'preprocessed.pth')
    return views
