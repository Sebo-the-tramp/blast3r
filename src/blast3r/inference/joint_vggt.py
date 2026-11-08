"""Joint VGGT-Omega predictions; deliberately separate from monocular caches."""
import hashlib
from pathlib import Path

import torch

from blast3r.image import detokenize
from blast3r.utils.geometry import depthmap_to_pts3d
from .multidepth import vggt_pca_head_basis
from .scene_cache import atomic_torch_save


def joint_input_identity(views, checkpoint_identity):
    digest = hashlib.sha256(f'joint-vggt-pca15-v1:{checkpoint_identity}'.encode())
    for patches, shape in zip(views['img'], views['true_shape']):
        digest.update(str(tuple(map(int, shape))).encode())
        digest.update(patches.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def compute_joint_vggt(model, views, cache_file):
    """One sequence forward, retaining jointly predicted poses, depth and features."""
    identity = joint_input_identity(views, model.cache_identity)
    cache_file = Path(cache_file)
    if cache_file.is_file():
        cached = torch.load(cache_file, map_location='cpu', weights_only=True)
        if cached['identity'] == identity:
            print(f'>> Joint VGGT cache hit: {cache_file}', flush=True)
            return cached
    shapes = [tuple(map(int, shape)) for shape in views['true_shape']]
    if len(set(shapes)) != 1:
        raise ValueError('Joint VGGT requires equal image grids; use --force_ar to select a common crop')
    device = model.inference_device
    images = torch.stack([detokenize(p.float(), shape, chan_dim=-3)
                          for p, shape in zip(views['img'], shapes)])
    images = ((images + 1) / 2).to(device)
    model.to(device)
    captured = []
    def capture(module, inputs):
        captured.append(inputs[0].detach().float().cpu().clone())
    handle = model.dense_head.proj.register_forward_pre_hook(capture)
    try:
        prediction = model(images[None])  # B=1, sequence=all ten images
    finally:
        handle.remove()
    from vggt_omega.utils.pose_enc import encoding_to_camera
    extrinsics, K = encoding_to_camera(prediction['pose_enc'].float(), shapes[0])
    K = K[0].cpu()
    w2c = torch.eye(4).repeat(len(images), 1, 1)
    w2c[:, :3] = extrinsics[0].cpu()
    depth = prediction['depth'][0, ..., 0].float().cpu()
    conf = prediction['depth_conf'][0].float().cpu()
    # DenseHead decodes sequences in chunks of eight frames.
    features = torch.cat(captured, dim=0)
    if depth.shape != (len(images), *shapes[0]) or features.shape[0] != len(images):
        raise ValueError(f'Unexpected joint VGGT geometry: depth={depth.shape}, features={features.shape}')
    if not (torch.isfinite(depth) & (depth > 0)).all():
        raise ValueError('Joint VGGT returned invalid depth')
    if not torch.isfinite(w2c).all() or not torch.isfinite(K).all() or not torch.isfinite(features).all():
        raise ValueError('Joint VGGT returned invalid cameras or features')
    bases = [torch.stack((d.log(), torch.ones_like(d)), -1) for d in depth]
    result = {'identity': identity, 'inference': 'joint', 'w2c': w2c, 'K': K,
              'conf': list(conf), 'projection_features': features,
              'pts3d': [depthmap_to_pts3d(d, k) for d, k in zip(depth, K)],
              'multidepth': bases,
              'pca_basis': [vggt_pca_head_basis(base, feat) for base, feat in zip(bases, features)]}
    atomic_torch_save(result, cache_file)
    return result
