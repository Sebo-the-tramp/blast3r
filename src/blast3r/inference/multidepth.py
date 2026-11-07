# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import types
import hashlib
import tempfile
from pathlib import Path
from time import perf_counter
from tqdm.auto import tqdm
import torch
from torch import autocast

from blast3r.utils.device import todevice
from blast3r.image import detokenize
from blast3r.utils.geometry import depthmap_to_pts3d


def compute_pts3d_multidepths(model, views, backend='blast3r', refine_head=False):
    match backend:
        case 'blast3r':
            return compute_pts3d_multidepths_blast3r(model, views)
        case 'vggt':
            return compute_pts3d_multidepths_vggt(model, views, refine_head=refine_head)
        case _:
            raise ValueError(f'Unknown depth backend: {backend}')


@torch.inference_mode()
def compute_pts3d_multidepths_blast3r(model, views):
    assert not isinstance(model, types.LambdaType), 'the model must be loaded before reaching here'

    imgs_shape = views['img'].shape
    assert imgs_shape[-3:] == (3, 16, 16), 'wrong format: images are not tokenized'
    if len(imgs_shape) == 4:
        n_imgs = 1
        batching = None
    elif len(imgs_shape)  == 5:
        n_imgs = imgs_shape[0]
        batching = ()
    else:
        raise TypeError(f'imgs have the wrong shape={imgs_shape}, should be a 4- or 5-tuple')

    print(f'>> Extracting depthmaps for {n_imgs} images ...')
    batched_views = {key:views[key][batching] for key in 'img token_pos true_shape'.split()}
    if 'depth' in views: batched_views['depth'] = views['depth'][batching]
    if 'K' in views: batched_views['K'] = views['K'][batching]
    batched_views = todevice(batched_views, model.device)

    preds = []
    with autocast(device_type=model.device.type, dtype=model.force_dtype, enabled=(model.force_dtype != torch.float32)):
        for i in range(n_imgs):
            pred = model({k: v[i:i+1].unsqueeze(1) for k, v in batched_views.items()}, step=0)
            preds.append(pred[0]) # MonoDepth returns one dict per view

    depth_mode = 'log_multidepth'
    pts3d = [pred['query_pts3d'][..., :3].squeeze(0) for pred in preds]
    conf = [pred['query_pts3d_conf'].squeeze(0) for pred in preds]
    multidepth = [pred['query_pts3d_raw'][..., 2:].squeeze(0) for pred in preds]

    # detokenize (per-img because shape can vary)
    H_Ws = views['true_shape'][batching].tolist()
    assert len(H_Ws) == len(preds) and len(H_Ws[0]) == 2, \
        f'expected {len(preds)} (H, W) pairs, got {H_Ws}'
    pts3d = [detokenize(x.float(), H_W) for x, H_W in zip(pts3d, H_Ws)]
    conf = [detokenize(x.float().exp_(), H_W, chan_dim=None) for x, H_W in zip(conf, H_Ws)] # logits -> linear space
    multidepth = [detokenize(x.float(), H_W) for x, H_W in zip(multidepth, H_Ws)]

    if depth_mode == 'log_multidepth':
        multidepth = [torch.cat((m[...,:1], torch.ones_like(m[...,:1]), m[...,1:]), -1) for m in multidepth]
    print('Found', multidepth[0].shape[-1], 'Z components')

    return conf, pts3d, multidepth, depth_mode


@torch.inference_mode()
def compute_pts3d_multidepths_vggt(model, views, refine_head=False):
    """Run VGGT-Omega independently on each preprocessed BLASt3R image.

    Requires a loaded VGGT-Omega model in eval mode with its depth head enabled,
    and either supplied intrinsics or an enabled camera head. Outputs use the
    input pixel grid and a two-channel log-depth basis (log depth, constant).
    """
    from vggt_omega.utils.pose_enc import encoding_to_camera

    images = views['img']
    if images.ndim == 4:
        images = images.unsqueeze(0)
        shapes = views['true_shape'][None]
        intrinsics = views['K'][None] if 'K' in views else None
    elif images.ndim == 5:
        shapes = views['true_shape']
        intrinsics = views.get('K')
    else:
        raise ValueError(f'Expected tokenized images with 4 or 5 dimensions, got {images.shape}')
    if images.shape[-3:] != (3, 16, 16):
        raise ValueError(f'Expected RGB 16x16 patches, got {images.shape}')

    # Read wrapper metadata without triggering lazy checkpoint loading.
    device = vars(model).get('inference_device', images.device)
    identity = vars(model).get('cache_identity')
    if identity is None:
        # Directly supplied models need a weight fingerprint too.
        digest = hashlib.sha256(type(model).__qualname__.encode())
        for name, tensor in model.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        identity = digest.hexdigest()
    cache_dir = Path('.cache') / 'vggt_monodepth'
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_ready = False
    conf, pts3d, multidepth = [], [], []
    print(f'>> Computing monocular depthmaps using VGGT-Omega for {len(images)} images ...')
    progress = tqdm(enumerate(images), total=len(images), desc='VGGT mono depth', unit='frame')
    for i, patches in progress:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = perf_counter()
        H, W = map(int, shapes[i])
        digest = hashlib.sha256(f'v1:{identity}:{H}:{W}'.encode())
        digest.update(patches.detach().float().cpu().contiguous().numpy().tobytes())
        if intrinsics is not None:
            digest.update(torch.as_tensor(intrinsics[i]).float().cpu().contiguous().numpy().tobytes())
        cache_file = cache_dir / f'{digest.hexdigest()}.pth'
        if cache_file.is_file():
            result = torch.load(cache_file, map_location='cpu', weights_only=True)
            if 'projection_features' in result:
                conf.append(result['conf'].to(device))
                pts3d.append(result['pts3d'].to(device))
                multidepth.append(vggt_pca_head_basis(result['multidepth'], result['projection_features'])
                                  if refine_head else result['multidepth'].to(device))
                progress.set_postfix_str(f'cache hit, {perf_counter() - started:.2f} s/frame')
                del result
                continue
        if not model_ready:
            model.to(device)
            model_ready = True
        # Undo patch packing and normalization without resizing or cropping.
        rgb = detokenize(patches.float(), (H, W), chan_dim=-3)
        rgb = ((rgb + 1.0) / 2.0).to(device)
        captured = {}
        def capture_projection_input(module, inputs):
            captured['projection_features'] = inputs[0].detach().float().cpu().clone()
        projection_handle = model.dense_head.proj.register_forward_pre_hook(capture_projection_input)
        try:
            pred = model(rgb[None, None])  # batch=1, sequence=1: monocular inference
        finally:
            projection_handle.remove()

        depth = pred['depth'][0, 0, ..., 0].float()
        confidence = pred['depth_conf'][0, 0].float()  # already activated
        if depth.shape != (H, W) or confidence.shape != (H, W):
            raise ValueError('VGGT-Omega outputs must match the input image pixel grid')
        if not (torch.isfinite(depth) & (depth > 0)).all():
            raise ValueError('VGGT-Omega returned non-finite or non-positive depth')

        if intrinsics is not None:
            K = torch.as_tensor(intrinsics[i], device=device, dtype=torch.float32)
        else:
            if 'pose_enc' not in pred:
                raise ValueError('VGGT-Omega needs its camera head enabled when views have no K')
            _, K = encoding_to_camera(pred['pose_enc'].float(), (H, W))
            K = K[0, 0]

        pts3d.append(depthmap_to_pts3d(depth, K))
        conf.append(confidence)
        multidepth.append(torch.stack((depth.log(), torch.ones_like(depth)), dim=-1))
        result = {
            'conf': conf[-1].cpu(),
            'pts3d': pts3d[-1].cpu(),
            'multidepth': multidepth[-1].cpu(),
            'depth_mode': 'log_multidepth',
            'intrinsics': K.cpu(),
            'projection_features': captured['projection_features'][0].contiguous(),
            'metadata': {'checkpoint': identity, 'processed_size_hw': (H, W),
                         'projection_features_layer': 'dense_head.proj.input'},
        }
        # Atomic replacement avoids leaving a partial cache after interruption.
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix='.tmp', delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            torch.save(result, temporary)
            temporary.replace(cache_file)
        finally:
            temporary.unlink(missing_ok=True)
        if refine_head:
            multidepth[-1] = vggt_pca_head_basis(result['multidepth'], result['projection_features'])
        del result, captured
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        progress.set_postfix_str(f'{perf_counter() - started:.2f} s/frame')
        del pred, rgb

    return conf, pts3d, multidepth, 'log_multidepth'


def vggt_pca_head_basis(base_multidepth, projection_features, eigenvalue_floor=1e-6, pca_dim=15):
    """Compact CPU base log-depth and pca quarter-resolution whitened PCA maps.

    Fit PCA over spatial locations, with the 256 projection-input channels as
    features. The maps are bilinearly sampled at track observations, producing a smooth
    correction rather than the original pixel-shuffled blockwise correction.
    Retained maps have zero mean and unit population variance. Directions below
    eigenvalue_floor times the largest variance are discarded, then zero-padded
    to 15 channels to preserve the solver's 17-channel layout. No target depths
    or model training are used to fit PCA.
    """
    if not 0 < eigenvalue_floor < 1:
        raise ValueError('eigenvalue_floor must lie strictly between zero and one')
    base = base_multidepth.detach().float().cpu()
    features = projection_features.detach().float().cpu()
    H, W, C = base.shape
    if C != 2 or features.shape != (256, H // 4, W // 4) or H % 4 or W % 4:
        raise ValueError('PCA head requires (H,W,2) base and (256,H/4,W/4) projection features')
    if not torch.isfinite(features).all():
        raise ValueError('Projection features must be finite for PCA')
    samples = features.flatten(1).T.double()
    centered = samples - samples.mean(0, keepdim=True)
    # A small deterministic 256x256 eigensolve, also valid for rank-deficient inputs.
    eigenvalues, vectors = torch.linalg.eigh(centered.T @ centered / len(samples))
    components = vectors[:, -pca_dim:].flip(1)
    variances = eigenvalues[-pca_dim:].flip(0)
    keep = variances > eigenvalue_floor * eigenvalues[-1].clamp_min(0)
    # Fix eigenvector signs so repeated cache reads produce consistent bases.
    pivots = components.abs().argmax(dim=0)
    components *= components[pivots, torch.arange(pca_dim)].sign()[None]
    scores = centered.new_zeros((len(samples), pca_dim))
    # Never divide discarded directions by a near-zero (or negative) eigenvalue.
    scores[:, keep] = (centered @ components[:, keep]) / variances[keep].sqrt()
    scores = scores.float().reshape(H // 4, W // 4, pca_dim)
    return {'base_log_depth': base[..., 0].clone(),
            'features': scores.permute(2, 0, 1).contiguous()}
