# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import types
from pathlib import Path

import torch
from matplotlib import colormaps
from torch import autocast
from torchvision.utils import make_grid, save_image

from blast3r.utils.device import todevice
from blast3r.image import detokenize
from blast3r.utils.geometry import depthmap_to_pts3d
from .vggt_features import pca_whiten_feature_map




@torch.inference_mode()
def compute_pts3d_multidepths(model, views, capture_backend_data=False,
                              backend_data=None, vggt_feature_rank=0,
                              vggt_whitening=1.0):
    assert not isinstance(model, types.LambdaType), 'the model must be loaded before reaching here'

    if getattr(model, 'depth_backend', 'blast3r') == 'vggt_omega':
        return compute_vggt_omega_depths(
            model, views, capture_backend_data=capture_backend_data,
            backend_data=backend_data, feature_rank=vggt_feature_rank,
            whitening=vggt_whitening)

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

    return conf, pts3d, multidepth, depth_mode, None


@torch.inference_mode()
def compute_vggt_omega_depths(model, views, capture_backend_data=False,
                              backend_data=None, feature_rank=0, whitening=1.0):
    """Adapt VGGT-Omega outputs to BLASt3R's log-multidepth interface.

    Every view is evaluated independently as a one-frame VGGT-Omega batch. The
    cached fused feature map is PCA-projected independently per image and added
    after ``[log(depth), 1]`` as a fixed-rank log-depth basis.
    """
    if backend_data is None:
        backend_data = extract_vggt_omega_backend(
            model, views,
            capture_features=capture_backend_data or feature_rank > 0)

    result = vggt_omega_backend_to_multidepth(
        backend_data, views, feature_rank=feature_rank, whitening=whitening)
    return *result, backend_data


@torch.inference_mode()
def extract_vggt_omega_backend(model, views, capture_features=True):
    """Run independent monocular VGGT forwards and return a CPU cache artifact."""
    imgs_shape = views['img'].shape
    assert len(imgs_shape) == 5 and imgs_shape[-3:] == (3, 16, 16), \
        f'expected tokenized multi-view images, got {imgs_shape}'

    shapes = [tuple(map(int, shape)) for shape in views['true_shape'].tolist()]
    images = []
    for image, shape in zip(views['img'], shapes):
        image = detokenize(image, shape, chan_dim=-3)
        images.append((image.float() / 2 + 0.5).clamp(0, 1))

    print(f'>> Extracting monocular VGGT-Omega depthmaps for {len(images)} images ...')
    native_depth, confidence, intrinsics_out, fused_features = [], [], [], []
    for image in images:
        # A separate call is intentional: this experiment tests VGGT-Omega as
        # a monocular prior, with no information exchange across views.
        depth, conf, intrinsics, features = model.predict(
            image.unsqueeze(0), capture_features=capture_features)
        native_depth.append(depth[0, 0, ..., 0].float().clamp_min(1e-8).cpu())
        confidence.append(conf[0, 0].float().cpu())
        intrinsics_out.append(intrinsics[0, 0].float().cpu())
        if capture_features:
            fused_features.append(features[0, 0])

    return {
        'schema': 1,
        'feature_name': 'dense_head_fused_pre_projection',
        'feature_layout': 'CHW',
        'feature_dtype': 'float16',
        'feature_stride': 4,
        'features': fused_features,
        'depth': native_depth,
        'confidence': confidence,
        'intrinsics': intrinsics_out,
    }


@torch.inference_mode()
def vggt_omega_backend_to_multidepth(backend_data, views, feature_rank=0,
                                     whitening=1.0):
    """Build full-resolution depth bases strictly downstream of the raw cache."""
    shapes = [tuple(map(int, shape)) for shape in views['true_shape'].tolist()]

    required = ('depth', 'confidence', 'intrinsics')
    if any(key not in backend_data for key in required):
        raise ValueError(f'invalid VGGT backend cache; expected keys {required}')
    if not all(len(backend_data[key]) == len(shapes) for key in required):
        raise ValueError('VGGT backend cache image count does not match the input views')
    if feature_rank and len(backend_data.get('features', ())) != len(shapes):
        raise ValueError(
            'VGGT cache has no fused features for the requested feature-rank ablation')

    device = views['img'].device
    out_conf, pts3d, multidepth = [], [], []
    for index, shape in enumerate(shapes):
        height, width = shape
        depth_i = backend_data['depth'][index].to(device=device, dtype=torch.float32)
        conf_i = backend_data['confidence'][index].to(device=device, dtype=torch.float32)
        K_i = backend_data['intrinsics'][index].to(device=device, dtype=torch.float32)
        if depth_i.shape != (height, width):
            raise ValueError(
                f'cached VGGT depth has shape {tuple(depth_i.shape)}, '
                f'expected {(height, width)}')
        log_depth = depth_i.log()
        feature = None
        if feature_rank:
            feature = backend_data['features'][index]
            expected_feature_shape = (256, height // 4, width // 4)
            if tuple(feature.shape) != expected_feature_shape:
                raise ValueError(
                    f'cached VGGT feature has shape {tuple(feature.shape)}, '
                    f'expected {expected_feature_shape}')
        basis = pca_whiten_feature_map(
            feature if feature_rank else torch.empty((256, 1, 1), dtype=torch.float16),
            feature_rank, shape, shrinkage=whitening, device=device)

        # if feature_rank:  # log-depth + independently scaled signed basis maps
        #     depth_viz = (log_depth - log_depth.amin()) / (log_depth.amax() - log_depth.amin()).clamp_min(1e-8)
        #     basis_viz = 0.5 + 0.5 * basis / basis.abs().amax(dim=(0, 1), keepdim=True).clamp_min(1e-8)
        #     depth_rgb = torch.from_numpy(colormaps['viridis'](depth_viz.cpu().numpy())[..., :3]).permute(2, 0, 1)
        #     basis_rgb = torch.from_numpy(colormaps['viridis'](basis_viz.permute(2, 0, 1)[:15].cpu().numpy())[..., :3]).permute(0, 3, 1, 2)
        #     tiles = torch.cat((depth_rgb[None], basis_rgb)).float()
        #     name = Path(str(views.get('instance', [index])[index])).stem
        #     viz_path = Path('.cache/blast3r/vggt_basis_viz') / str(views.get('label', ['scene'])[index]) / f'{name}_r{feature_rank}.png'
        #     viz_path.parent.mkdir(parents=True, exist_ok=True)
        #     save_image(make_grid(tiles, nrow=4), viz_path)

        out_conf.append(conf_i)
        pts3d.append(depthmap_to_pts3d(depth_i, K_i))
        depth_basis = torch.cat((
            log_depth[..., None], torch.ones_like(log_depth[..., None]), basis), dim=-1)
        expected_multidepth_shape = (height, width, feature_rank + 2)
        if tuple(depth_basis.shape) != expected_multidepth_shape:
            raise RuntimeError(
                f'VGGT multidepth has shape {tuple(depth_basis.shape)}, '
                f'expected {expected_multidepth_shape}')
        multidepth.append(depth_basis.contiguous())

    print(f'Found {feature_rank + 2} Z components '
          f'(VGGT-Omega log-depth + constant + {feature_rank} feature maps)')
    return out_conf, pts3d, multidepth, 'log_multidepth'
