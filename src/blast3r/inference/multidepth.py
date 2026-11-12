# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import types
import torch
from torch import autocast

from blast3r.utils.device import todevice
from blast3r.image import detokenize
from blast3r.utils.geometry import depthmap_to_pts3d




@torch.inference_mode()
def compute_pts3d_multidepths(model, views, capture_backend_data=False):
    assert not isinstance(model, types.LambdaType), 'the model must be loaded before reaching here'

    if getattr(model, 'depth_backend', 'blast3r') == 'vggt_omega':
        return compute_vggt_omega_depths(
            model, views, capture_backend_data=capture_backend_data)

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
def compute_vggt_omega_depths(model, views, capture_backend_data=False):
    """Adapt VGGT-Omega outputs to BLASt3R's log-multidepth interface.

    Every view is evaluated independently as a one-frame VGGT-Omega batch. Its
    single positive depth map becomes ``[log(depth), 1]``, so BA can optimize
    alpha and beta but no corrective depth components.
    """
    imgs_shape = views['img'].shape
    assert len(imgs_shape) == 5 and imgs_shape[-3:] == (3, 16, 16), \
        f'expected tokenized multi-view images, got {imgs_shape}'

    shapes = [tuple(map(int, shape)) for shape in views['true_shape'].tolist()]
    images = []
    for image, shape in zip(views['img'], shapes):
        image = detokenize(image, shape, chan_dim=-3)
        images.append((image.float() / 2 + 0.5).clamp(0, 1))

    print(f'>> Extracting monocular VGGT-Omega depthmaps for {len(images)} images ...')
    out_conf, pts3d, multidepth = [], [], []
    native_depth, intrinsics_out, fused_features = [], [], []
    for image in images:
        # A separate call is intentional: this experiment tests VGGT-Omega as
        # a monocular depth prior, with no information exchange across views.
        depth, conf, intrinsics, features = model.predict(
            image.unsqueeze(0), capture_features=capture_backend_data)
        depth_i = depth[0, 0, ..., 0].float().clamp_min(1e-8)
        conf_i = conf[0, 0].float()
        K_i = intrinsics[0, 0].float()

        out_conf.append(conf_i)
        pts3d.append(depthmap_to_pts3d(depth_i, K_i))
        log_depth = depth_i.log()
        multidepth.append(torch.stack((log_depth, torch.ones_like(log_depth)), dim=-1))

        if capture_backend_data:
            native_depth.append(depth_i.cpu())
            intrinsics_out.append(K_i.cpu())
            fused_features.append(features[0, 0])

    print('Found 2 Z components (VGGT-Omega log-depth + constant)')
    backend_data = None
    if capture_backend_data:
        backend_data = {
            'schema': 1,
            'feature_name': 'dense_head_fused_pre_projection',
            'feature_layout': 'CHW',
            'feature_dtype': 'float16',
            'feature_stride': 4,
            'features': fused_features,
            'depth': native_depth,
            'confidence': [value.cpu() for value in out_conf],
            'intrinsics': intrinsics_out,
        }
    return out_conf, pts3d, multidepth, 'log_multidepth', backend_data
