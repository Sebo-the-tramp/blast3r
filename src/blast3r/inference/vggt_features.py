# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Turn cached VGGT-Omega dense features into depth-basis maps."""

import torch
import torch.nn.functional as F


def pca_whiten_feature_map(feature, rank, output_shape, shrinkage=1.0, device=None,
                           effective_rank=None, variance_rho=None):
    """Project one cached ``(256, H/4, W/4)`` map to ``rank`` full-size maps.

    ``shrinkage=0`` is an ordinary PCA rotation and ``shrinkage=1`` is full
    whitening.  The returned maps preserve the PCA amplitudes; apart from the
    bilinear resize, no spatial normalization is applied.
    """
    if feature.ndim != 3:
        raise ValueError(f'expected a CHW feature map, got shape {tuple(feature.shape)}')
    channels, height, width = feature.shape
    if channels != 256:
        raise ValueError(f'expected 256 VGGT feature channels, got {channels}')
    if not 0 <= rank <= channels:
        raise ValueError(f'feature rank must be in [0, {channels}], got {rank}')
    if not 0 <= shrinkage <= 1:
        raise ValueError(f'PCA whitening shrinkage must be in [0, 1], got {shrinkage}')
    if effective_rank is not None and not 0 <= effective_rank <= rank:
        raise ValueError(f'effective rank must be in [0, {rank}], got {effective_rank}')
    if variance_rho is not None and not 0 < variance_rho <= 1:
        raise ValueError(f'PCA variance rho must be in (0, 1], got {variance_rho}')
    if effective_rank is not None and variance_rho is not None:
        raise ValueError('effective rank and PCA variance rho are mutually exclusive')

    out_height, out_width = map(int, output_shape)
    if rank == 0:
        basis = torch.empty(
            (out_height, out_width, 0), device=device or feature.device,
            dtype=torch.float32)
        return basis, {'effective_rank': 0, 'rho_rank': None, 'variance': 0.0}

    # The fp16 cached tensor is the canonical ablation input.  Do all linear
    # algebra in fp32, on the requested inference device.
    x = feature.to(device=device, dtype=torch.float32)
    x = x.permute(1, 2, 0).reshape(height * width, channels)
    x = x - x.mean(dim=0, keepdim=True)

    covariance = x.T @ x / x.shape[0]
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    spectrum = eigenvalues[order].clamp_min(0)
    total_variance = spectrum.sum()
    rho_rank = None
    if variance_rho is not None:
        if total_variance > 0:
            cumulative = spectrum.cumsum(0)
            threshold = (variance_rho * total_variance).clamp_max(cumulative[-1])
            rho_rank = int(torch.searchsorted(cumulative, threshold).item()) + 1
        else:
            rho_rank = 0
        active_rank = min(rank, rho_rank)
    else:
        active_rank = rank if effective_rank is None else effective_rank

    eigenvalues = spectrum[:rank].clamp_min(1e-14)
    eigenvectors = eigenvectors[:, order[:rank]]

    projection = eigenvectors * eigenvalues.pow(-0.5 * shrinkage).unsqueeze(0)
    projected = x @ projection

    # Eigenvector signs are arbitrary.  Fix each sign using its largest-magnitude
    # projected value so cache hits and fresh runs produce visualizable maps with
    # a deterministic orientation.
    extrema = projected.abs().argmax(dim=0)
    signs = projected[extrema, torch.arange(rank, device=projected.device)].sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    projected = projected * signs
    projected[:, active_rank:] = 0

    projected = projected.reshape(height, width, rank).permute(2, 0, 1).unsqueeze(0)
    projected = F.interpolate(
        projected, size=(out_height, out_width), mode='bilinear', align_corners=False)
    basis = projected.squeeze(0).permute(1, 2, 0).contiguous()
    kept_variance = spectrum[:active_rank].sum()
    variance = float((kept_variance / total_variance).item()) if total_variance > 0 else 0.0
    return basis, {
        'effective_rank': active_rank,
        'rho_rank': rho_rank,
        'variance': variance,
    }
