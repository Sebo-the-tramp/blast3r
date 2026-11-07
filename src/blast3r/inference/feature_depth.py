"""Sparse evaluation of low-resolution log-depth correction features."""
import numpy as np
import torch
import torch.nn.functional as F


def sample_feature_depth(maps, pixels):
    """Return [base log-depth, features] at full-image pixel centers.

    BLASt3R uses centers at (x+0.5, y+0.5), unlike integer-index inputs in EPO.
    Thus grid = pixels / [W,H] * 2 - 1, with no additional half-pixel shift.
    The same normalized grid samples both full-resolution depth and low-res features.
    """
    base = torch.as_tensor(maps['base_log_depth'])
    features = torch.as_tensor(maps['features'], device=base.device, dtype=base.dtype)
    pixels = torch.as_tensor(pixels, device=base.device, dtype=base.dtype)
    H, W = base.shape
    if pixels.ndim != 2 or pixels.shape[-1] != 2:
        raise ValueError('Expected (P,2) full-image pixel centers')
    if len(pixels) == 0:
        return base.new_empty((0, 2 + features.shape[0]))
    grid = (pixels / pixels.new_tensor([W, H]) * 2 - 1)[None, None]
    def sample(image):
        return F.grid_sample(image[None], grid, mode='bilinear',
                             padding_mode='border', align_corners=False)[0, :, 0].T
    base_samples = sample(base[None])
    ones = base.new_ones((len(pixels), 1))
    return torch.cat((base_samples, ones, sample(features)), dim=-1)


def reconstruct_feature_depth(maps, coefficients):
    """Combine low-res features first; upsample only the scalar correction map."""
    as_numpy = isinstance(maps['base_log_depth'], np.ndarray)
    base = torch.as_tensor(maps['base_log_depth'])
    features = torch.as_tensor(maps['features'], device=base.device, dtype=base.dtype)
    z = torch.as_tensor(coefficients, device=base.device, dtype=base.dtype)
    if z.ndim == 0:
        result = z * base.clamp(max=80).exp()
    else:
        if len(z) != features.shape[0] + 2:
            raise ValueError('Coefficient count must match base + bias + feature channels')
        low = torch.einsum('c,chw->hw', z[2:], features)
        correction = F.interpolate(low[None, None], size=base.shape, mode='bilinear',
                                   align_corners=False)[0, 0]
        result = (z[0] * base + z[1] + correction).clamp(max=80).exp()
    return result.numpy() if as_numpy else result
