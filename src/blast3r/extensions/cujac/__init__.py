# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import torch

# dedicated CUDA kernels to compute residuals and Jacobians

try:
    from . import ba_with_rigs_cuda_kernels
except ImportError:
    def NoCUDA(*args, **kwargs):
        raise RuntimeError('The bundle-adjustment CUDA extension is not compiled; '
                           'reinstall with `pip install --no-build-isolation -e .` and a working CUDA_HOME')
    class DummyModule:
        def __getattr__(self, n):
            return NoCUDA
    ba_with_rigs_cuda_kernels = DummyModule()

compute_residuals_with_rigs = ba_with_rigs_cuda_kernels.residuals_with_rigs
compute_jacobians_with_rigs = ba_with_rigs_cuda_kernels.jacobians_with_rigs
chol3x3_solve = ba_with_rigs_cuda_kernels.chol3x3_solve
_blockcoo_matvec_cuda = ba_with_rigs_cuda_kernels.blockcoo_matvec


def blockcoo_matvec(blocks, row_indices, col_indices, x, HB1, WB1, transpose_blocks):
    """y = A x (or A^T x) for a block-COO A of [N, bh, bw] blocks. The CUDA kernel only
    instantiates a fixed set of block shapes; others fall back to an equivalent torch version."""
    try:
        return _blockcoo_matvec_cuda(blocks, row_indices, col_indices, x, HB1, WB1, transpose_blocks)
    except RuntimeError as error:
        if 'cannot handle COO blocks' not in str(error):
            raise
    _, bh, bw = blocks.shape
    rows, cols = row_indices.long(), col_indices.long()
    if transpose_blocks:
        contrib = torch.einsum('nij,ni->nj', blocks, x.reshape(HB1, bh).index_select(0, rows))
        y = blocks.new_zeros((WB1, bw)).index_add_(0, cols, contrib)
    else:
        contrib = torch.einsum('nij,nj->ni', blocks, x.reshape(WB1, bw).index_select(0, cols))
        y = blocks.new_zeros((HB1, bh)).index_add_(0, rows, contrib)
    return y.reshape(-1)


_fused_einsum_bki_bkj_cuda = ba_with_rigs_cuda_kernels.fused_einsum_bki_bkj


def fused_einsum_bki_bkj(left, right):
    """out[b] = left[b].T @ right[b]. The CUDA kernel only instantiates a fixed set of block
    sizes (up to 17 depth channels); other sizes (e.g. 34/66/258 VGGT basis channels) fall
    back to an equivalent batched matmul."""
    try:
        return _fused_einsum_bki_bkj_cuda(left, right)
    except RuntimeError as error:
        if 'unsupported (I,J,K)' not in str(error):
            raise
        return torch.bmm(left.transpose(1, 2), right)


# Lie-algebra update functions for SE(3)

def skew_symmetric(w, device):
    """Convert a 3D vector into a skew-symmetric matrix.

    Args:
        w: (B, 3) tensor.
    """
    wx, wy, wz = w.T
    res = w.new_zeros((len(w), 3, 3), device=device)
    res[:, 2, 1] = wx
    res[:, 1, 2] =-wx
    res[:, 2, 0] =-wy
    res[:, 0, 2] = wy
    res[:, 0, 1] =-wz
    res[:, 1, 0] = wz
    return res

def hat_se3(xi):
    """Convert a (B, 6) se(3) vector into a (B, 4, 4) pose matrix.

    Args:
        xi: (B, 6) tensor of [w_x, w_y, w_z, v_x, v_y, v_z].
    """
    w = xi[:,:3]  # rotation part
    v = xi[:,3:]  # translation part
    hat = xi.new_zeros((len(xi), 4, 4))
    hat[:, :3, :3] = skew_symmetric(w, device=xi.device)
    hat[:, :3, 3] = v
    return hat

def exp_se3(xi):
    """Matrix exponential of an se(3) element, returning an element of SE(3)."""
    return torch.matrix_exp(hat_se3(xi))
