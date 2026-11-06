"""PCA algebra and the complete BA linear solve on CPU, with kernel-shape checks."""
import re
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from blast3r.inference.multidepth import vggt_pca_head_basis
from blast3r.inference import ba_solver as ba
from blast3r.inference.feature_depth import reconstruct_feature_depth


class PcaHeadTests(unittest.TestCase):
    def test_pca_shapes_initialization_and_repeatability(self):
        torch.manual_seed(2)
        base = torch.stack((torch.randn(16, 20), torch.ones(16, 20)), -1)
        features = torch.randn(256, 4, 5)
        basis = vggt_pca_head_basis(base, features)
        self.assertEqual(basis['features'].shape, (15, 4, 5))
        weights = torch.zeros(17)
        weights[0] = 1
        torch.testing.assert_close(reconstruct_feature_depth(basis, weights), base[..., 0].exp())
        weights[1] = torch.tensor(2.).log()
        torch.testing.assert_close(reconstruct_feature_depth(basis, weights), 2 * base[..., 0].exp())
        scores = basis['features'].flatten(1).T
        torch.testing.assert_close(scores.mean(0), torch.zeros(15), atol=1e-6, rtol=0)
        cov = scores.T @ scores
        torch.testing.assert_close(cov, cov.diag().diag(), atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(basis, vggt_pca_head_basis(base, features))

    def test_constant_features(self):
        basis = vggt_pca_head_basis(torch.ones(8, 8, 2), torch.ones(256, 2, 2))
        self.assertEqual(basis['features'].count_nonzero(), 0)

    def test_full_linear_solve_uses_supported_kernel_shapes(self):
        # Emulate custom kernels on CPU, rejecting shapes absent from CUDA source.
        source = (Path(__file__).resolve().parents[1] / 'src/blast3r/extensions/cujac/sfm_jacobian_rigs.cu').read_text()
        inner_shapes = {tuple(map(int, x)) for x in re.findall(r'kernel2<T,(\d+),(\d+),(\d+)>', source)}
        coo_shapes = {tuple(map(int, x)) for x in re.findall(r'LAUNCH_KERNEL\((\d+), (\d+)\)', source)}
        visited = set()
        def inner(left, right):
            shape = (left.shape[2], right.shape[2], left.shape[1])
            self.assertIn(shape, inner_shapes)
            visited.add(('inner', shape))
            return left.transpose(1, 2) @ right
        def matvec(blocks, rows, cols, x, nr, nc, transpose):
            h, w = blocks.shape[1:]
            self.assertIn((h, w), coo_shapes)
            visited.add(('coo', (h, w)))
            matrix = blocks.new_zeros(nr*h, nc*w)
            for b, r, c in zip(blocks, rows, cols):
                matrix[r*h:(r+1)*h, c*w:(c+1)*w] += b
            return (matrix.T if transpose else matrix) @ x.ravel()
        def chol(blocks, rhs, jitter):
            self.assertEqual(blocks.shape[1:], (3, 3))
            return torch.linalg.solve(blocks + jitter*torch.eye(3), rhs[..., None]).squeeze(-1)
        torch.manual_seed(3)
        n_images, n_points, n_obs = 6, 24, 48
        rigs = torch.stack((8 * torch.arange(1, 7), torch.arange(6), torch.arange(6)))
        ids = torch.cat([(torch.arange(8) + 4*i) % n_points for i in range(n_images)])
        for shared_camera in (False, True):
            local_rigs = rigs.clone()
            if shared_camera: local_rigs[2] = 0
            solver = ba.GaussNewtonSolver(1 if shared_camera else n_images, n_images, n_images, n_points, 'cpu',
                                         dampen=.1, adaptive_dampen=False, dtype=torch.float64)
            jcam = torch.randn(n_obs, 3, 8, dtype=torch.float64)
            jdepth = torch.randn(n_obs, 1, 17, dtype=torch.float64)
            jdepth[..., 0] = 0  # fixed original log-depth coefficient
            jpoints = torch.randn(n_obs, 3, 3, dtype=torch.float64)
            J = solver.init_from_packed(local_rigs, ids, jcam, jdepth, jpoints)
            dense = J.to_dense().clone()
            residuals = torch.randn(n_obs, 3, dtype=torch.float64)
            weights = torch.ones(n_obs, 1, dtype=torch.float64)
            expected = torch.linalg.solve(dense.T @ dense + .1*torch.eye(dense.shape[1]),
                                          dense.T @ residuals.ravel())
            with patch.object(ba, 'block_inner_product', inner), patch.object(ba, 'blockcoo_matvec', matvec), patch.object(ba, 'chol3x3_solve', chol):
                actual = solver.solve_delta(J, residuals, weights)
            torch.testing.assert_close(actual.double(), expected, atol=1e-4, rtol=1e-3)
        self.assertIn(('coo', (3, 17)), visited)
        self.assertIn(('inner', (17, 2, 1)), visited)


if __name__ == '__main__':
    unittest.main()
