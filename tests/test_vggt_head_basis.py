"""CPU checks for shared VGGT head refinement; no checkpoint or GPU required."""
import unittest
import torch
import torch.nn.functional as F
from blast3r.inference.multidepth import vggt_shared_head_basis


class SharedHeadTests(unittest.TestCase):
    def test_basis_matches_shared_update_to_real_projection(self):
        torch.manual_seed(0)
        features = torch.randn(1, 256, 3, 5)
        head = torch.nn.Conv2d(256, 16, 1)
        base_log = F.pixel_shuffle(head(features), 4)[0, 0].detach()
        base = torch.stack((base_log, torch.ones_like(base_log)), -1)
        basis = vggt_shared_head_basis(base, features[0])
        delta_w = torch.randn(256) * 0.01
        delta_b = torch.tensor(0.15)
        coefficients = torch.cat((torch.ones(1), delta_b[None], delta_w))
        with torch.no_grad():
            head.weight += delta_w[None, :, None, None]
            head.bias += delta_b
        expected = F.pixel_shuffle(head(features), 4)[0, 0]
        torch.testing.assert_close(basis @ coefficients, expected, atol=2e-6, rtol=2e-5)
        self.assertEqual(basis.shape, (12, 20, 258))
        self.assertEqual(basis.device.type, 'cpu')

    def test_initialization_and_scale(self):
        base_log = torch.randn(8, 12)
        basis = vggt_shared_head_basis(
            torch.stack((base_log, torch.ones_like(base_log)), -1), torch.randn(256, 2, 3))
        coefficients = torch.zeros(258)
        coefficients[0] = 1
        torch.testing.assert_close(basis @ coefficients, base_log)
        coefficients[1] = torch.tensor(2.).log()
        torch.testing.assert_close((basis @ coefficients).exp(), 2 * base_log.exp())

    def test_invalid_feature_geometry(self):
        with self.assertRaises(ValueError):
            vggt_shared_head_basis(torch.zeros(8, 12, 2), torch.zeros(256, 3, 2))


if __name__ == '__main__':
    unittest.main()
