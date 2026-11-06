import unittest
import torch
from blast3r.inference.ba_solver import block_inner_product, grouped_large_gram


class LargeBlocksTests(unittest.TestCase):
    def test_cross_products(self):
        for width in (2, 3, 6, 258):
            left = torch.randn(7, 3, 258, dtype=torch.float64)[:, 2:]
            right = torch.randn(7, 3, width, dtype=torch.float64)[:, 2:]
            actual = block_inner_product(left, right)
            expected = torch.stack([l.T @ r for l, r in zip(left, right)])
            torch.testing.assert_close(actual, expected)

    def test_chunked_gram_matches_dense_with_missing_group(self):
        torch.manual_seed(0)
        blocks = torch.randn(90, 1, 258, dtype=torch.float64)
        indices = torch.arange(90) % 3
        actual = grouped_large_gram(blocks, indices, 4)
        expected = torch.zeros_like(actual)
        for i in range(4):
            rows = blocks[indices == i].reshape(-1, 258)
            expected[i] = rows.T @ rows
        torch.testing.assert_close(actual, expected)

    def test_empty_gram(self):
        out = grouped_large_gram(torch.empty(0, 1, 258), torch.empty(0, dtype=torch.long), 2)
        self.assertEqual(out.shape, (2, 258, 258))
        self.assertEqual(out.count_nonzero(), 0)


if __name__ == '__main__':
    unittest.main()
