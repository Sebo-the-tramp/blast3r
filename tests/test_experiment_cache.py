"""Frozen sampling, cache invalidation, and warm-cache execution without a checkpoint."""
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cache_eth3d import prepare_subset
from blast3r.image import load_image, tokenize
from blast3r.inference.loading import load_scene, sample_scene_items
from blast3r.inference.multidepth import compute_pts3d_multidepths_vggt
from blast3r.inference.scene_cache import load_cached_views, save_cached_views


class ExperimentCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source' / 'courtyard'
        (self.source / 'undistorted_images').mkdir(parents=True)
        (self.source / 'poses').mkdir()
        for i in range(12):
            name = f'{i:03}'
            Image.new('RGB', (48, 32), (i * 10, 20, 30)).save(self.source / 'undistorted_images' / f'{name}.png')
            pose = np.eye(4)
            pose[0, 3] = i
            np.save(self.source / 'poses' / f'{name}.npy', pose)
        self.output = self.root / 'frozen' / 'courtyard'

    def test_frozen_selection_gt_and_idempotency(self):
        images = prepare_subset(self.source, self.output)
        expected = sample_scene_items(sorted((self.source / 'undistorted_images').iterdir()), 'courtyard', 10, 0)
        self.assertEqual([p.name for p in images], [p.name for p in expected])
        self.assertTrue(all(p.is_file() and not p.is_symlink() for p in images))
        manifest = json.loads((self.output / 'selection.json').read_text())
        self.assertEqual(len(manifest['images']), 10)
        with open(self.output / 'gt.pkl', 'rb') as stream:
            gt = pickle.load(stream)
        for name, pose in zip(gt['filenames'], gt['poses']):
            self.assertEqual(pose[0, 3], -int(Path(name).stem))
        times = [p.stat().st_mtime_ns for p in images]
        self.assertEqual(prepare_subset(self.source, self.output), images)
        self.assertEqual(times, [p.stat().st_mtime_ns for p in images])
        with self.assertRaisesRegex(ValueError, 'Subset changed'):
            prepare_subset(self.source, self.output, seed=1)

    def test_corrupt_image_is_rejected(self):
        images = prepare_subset(self.source, self.output)
        images[0].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Cached image changed'):
            prepare_subset(self.source, self.output)

    def test_preprocessed_views_match_and_invalidate(self):
        images = prepare_subset(self.source, self.output)
        views = save_cached_views(self.output, images, size=32)
        fresh = load_image(images[0], size=32)
        torch.testing.assert_close(views['img'][0], fresh['img'])
        with patch('blast3r.inference.loading.load_images', side_effect=AssertionError('decoded again')):
            loaded = load_scene(self.output, size=32, max_frames=10, seed=0)
        torch.testing.assert_close(loaded['img'], views['img'])
        self.assertEqual(loaded['label'], ['courtyard'] * 10)
        self.assertNotIn('K', loaded)
        self.assertNotIn('poses', loaded)
        self.assertIsNone(load_cached_views(self.output, images, size=48))
        self.assertIsNone(load_cached_views(self.output, images, size=32, force_ar=1))
        images[0].touch()
        self.assertIsNone(load_cached_views(self.output, images, size=32))

    def test_vggt_warm_cache_skips_model_and_pca_and_tracks_identity(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cache_identity = 'weights-a'
                self.inference_device = torch.device('cpu')
                self.dense_head = torch.nn.Module()
                self.dense_head.proj = torch.nn.Conv2d(256, 16, 1)
                self.calls = 0

            def forward(self, rgb):
                self.calls += 1
                self.dense_head.proj(torch.randn(1, 256, 4, 4))
                return {'depth': torch.ones(1, 1, 16, 16, 1),
                        'depth_conf': torch.ones(1, 1, 16, 16)}

        model = FakeModel()
        views = {'img': tokenize(torch.zeros(3, 16, 16), 16, chan_dim=-3)[None],
                 'true_shape': torch.tensor([[16, 16]]), 'K': torch.eye(3)[None]}
        directory = self.root / 'vggt'
        first = compute_pts3d_multidepths_vggt(model, views, refine_head=True, cache_dir=directory)
        self.assertEqual(model.calls, 1)
        with patch.object(model, 'forward', side_effect=AssertionError('inference repeated')), \
             patch('blast3r.inference.multidepth.vggt_pca_head_basis', side_effect=AssertionError('PCA repeated')):
            second = compute_pts3d_multidepths_vggt(model, views, refine_head=True, cache_dir=directory)
            scale_only = compute_pts3d_multidepths_vggt(model, views, cache_dir=directory)
        torch.testing.assert_close(first[2], second[2])
        self.assertEqual(scale_only[2][0].shape, (16, 16, 2))
        model.cache_identity = 'weights-b'
        compute_pts3d_multidepths_vggt(model, views, cache_dir=directory)
        self.assertEqual(model.calls, 2)
        views['img'].add_(.01)
        compute_pts3d_multidepths_vggt(model, views, cache_dir=directory)
        self.assertEqual(model.calls, 3)
        views['K'][0, 0, 0] = 2
        compute_pts3d_multidepths_vggt(model, views, cache_dir=directory)
        self.assertEqual(model.calls, 4)


if __name__ == '__main__':
    unittest.main()
