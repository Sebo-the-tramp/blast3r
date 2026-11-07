import unittest

from blast3r.inference.loading import sample_scene_items


class SceneSamplingTests(unittest.TestCase):
    def test_subset_is_reproducible_and_keeps_source_order(self):
        items = list(range(100))
        first = sample_scene_items(items, 'scene-a', max_frames=12, seed=7)
        second = sample_scene_items(items, 'scene-a', max_frames=12, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 12)

    def test_scene_and_base_seed_affect_draw(self):
        items = list(range(100))
        baseline = sample_scene_items(items, 'scene-a', max_frames=12, seed=7)
        self.assertNotEqual(baseline, sample_scene_items(items, 'scene-b', 12, seed=7))
        self.assertNotEqual(baseline, sample_scene_items(items, 'scene-a', 12, seed=8))

    def test_short_scene_is_unchanged(self):
        items = list(range(5))
        self.assertEqual(sample_scene_items(items, 'scene', max_frames=10), items)

    def test_minus_one_means_unlimited(self):
        items = list(range(100))
        self.assertEqual(sample_scene_items(items, 'scene', max_frames=-1), items)

    def test_rejects_nonpositive_cap_when_sampling_is_needed(self):
        with self.assertRaisesRegex(ValueError, 'unlimited.*positive integer'):
            sample_scene_items(range(5), 'scene', max_frames=0)
        with self.assertRaisesRegex(ValueError, 'unlimited.*positive integer'):
            sample_scene_items(range(5), 'scene', max_frames=-2)


if __name__ == '__main__':
    unittest.main()
