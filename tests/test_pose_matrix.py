"""Protocol isolation, exact AUC, frozen initialization, and zero-weight residuals."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_pose_matrix import (write_report, dataset_identity, evaluate, select_experiments,
                             initial_cache_path, prepare_scene)
from blast3r.eval.metrics.relpose import calculate_auc_exact_np, compute_pose_errors
from blast3r.inference.experiment_matrix import EXPERIMENTS, Experiment, run_experiment, initialization_key
from blast3r.inference.joint_vggt import compute_joint_vggt, compute_mono_vggt, joint_input_identity


class MatrixTests(unittest.TestCase):
    def test_mono_mode_has_no_relative_vggt_pose_rows(self):
        rows = select_experiments('mono')
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(e.pose_source == 'blast3r' for e in rows))
        self.assertEqual(len(select_experiments('joint')), 8)
        self.assertEqual([e.name for e in select_experiments('mono', ['blast3r_ba_vggt'])], ['blast3r_ba_vggt'])
        with self.assertRaisesRegex(ValueError, 'cannot supply relative poses'):
            select_experiments('mono', ['vggt_no_ba'])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'config.json').write_text(json.dumps({'vggt_inference': 'mono'}))
            report = write_report(output, ['one'], 'a')
            self.assertEqual([r['experiment'] for r in report], [e.name for e in rows])
            self.assertIn('Single-image VGGT', (output / 'mean_auc.tex').read_text())

    def test_mono_inference_uses_one_image_and_reuses_caches_in_any_order(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cache_identity = 'mono-test'
                self.inference_device = torch.device('cpu')
                self.dense_head = torch.nn.Module()
                self.dense_head.proj = torch.nn.Conv2d(256, 16, 1)
                self.calls = 0
            def forward(self, images):
                self.calls += 1
                self_shape = tuple(images.shape)
                if self_shape != (1, 1, 3, 16, 16):
                    raise AssertionError(f'Cross-image inference: {self_shape}')
                self.dense_head.proj(torch.ones(1, 256, 4, 4))
                pose = torch.zeros(1, 1, 9)
                pose[..., 3] = 1
                pose[..., 7:] = torch.pi / 2
                return {'pose_enc': pose, 'depth': torch.full((1, 1, 16, 16, 1), float(self.calls)),
                        'depth_conf': torch.ones(1, 1, 16, 16)}
        views = {'img': torch.stack([torch.full((1, 3, 16, 16), float(i)) for i in range(10)]),
                 'true_shape': torch.tensor([[16, 16]] * 10)}
        def pca(base, features):
            return {'base_log_depth': base[..., 0], 'features': torch.zeros(15, 4, 4)}
        with tempfile.TemporaryDirectory() as temporary, patch(
                'blast3r.inference.multidepth.vggt_pca_head_basis', side_effect=pca):
            model = Model()
            result = compute_mono_vggt(model, views, temporary)
            self.assertEqual(model.calls, 10)
            self.assertNotIn('w2c', result)
            self.assertEqual(result['inference'], 'mono')
            self.assertEqual(len(result['pca_basis']), 10)
            torch.testing.assert_close(torch.stack([p[0, 0, 2] for p in result['pts3d']]), torch.arange(1, 11).float())
            with patch.object(model, 'forward', side_effect=AssertionError('Warm cache loaded model')):
                reverse = compute_mono_vggt(model, {k: v.flip(0) for k, v in views.items()}, temporary)
            for i, base in enumerate(reverse['multidepth']):
                self.assertEqual(base.shape, (16, 16, 2))
                torch.testing.assert_close(base[..., 0], result['multidepth'][9 - i][..., 0])
                torch.testing.assert_close(base[..., 1], torch.ones(16, 16))

    def test_modes_share_tracks_and_native_initialization_but_not_vggt_initialization(self):
        from types import SimpleNamespace
        views = {'img': torch.zeros(10, 1, 3, 16, 16), 'true_shape': torch.tensor([[16, 16]] * 10)}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            torch.save({'inference': 'native'}, directory / 'blast3r_depth.pth')
            tracks = torch.ones(3, 10, 2)
            torch.save(tracks, directory / 'tracks.pth')
            def initialize(source, prediction, tracks, mono_cam):
                return {'source': source, 'inference': prediction['inference']}
            with patch('blast3r.inference.joint_vggt.compute_mono_vggt', return_value={'inference': 'mono'}), \
                 patch('blast3r.inference.joint_vggt.compute_joint_vggt', return_value={'inference': 'joint'}) as joint, \
                 patch('blast3r.inference.experiment_matrix.initialize_source', side_effect=initialize) as init:
                models = {'vggt': SimpleNamespace(cache_identity='weights')}
                mono = prepare_scene(views, directory, models, True, 0, False, directory, 'mono')
                joint.assert_not_called()
                self.assertNotIn('vggt', mono[3])
                sequence = prepare_scene(views, directory, models, True, 0, False, directory, 'joint')
                self.assertEqual(init.call_count, 4)  # Native initializer is built once across both modes.
                torch.testing.assert_close(mono[2], sequence[2])
                self.assertEqual(mono[3]['blast3r_vggt']['inference'], 'mono')
                self.assertEqual(sequence[3]['blast3r_vggt']['inference'], 'joint')
                for mode in ('mono', 'joint'):
                    self.assertTrue(initial_cache_path(directory, 'blast3r_vggt', mode).is_file())

    def test_tree_initialization_follows_the_ba_depth_source(self):
        for residual in ('blast3r', 'vggt'):
            self.assertEqual(initialization_key(Experiment('blast3r', residual)), f'blast3r_{residual}')
            self.assertEqual(initialization_key(Experiment('vggt', residual)), 'vggt')
        self.assertEqual(initialization_key(Experiment('blast3r', 'off')), 'blast3r_blast3r')
        self.assertEqual(initialization_key(Experiment('blast3r', 'none')), 'blast3r_blast3r')
        self.assertEqual(initialization_key(Experiment('blast3r', 'none'), baseline_init_depth='vggt'), 'blast3r_vggt')

    def test_exact_auc_and_global_gauge_invariance(self):
        self.assertAlmostEqual(calculate_auc_exact_np(np.array([.5]), np.array([0.]), 1), .5)
        self.assertAlmostEqual(calculate_auc_exact_np(np.array([0., 2.]), np.array([0., 0.]), 1), .5)
        poses = torch.eye(4).repeat(3, 1, 1)
        poses[:, 0, 3] = torch.tensor([0., 1., 2.])
        other = poses.clone()
        other[:, :3, 3] *= 7
        other[:, :3, 3] += torch.tensor([3., 2., 1.])
        metrics = compute_pose_errors(poses, other, auc_mode='exact')
        self.assertGreater(metrics['Auc_1'], 99.9)

    def test_evaluation_outputs_finite_json_serializable_metrics(self):
        import pickle
        poses = np.repeat(np.eye(4)[None], 10, axis=0).astype(np.float32)
        poses[:, 0, 3] = np.arange(10)
        names = [f'{i:02}.jpg' for i in range(10)]
        with tempfile.TemporaryDirectory() as temporary:
            with open(Path(temporary) / 'gt.pkl', 'wb') as stream:
                pickle.dump({'filenames': names, 'poses': poses}, stream)
            result = evaluate(temporary, {'views': {name: {'cam2w': pose} for name, pose in zip(names, poses)}}, 'exact')
            json.dumps(result, allow_nan=False)
            self.assertGreater(result['Auc_1'], 99.9)
            self.assertEqual(result['n_pairs'], 45)

    def test_reports_do_not_hide_failures_or_pending_scenes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            directory = output / EXPERIMENTS[0].name
            directory.mkdir()
            first = dict(status='ok', protocol='a', metrics={'Auc_1': 80, 'Auc_5': 90, 'Auc_30': 100})
            (directory / 'one.json').write_text(json.dumps(first))
            rows = write_report(output, ['one', 'two'], 'a')
            self.assertIsNone(rows[0]['Auc_1'])
            (directory / 'two.json').write_text(json.dumps(dict(status='failed', protocol='a', error='disconnected')))
            rows = write_report(output, ['one', 'two'], 'a')
            self.assertEqual(rows[0]['Auc_1'], 40)
            self.assertEqual(rows[0]['n_failed'], 1)
            self.assertTrue(rows[0]['complete'])
            self.assertIsNone(write_report(output, ['one', 'two'], 'different')[0]['Auc_1'])
            self.assertEqual(len(rows), 8)
            self.assertIn('l ccc ccc', (output / 'mean_auc.tex').read_text())

    def test_frozen_image_changes_are_rejected_before_result_reuse(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'scene'
            directory.mkdir()
            entries = []
            for i in range(10):
                name = f'{i:02}.jpg'
                content = str(i).encode()
                (directory / name).write_bytes(content)
                entries.append({'name': name, 'sha256': hashlib.sha256(content).hexdigest()})
            (directory / 'selection.json').write_text(json.dumps({'seed': 0, 'max_frames': 10, 'images': entries}))
            self.assertEqual(len(dataset_identity(temporary, ['scene'], 0)[0]['images']), 10)
            (directory / '00.jpg').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Frozen image changed'):
                dataset_identity(temporary, ['scene'], 0)

    def test_joint_cache_depends_on_order_and_needs_no_model_on_hit(self):
        views = {'img': torch.stack([torch.zeros(1, 3, 16, 16), torch.ones(1, 3, 16, 16)]),
                 'true_shape': torch.tensor([[16, 16], [16, 16]])}
        reverse = {key: value.flip(0) for key, value in views.items()}
        key = joint_input_identity(views, 'weights')
        self.assertNotEqual(key, joint_input_identity(reverse, 'weights'))
        self.assertNotEqual(key, joint_input_identity(views, 'other-weights'))
        class NoModel:
            cache_identity = 'weights'
            def to(self, device):
                raise AssertionError('Loaded model for warm cache')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'joint.pth'
            torch.save({'identity': key, 'inference': 'joint'}, path)
            self.assertEqual(compute_joint_vggt(NoModel(), views, path)['inference'], 'joint')

    def test_joint_feature_capture_preserves_all_ten_frames_across_head_chunks(self):
        class Model(torch.nn.Module):
            cache_identity = 'joint-test'
            inference_device = torch.device('cpu')
            def __init__(self):
                super().__init__()
                self.dense_head = torch.nn.Module()
                self.dense_head.proj = torch.nn.Conv2d(256, 16, 1)
            def forward(self, images):
                n = images.shape[1]
                for start in range(0, n, 8):
                    features = torch.arange(start, min(n, start + 8)).float()[:, None, None, None]
                    self.dense_head.proj(features.expand(-1, 256, 4, 4))
                pose = torch.zeros(1, n, 9)
                pose[..., 3] = 1
                pose[..., 7:] = torch.pi / 2
                return {'pose_enc': pose, 'depth': torch.ones(1, n, 16, 16, 1),
                        'depth_conf': torch.ones(1, n, 16, 16)}
        views = {'img': torch.zeros(10, 1, 3, 16, 16), 'true_shape': torch.tensor([[16, 16]] * 10)}
        with tempfile.TemporaryDirectory() as temporary:
            model = Model()
            cache = Path(temporary) / 'joint.pth'
            result = compute_joint_vggt(model, views, cache)
            self.assertEqual(result['projection_features'].shape, (10, 256, 4, 4))
            torch.testing.assert_close(result['projection_features'][:, 0, 0, 0], torch.arange(10).float())
            self.assertEqual(len(result['pca_basis']), 10)
            with patch.object(model, 'forward', side_effect=AssertionError('Repeated joint inference')):
                loaded = compute_joint_vggt(model, views, cache)
            torch.testing.assert_close(loaded['w2c'], result['w2c'])

    def test_ba_depth_modes_reuse_initial_k_p_x_without_mutation(self):
        from blast3r.inference.rigs import RigsData
        from blast3r.inference.experiment_matrix import pack_rigs
        n = 2
        K = torch.tensor([[[100., 0, 8], [0, 100, 8], [0, 0, 1]]])
        rig = RigsData(1, np.zeros(n, np.int32), K, torch.eye(4)[None], torch.ones(n), None,
                       'log', n, np.arange(n, dtype=np.int32), torch.eye(4).repeat(n, 1, 1))
        points = torch.tensor([[0., 0., 2.], [.01, 0., 2.], [0., .01, 2.]])
        initial = {'rigs': pack_rigs(rig), 'points': points.clone()}
        tracks = torch.tensor([[[8., 8.], [8., 8.]], [[8.5, 8.], [8.5, 8.]], [[8., 8.5], [8., 8.5]]])
        basis = torch.zeros(16, 16, 17)
        basis[..., 0] = torch.tensor(2.).log()
        basis[..., 1] = 1
        native = {'pts3d': [torch.ones(16, 16, 3)] * n, 'conf': [torch.ones(16, 16)] * n,
                  'multidepth': [basis] * n}
        joint = {**native, 'pca_basis': [{'base_log_depth': basis[..., 0], 'features': torch.zeros(15, 4, 4)}] * n}
        seen = []
        def fake_solver(rigs, data, xyz, **options):
            torch.testing.assert_close(xyz, points)
            torch.testing.assert_close(rigs.K, K)
            self.assertTrue(torch.equal(rigs.P_w2rig, torch.eye(4).repeat(n, 1, 1)))
            self.assertEqual(options['optim_Z'], options['weight_z'] != 0)
            self.assertFalse(options['optim_Z0'])
            seen.append(data.pix2d_dep.shape[-1])
            rigs.K.add_(1)
            xyz.add_(1)
            return rigs, xyz
        with patch('blast3r.inference.engine.bundle_adjustment', side_effect=fake_solver):
            for depth in ('none', 'blast3r', 'vggt'):
                run_experiment(Experiment('vggt', depth), initial, tracks, native, joint, device='cpu')
        self.assertEqual(seen, [2, 17, 17])
        torch.testing.assert_close(initial['points'], points)
        torch.testing.assert_close(initial['rigs']['K'], K)

    def test_all_pairs_does_not_require_a_retrieval_model(self):
        from blast3r.inference.tracks import OnlineMatcher
        class Matcher(torch.nn.Module):
            patch_size, n_cls, enc_dim, dec_dim = 16, 0, 4, 4
        with patch('blast3r.inference.tracks.CausalTrack3R', Matcher):
            matcher = OnlineMatcher((Matcher(), None), 2, 1, 16, 'cpu')
            matcher.init(allow_no_retrieval=True)
            self.assertIsNone(matcher.retriever)
            with self.assertRaisesRegex(ValueError, 'Image retrieval requires'):
                matcher.init()

    def test_zero_weight_is_not_clamped_back_on(self):
        from blast3r.inference import ba_solver as ba
        class FakeJac:
            shape = (3, 19)
            def __rmatmul__(self, values):
                return self
            def to_dense(self):
                return torch.zeros(19)
            def __imul__(self, values):
                self.weights = values
                raise StopIteration
        solver = ba.GaussNewtonSolver(1, 1, 1, 1, 'cpu')
        solver.n_kpts, solver.n_zcf = 1, 8
        jac = FakeJac()
        with self.assertRaises(StopIteration):
            solver.solve_delta(jac, torch.ones(1, 3), torch.tensor([[1., 1., 0.]]))
        self.assertEqual(jac.weights[2, 0], 0)


if __name__ == '__main__':
    unittest.main()
