"""Independent pose-source and BA-depth choices for the ETH3D experiment table."""
import copy
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Experiment:
    pose_source: str
    depth_residual: str  # off (no BA), none (2D BA), blast3r, vggt

    @property
    def name(self):
        return f'{self.pose_source}_{"no_ba" if self.depth_residual == "off" else "ba_" + self.depth_residual}'

    @property
    def ba(self):
        return self.depth_residual != 'off'


EXPERIMENTS = tuple(Experiment(source, depth) for source in ('vggt', 'blast3r')
                    for depth in ('off', 'none', 'blast3r', 'vggt'))


def initialization_key(experiment, blast3r_init_depth='residual', baseline_init_depth='blast3r'):
    if experiment.pose_source == 'vggt':
        return 'vggt'
    if blast3r_init_depth == 'residual':
        depth = experiment.depth_residual if experiment.depth_residual in ('blast3r', 'vggt') else baseline_init_depth
    else:
        depth = blast3r_init_depth
    return f'blast3r_{depth}'


def make_depth_views(predictions, basis_key='multidepth'):
    return [{'pts3d': xyz, 'conf': conf, 'log_multidepth': basis}
            for xyz, conf, basis in zip(predictions['pts3d'], predictions['conf'], predictions[basis_key])]


def pack_rigs(rigs):
    from blast3r.utils.device import todevice
    return todevice(vars(rigs), 'cpu')


def unpack_rigs(state):
    from .rigs import RigsData
    return RigsData(**copy.deepcopy(state))


def initialize_source(source, predictions, tracks, mono_cam=True):
    """Freeze K/P/X for one pose-source/depth-initializer combination."""
    from .rigs import RigsData, gather_pts3d, init_poses_minspan_tree
    from .tracks import gather_data_from_dense_tracks
    frames = make_depth_views(predictions)
    observations = gather_data_from_dense_tracks(frames, tracks)
    if source == 'blast3r':
        rigs = init_poses_minspan_tree(frames, observations, 'log_multidepth', mono_cam=mono_cam)
    elif source == 'vggt':
        n = len(frames)
        K = predictions['K'].clone()
        if mono_cam:
            # Same square-pixel/shared-focal convention as BLAST3R initialization.
            focal = K[:, 0, 0].median()
            K[:, 0, 0] = K[:, 1, 1] = focal
            K = K[:1]
        ids = np.arange(n, dtype=np.int32)
        rigs = RigsData(len(K), np.zeros(n, dtype=np.int32) if mono_cam else ids.copy(), K,
                        torch.eye(4)[None].repeat(len(K), 1, 1), torch.ones(n), None, 'log',
                        n, ids.copy(), predictions['w2c'].clone())
    else:
        raise ValueError(f'Unknown pose source: {source}')
    points = gather_pts3d(observations, rigs)
    if not torch.isfinite(points).all():
        raise ValueError(f'Invalid initial points from {source}')
    return {'rigs': pack_rigs(rigs), 'points': points.cpu()}


def fit_depth_coefficients(rigs, data, points):
    """Fit only per-image log scale to the frozen geometry; no camera/point update."""
    z = torch.zeros(data.n_imgs, data.pix2d_dep.shape[-1], device=points.device)
    z[:, 0] = 1
    for image, selection in enumerate(data.img_slices):
        xyz = points[data.pids[selection].long()]
        P = rigs.get_w2cam(image)
        depths = (xyz @ P[:3, :3].T + P[:3, 3])[:, 2]
        base = data.pix2d_dep[selection, 0]
        good = torch.isfinite(depths) & (depths > 1e-3) & torch.isfinite(base)
        if good.any():
            z[image, 1] = (depths[good].log() - base[good]).median()
    return z


def observation_data(predictions, tracks, residual):
    from .tracks import gather_data_from_dense_tracks
    if residual == 'none':
        # No depth values or adjustable coefficients contribute to 2D-only BA.
        frames = [{'log_multidepth': torch.zeros(*xyz.shape[:2], 2)} for xyz in predictions['pts3d']]
    else:
        frames = make_depth_views(predictions, 'pca_basis' if residual == 'vggt' else 'multidepth')
    return gather_data_from_dense_tracks(frames, tracks)


def run_experiment(experiment, initial, tracks, native, joint, device='cuda', max_iters=100,
                   weight_z=1e-4, huber_delta=.5, min_loss_delta=1e-4):
    """Every row starts from a deep copy of its explicitly selected initial K/P/X."""
    if 'error' in initial:
        raise ValueError(f'Initialization failed for {experiment.pose_source}: {initial["error"]}')
    rigs = unpack_rigs(initial['rigs'])
    points = initial['points'].clone()
    if not experiment.ba:
        return rigs, points
    from .engine import bundle_adjustment
    predictions = joint if experiment.depth_residual == 'vggt' else native
    data = observation_data(predictions, tracks, experiment.depth_residual).to(device)
    rigs = rigs.to(device)
    points = points.to(device)
    if experiment.depth_residual == 'none':
        coefficients = torch.zeros(data.n_imgs, 2, device=device)
        coefficients[:, 0] = 1
    else:
        if data.pix2d_dep.shape[-1] != 17:
            raise ValueError('Depth BA requires fixed base + bias + 15 corrective channels')
        coefficients = fit_depth_coefficients(rigs, data, points)
    # Fix the first pose to select a gauge, consistently across all BA variants.
    pose_mask = torch.ones(rigs.n_nodes, dtype=torch.bool, device=device)
    pose_mask[0] = False
    return bundle_adjustment(
        rigs, data, points, z_coefs=coefficients, device=device, max_iters=max_iters,
        pnorm=1, weight_z=0 if experiment.depth_residual == 'none' else weight_z,
        optim_K=True, optim_P=pose_mask, optim_X=True,
        optim_Z=experiment.depth_residual != 'none', optim_Z0=False, n_zcfs=None,
        tangent_reg=0, huber_delta=huber_delta, min_loss_delta=min_loss_delta)


def pose_predictions(rigs, names):
    return {'views': {name: {'K': rigs.get_K(i).detach().cpu().numpy(),
                             'cam2w': rigs.get_cam2w(i).detach().cpu().numpy()}
                      for i, name in enumerate(names)}}
