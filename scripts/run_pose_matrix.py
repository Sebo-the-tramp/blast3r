#!/usr/bin/env python3
"""Eight ETH3D pose-source x BA-depth experiments, with shared cached inputs."""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import time
import tempfile
import traceback

import torch
from blast3r.inference.experiment_matrix import EXPERIMENTS, initialization_key
from blast3r.inference.scene_cache import atomic_torch_save


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images_root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cache_root', default=os.environ.get('BLAST3R_CACHE_ROOT', '.cache'))
    p.add_argument('--ckpt_match', default='naver/blast3r-matcher')
    p.add_argument('--ckpt_depth', default='naver/blast3r-depth')
    p.add_argument('--ckpt_vggt', required=True)
    p.add_argument('--vggt_inference', choices=['joint', 'mono'], default='joint',
                   help='joint ten-image inference, or independent images (BLASt3R-pose rows only)')
    p.add_argument('--scenes', nargs='+')
    p.add_argument('--rows', nargs='+', choices=[e.name for e in EXPERIMENTS])
    p.add_argument('--blast3r_init_depth', choices=['residual', 'blast3r', 'vggt'], default='residual',
                   help='default: match tree initialization depths to the BA depth basis')
    p.add_argument('--baseline_init_depth', choices=['blast3r', 'vggt'], default='blast3r',
                   help='tree initializer for the no-BA and no-depth-BA rows')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_iters', type=int, default=100)
    p.add_argument('--weight_z', type=float, default=1e-4)
    p.add_argument('--huber_delta', type=float, default=.5)
    p.add_argument('--min_loss_delta', type=float, default=1e-4)
    p.add_argument('--auc_mode', choices=['exact', 'histogram'], default='exact')
    for option in ('independent_intrinsics', 'optim_memory', 'report_only', 'dry_run', 'continue_on_error', 'rerun'):
        p.add_argument('--' + option, action='store_true')
    return p


def available_experiments(vggt_inference):
    return tuple(e for e in EXPERIMENTS if vggt_inference == 'joint' or e.pose_source == 'blast3r')


def select_experiments(vggt_inference, rows=None):
    available = available_experiments(vggt_inference)
    if rows is not None and set(rows) - {e.name for e in available}:
        raise ValueError('Single-image VGGT cannot supply relative poses; select only blast3r_* rows')
    return [e for e in available if rows is None or e.name in rows]


def initial_cache_path(directory, key, vggt_inference):
    # Native initialization is identical in both modes; VGGT-depth initialization is not.
    suffix = '_mono' if vggt_inference == 'mono' and key == 'blast3r_vggt' else ''
    return Path(directory) / f'initial_{key}{suffix}.pth'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write('\n')
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checkpoint_identity(path):
    path = Path(path).resolve()
    files = [path] if path.is_file() else sorted(p for p in path.iterdir() if p.is_file())
    return [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in files]


def resolve_matrix_checkpoint(path):
    """Prefer already downloaded snapshots so compute nodes can run offline."""
    from blast3r.inference.loading import resolve_checkpoint
    if Path(path).is_dir():
        return Path(path)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        return Path(snapshot_download(path, local_files_only=True))
    except LocalEntryNotFoundError:
        return resolve_checkpoint(path)


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in (root / 'src/blast3r', root / 'scripts'):
        for p in sorted(directory.rglob('*.py')):
            digest.update(str(p.relative_to(root)).encode())
            digest.update(p.read_bytes())
    return digest.hexdigest()


def dataset_identity(root, scenes, seed):
    result = []
    for scene in scenes:
        directory = Path(root) / scene
        selection = json.loads((directory / 'selection.json').read_text())
        if selection['seed'] != seed or selection['max_frames'] != 10 or len(selection['images']) != 10:
            raise ValueError(f'{scene}: expected frozen seed {seed}, ten images')
        entries = []
        for entry in selection['images']:
            name = entry['name']
            if Path(name).name != name:
                raise ValueError(f'Invalid frozen filename: {name}')
            digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if digest != entry['sha256']:
                raise ValueError(f'Frozen image changed: {directory / name}')
            entries.append([name, digest])
        result.append({'scene': scene, 'images': entries})
    return result


def cached(path, build):
    if path.is_file():
        return torch.load(path, map_location='cpu', weights_only=True)
    value = build()
    atomic_torch_save(value, path)
    return value


def scene_identity(views):
    digest = hashlib.sha256()
    for path, image, shape in zip(views['instance'], views['img'], views['true_shape']):
        digest.update(Path(path).name.encode())
        digest.update(str(tuple(map(int, shape))).encode())
        digest.update(image.float().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare_scene(views, directory, models, mono_cam, seed, optim_memory, cache_root, vggt_inference='joint'):
    from blast3r.inference.loading import get_ntokens
    from blast3r.inference.multidepth import compute_pts3d_multidepths_blast3r
    from blast3r.inference.joint_vggt import compute_joint_vggt, compute_mono_vggt, joint_input_identity
    from blast3r.inference.experiment_matrix import initialize_source
    from blast3r.utils.device import todevice
    directory.mkdir(parents=True, exist_ok=True)
    def native_prediction():
        conf, xyz, basis, mode = compute_pts3d_multidepths_blast3r(models['native'], views)
        return todevice({'conf': conf, 'pts3d': xyz, 'multidepth': basis}, 'cpu')
    native = cached(directory / 'blast3r_depth.pth', native_prediction)
    if vggt_inference == 'mono':
        vggt = compute_mono_vggt(models['vggt'], views, cache_root)
    else:
        joint_key = joint_input_identity(views, models['vggt'].cache_identity)
        vggt = compute_joint_vggt(models['vggt'], views, Path(cache_root) / 'vggt_joint' / f'{joint_key}.pth')
    def match():
        from blast3r.inference.tracks import OnlineMatcher, compute_tracks
        torch.manual_seed(seed)
        n_tokens, patch = get_ntokens(views)
        matcher = OnlineMatcher(models['matcher'], len(views['img']), n_tokens, patch, 'cuda')
        return compute_tracks(matcher, todevice(views, 'cuda'), kpt_spacing=8,
                              remove_far_kpts=True, retrieval_mode='none',
                              num_retrieved_images=40, optim_memory=optim_memory).cpu()
    tracks = cached(directory / 'tracks.pth', match)
    visible = torch.isfinite(tracks).all(-1).float()
    counts = visible.T @ visible
    write_json(directory / 'match_graph.json',
               {'shared_track_counts': counts.int().tolist(), 'n_tracks': len(tracks),
                'description': 'Shared BLAST3R tracks per image pair; matching support, not GT overlap'})
    initial = {}
    sources = [('blast3r_blast3r', 'blast3r', native), ('blast3r_vggt', 'blast3r', vggt)]
    if vggt_inference == 'joint':
        sources.insert(0, ('vggt', 'vggt', vggt))
    for key, source, prediction in sources:
        try:
            initial[key] = cached(initial_cache_path(directory, key, vggt_inference),
                                     lambda: initialize_source(source, prediction, tracks, mono_cam))
        except Exception as error:
            # A disconnected BLASt3R tree must not invalidate VGGT pose baselines.
            initial[key] = {'error': f'{type(error).__name__}: {error}'}
    return native, vggt, tracks, initial


def evaluate(scene_dir, predictions, auc_mode):
    from blast3r.eval.datasets import read_scene_gt_poses
    from blast3r.eval.metrics.relpose import compute_pose_errors
    gt = read_scene_gt_poses(scene_dir)
    names = list(predictions['views'])
    if set(names) != set(gt) or len(names) != 10:
        raise ValueError('GT and prediction must have exactly the same ten filenames')
    gt_poses = torch.stack([torch.as_tensor(gt[n]).float() for n in names])
    pred_poses = torch.stack([torch.as_tensor(predictions['views'][n]['cam2w']).float() for n in names])
    metrics = compute_pose_errors(gt_poses, pred_poses, auc_mode=auc_mode)
    metrics = {key: value.item() if hasattr(value, 'item') else value for key, value in metrics.items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError('Non-finite relative-pose metrics')
    return metrics


def write_report(output, scenes, protocol):
    """Pending means stay blank; explicit failed scenes score zero."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    details, means = [], []
    for experiment in available_experiments(config.get('vggt_inference', 'joint')):
        rows = []
        key = initialization_key(experiment, config.get('blast3r_init_depth', 'residual'),
                                 config.get('baseline_init_depth', 'blast3r'))
        init_depth = 'vggt' if key == 'vggt' else key.removeprefix('blast3r_')
        for scene in scenes:
            path = output / experiment.name / f'{scene}.json'
            record = json.loads(path.read_text()) if path.is_file() else {}
            if record.get('protocol') != protocol:
                record = {}
            row = {'experiment': experiment.name, 'scene': scene,
                   'status': record.get('status', 'pending'), 'error': record.get('error', ''),
                   'initial_depth_source': record.get('initial_depth_source', init_depth)}
            row.update({f'Auc_{t}': 0.0 if row['status'] == 'failed' else record.get('metrics', {}).get(f'Auc_{t}')
                        for t in (1, 5, 30)})
            rows.append(row)
        details.extend(rows)
        complete = all(row['status'] in ('ok', 'failed') for row in rows)
        mean = {'experiment': experiment.name, 'pose_source': experiment.pose_source,
                'ba': experiment.ba, 'depth_residual': experiment.depth_residual,
                'initial_depth_source': init_depth,
                'n_ok': sum(r['status'] == 'ok' for r in rows),
                'n_failed': sum(r['status'] == 'failed' for r in rows),
                'n_pending': sum(r['status'] == 'pending' for r in rows),
                'n_scenes': len(scenes), 'complete': complete}
        mean.update({f'Auc_{t}': sum(r[f'Auc_{t}'] for r in rows) / len(rows) if complete else None for t in (1, 5, 30)})
        means.append(mean)
    for name, rows in (('per_scene.csv', details), ('mean_auc.csv', means)):
        with open(output / name, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / 'summary.json', {'protocol': protocol, 'scenes': scenes, 'rows': means})
    tex = [r'\begin{table*}[t]', r'\centering',
           r'\caption{Mean relative-pose AUC on ETH3D with 10 frozen images per scene (' + str(len(scenes)) +
           r' scenes). ' + ('Single-image' if config.get('vggt_inference') == 'mono' else 'Joint') +
           r' VGGT-$\Omega$ uses a rank-15 feature basis. All rows share BLASt3R tracks. '
           r'BLASt3R initialization follows the BA depth basis, using native depth for the no-depth baselines. '
           r'Pending entries are blank; failed scenes score zero.}',
           r'\label{tab:mean-auc}', r'\begin{tabular}{@{}l ccc ccc@{}}', r'\toprule',
           r'& \multicolumn{3}{c}{BA optimized variables} & \multicolumn{3}{c}{Mean AUC} \\',
           r'\cmidrule(lr){2-4} \cmidrule(lr){5-7}',
           r'Pose source & BA & BLASt3R depth & VGGT feat. depth & @1$^\circ$ & @5$^\circ$ & @30$^\circ$ \\', r'\midrule']
    mark = lambda yes: r'\ensuremath{\checkmark}' if yes else r'\ensuremath{\times}'
    for i, row in enumerate(means):
        if i > 0 and row['pose_source'] != means[i - 1]['pose_source']:
            tex.append(r'\midrule')
        cells = [r'VGGT-$\Omega$' if row['pose_source'] == 'vggt' else 'BLASt3R-matcher', mark(row['ba']),
                 mark(row['depth_residual'] == 'blast3r') if row['ba'] else '--',
                 mark(row['depth_residual'] == 'vggt') if row['ba'] else '--']
        cells += ['' if row[f'Auc_{t}'] is None else f"{row[f'Auc_{t}']:.2f}" for t in (1, 5, 30)]
        tex.append(' & '.join(cells) + r' \\')
    config_path = output / 'config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config.get('blast3r_init_depth', 'residual') != 'residual' or config.get('baseline_init_depth', 'blast3r') != 'blast3r':
            tex = [line.replace('BLASt3R initialization follows the BA depth basis, using native depth for the no-depth baselines.',
                               'BLASt3R tree initialization policy: ' + config.get('blast3r_init_depth', 'residual') +
                               '; no-depth baseline: ' + config.get('baseline_init_depth', 'blast3r') + '.') for line in tex]
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (output / 'mean_auc.tex').write_text('\n'.join(tex) + '\n')
    return means


def main():
    args = parser().parse_args()
    if args.max_iters < 1 or args.size < 16 or not 0 < args.weight_z <= 10:
        raise ValueError('Require positive BA iterations, size >=16 and depth weight in (0,10]')
    from preprocessing.prepare_eth3d import ETH3D_SCENES
    scenes = args.scenes or ETH3D_SCENES
    if not scenes or len(set(scenes)) != len(scenes) or any(Path(s).name != s for s in scenes):
        raise ValueError('Scene names must be unique directory names')
    selected = select_experiments(args.vggt_inference, args.rows)
    plan = {'scenes': scenes, 'images_per_scene': 10, 'seed': args.seed, 'vggt_inference': args.vggt_inference,
            'shared_tracks': 'blast3r-all-pairs', 'shared_vggt_initial_state': args.vggt_inference == 'joint', 'blast3r_initialization_matches_ba_depth': args.blast3r_init_depth == 'residual',
            'rows': [dict(name=e.name, pose_source=e.pose_source, ba=e.ba, depth_residual=e.depth_residual)
                     for e in available_experiments(args.vggt_inference)],
            'max_iters': args.max_iters, 'weight_z': args.weight_z, 'auc_mode': args.auc_mode,
            'mono_cam': not args.independent_intrinsics, 'blast3r_init_depth': args.blast3r_init_depth, 'baseline_init_depth': args.baseline_init_depth}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))))
    if args.report_only:
        config = json.loads((output / 'config.json').read_text())
        write_report(output, config['scenes'], config['protocol'])
        return
    from blast3r.inference.loading import load_depth_model, load_match_model, load_scene
    from blast3r.inference.experiment_matrix import pose_predictions, run_experiment
    dataset = dataset_identity(args.images_root, scenes, args.seed)
    native_path, matcher_path = resolve_matrix_checkpoint(args.ckpt_depth), resolve_matrix_checkpoint(args.ckpt_match)
    config = {**plan, 'dataset': dataset,
              'runtime': {'python': sys.version, 'torch': str(torch.__version__), 'cuda': torch.version.cuda},
              'size': args.size, 'huber_delta': args.huber_delta, 'min_loss_delta': args.min_loss_delta,
              'implementation': implementation_identity(),
              'checkpoints': {'blast3r': checkpoint_identity(native_path), 'matcher': checkpoint_identity(matcher_path),
                              'vggt': checkpoint_identity(args.ckpt_vggt)}}
    protocol = fingerprint(config)
    config['protocol'] = protocol
    config_file = output / 'config.json'
    if config_file.exists() and json.loads(config_file.read_text())['protocol'] != protocol:
        raise ValueError(f'Output configuration changed; choose another --output directory: {output}')
    write_json(config_file, config)
    write_report(output, scenes, protocol)
    models = {'native': load_depth_model(str(native_path), device='cuda', dtype=torch.float32),
              'vggt': load_depth_model(args.ckpt_vggt, device='cuda', backend='vggt'),
              'matcher': load_match_model(str(matcher_path), device='cuda', dtype=torch.float32)}
    for scene in scenes:
        directory = Path(args.images_root) / scene
        if all((output / e.name / f'{scene}.json').is_file() for e in selected) and not args.rerun:
            print(f'>> {scene}: requested rows already exist', flush=True)
            continue
        preparation_error = None
        try:
            manifest = json.loads((directory / 'selection.json').read_text())
            if manifest['seed'] != args.seed or manifest['max_frames'] != 10:
                raise ValueError('Frozen selection must match the requested seed and ten images')
            views = load_scene(directory, size=args.size)
            names = [Path(p).name for p in views['instance']]
            if len(names) != 10 or names != [image['name'] for image in manifest['images']]:
                raise ValueError('Frozen image names differ from selection manifest')
            infer_id = fingerprint({'inputs': scene_identity(views), 'seed': args.seed,
                                    'mono_cam': not args.independent_intrinsics, 'blast3r_init_depth': args.blast3r_init_depth,
                                    'checkpoints': config['checkpoints'], 'implementation': config['implementation'],
                                    'runtime': config['runtime']})
            cache = Path(args.cache_root) / 'pose_matrix' / scene / infer_id
            print(f'>> Preparing shared inputs for {scene}', flush=True)
            native, vggt, tracks, initial = prepare_scene(views, cache, models, not args.independent_intrinsics,
                                                       args.seed, args.optim_memory, args.cache_root, args.vggt_inference)
        except Exception as error:
            preparation_error = error
            traceback.print_exc()
        for experiment in selected:
            record_path = output / experiment.name / f'{scene}.json'
            if record_path.is_file() and not args.rerun:
                continue
            record_path.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            try:
                if preparation_error is not None:
                    raise preparation_error
                print(f'>> {scene}: {experiment.name}', flush=True)
                state_key = initialization_key(experiment, args.blast3r_init_depth, args.baseline_init_depth)
                rigs, points = run_experiment(experiment, initial[state_key], tracks, native, vggt,
                                              max_iters=args.max_iters, weight_z=args.weight_z,
                                              huber_delta=args.huber_delta, min_loss_delta=args.min_loss_delta)
                predictions = pose_predictions(rigs, names)
                metrics = evaluate(directory, predictions, args.auc_mode)
                state_path = str(initial_cache_path(cache, state_key, args.vggt_inference))
                predictions.update(experiment=experiment.name, protocol=protocol, initial_state_cache=state_path,
                                   vggt_inference=args.vggt_inference,
                                   initial_depth_source='vggt' if state_key == 'vggt' else state_key.removeprefix('blast3r_'))
                torch.save(predictions, record_path.with_suffix('.pth'))
                record = dict(status='ok', protocol=protocol, experiment=experiment.name, scene=scene,
                              vggt_inference=args.vggt_inference,
                              metrics=metrics, initial_state_cache=state_path,
                              initial_depth_source=predictions['initial_depth_source'], n_tracks=len(tracks),
                              n_images=len(names), seconds=time.perf_counter() - started)
                del rigs, points
            except Exception as error:
                record = dict(status='failed', protocol=protocol, experiment=experiment.name, scene=scene,
                              error=f'{type(error).__name__}: {error}')
                write_json(record_path, record)
                write_report(output, scenes, protocol)
                if not args.continue_on_error:
                    raise
                traceback.print_exc()
            write_json(record_path, record)
            write_report(output, scenes, protocol)
        if preparation_error is None:
            del views, native, vggt, tracks, initial
        torch.cuda.empty_cache()
    print(f'Wrote per_scene.csv, mean_auc.csv and mean_auc.tex to {output}', flush=True)


if __name__ == '__main__':
    main()
