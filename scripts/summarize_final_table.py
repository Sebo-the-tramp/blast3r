#!/usr/bin/env python3
"""Final ETH3D / ScanNet++ / AerialMegaDepth ablation table from jobs/job_blast3r_final416.sh.

    PYTHONPATH=src .venv/bin/python scripts/summarize_final_table.py [out/final_416]

Reads <root>/<dataset>/<cfg>/seed_0/c*/ (finished chunks have a DONE file): pose = combined
relative-pose recall at 1 degree (Auc_1), dense F1 at the table's four thresholds, time = mean
total reconstruction time per scene (warmup scene excluded, so no model loading). Means are over
all expected scenes; a failed scene scores zero (time too), as in the paper convention, and is
reported. A configuration with an unfinished chunk is pending (--). Writes final_table.tex and
final_table.md to <root>.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROWS = [('No', 'no_ba'), ('2D', 'ba_2d'), ('3D (fixed depth)', 'ba_fixed_depth'),
        (r'3D ($\alpha_{n,1},\beta_n$ free)', 'ba_alpha_beta'), ('3D (full depth)', 'ba_full'),
        ('$r=0$', 'vggt_alpha_beta'), ('$r=1$', 'vggt_rank1'), ('$r=4$', 'vggt_rank4'),
        ('$r=8$', 'vggt_rank8'), (r'$\rho=0.95$', 'vggt_rho95'), ('$r=16$', 'vggt_rank15'),
        ('$r=32$', 'vggt_rank32'), (r'$r=\mathrm{full}$', 'vggt_rank256')]
N_BA = 5  # rows in the "BLASt3R / BA type" group
DATASETS = {
    'eth3d': ('ETH3D', 'data/eth3d_fullres', ('f1_1cm', 'f1_2cm', 'f1_5cm', 'f1_20cm'), 'eth3d_pointcloud.csv'),
    'scannetpp': ('ScanNet++', 'data/scannetpp', (0.01, 0.02, 0.05, 0.2), None),
    'aerialmegadepth': ('AerialMegaDepth', 'data/aerialmegadepth',
                        ('f1_50cm', 'f1_100cm', 'f1_200cm', 'f1_500cm'), 'amd_pointcloud.csv'),
}


def scene_metrics(chunk, dataset, f1_keys, f1_file):
    """{scene: [pose, f1 x4, time]} for the scenes evaluated in one finished chunk."""
    timing = json.loads((chunk / 'timing.json').read_text())
    out = {}
    if dataset == 'scannetpp':
        for path in chunk.glob('*_scannetpp_metrics.json'):
            m = json.loads(path.read_text())
            tol = [round(t, 6) for t in m['cloud']['tolerances_m']]
            f1 = [100 * m['cloud']['f1_score'][tol.index(t)] for t in f1_keys]
            out[m['scene']] = [m['pose']['Auc_1'], *f1, timing[m['scene']]['total']]
        return out
    pose = pd.read_csv(chunk / 'relpose.csv', index_col='scene').drop(index='AVG', errors='ignore')
    cloud = pd.read_csv(chunk / f1_file, index_col='scene').drop(index='AVG', errors='ignore')
    for scene in pose.index:
        if pose.loc[scene].get('status', 'ok') != 'ok' or scene not in cloud.index or scene not in timing:
            continue
        out[scene] = [pose.loc[scene, 'Auc_1'], *cloud.loc[scene, list(f1_keys)], timing[scene]['total']]
    return out


def collect(root):
    table, notes = {}, []
    for dataset, (name, data_dir, f1_keys, f1_file) in DATASETS.items():
        expected = sorted(p.name for p in Path(data_dir).iterdir() if (p / 'gt.pkl').is_file())
        tasks = [line.split() for line in (root / f'tasks_{dataset}.txt').read_text().splitlines()]
        for _, cfg in ROWS:
            chunks = [root / dataset / cfg / 'seed_0' / f'c{k}' for c, k, *_ in tasks if c == cfg]
            if not chunks or not all((c / 'DONE').is_file() for c in chunks):
                table[dataset, cfg] = None
                continue
            per_scene = {}
            for chunk in chunks:
                per_scene.update(scene_metrics(chunk, dataset, f1_keys, f1_file))
            failed = [s for s in expected if s not in per_scene]
            if failed:
                notes.append(f'{name} {cfg}: {len(failed)} failed scene(s) scored zero: {failed}')
            values = np.array([per_scene.get(s, [0.0] * 6) for s in expected], dtype=float)
            table[dataset, cfg] = values.mean(0)
    return table, notes


def marks(column, lower_is_better):
    """\\best / \\better on the unrounded values (as the paper table); exact ties share the mark."""
    shown = {k: v for k, v in column.items() if v is not None}
    distinct = sorted(set(shown.values()), reverse=not lower_is_better)
    best = distinct[0] if distinct else None
    better = distinct[1] if len(distinct) > 1 else None
    return {k: 'best' if v == best else 'better' if v == better else '' for k, v in shown.items()}


def render(table):
    cells = {}
    for dataset in DATASETS:
        for j in range(6):
            column = {cfg: (None if table[dataset, cfg] is None else table[dataset, cfg][j]) for _, cfg in ROWS}
            mark = marks(column, lower_is_better=(j == 5))
            for cfg, value in column.items():
                text = '--' if value is None else f'{value:.1f}'
                cells[dataset, cfg, j] = rf'\{mark[cfg]}{{{text}}}' if value is not None and mark[cfg] else text
    lines = []
    for i, (label, cfg) in enumerate(ROWS):
        if i == N_BA:
            lines += [r'\midrule', r'\multicolumn{19}{@{}l}{\textit{Ours}} \\', r'\addlinespace[2pt]']
        lines.append(' & '.join([label] + [cells[d, cfg, j] for d in DATASETS for j in range(6)]) + r' \\')
    return '\n'.join(lines)


HEADER = r"""\begin{table*}[t]
\centering
\caption{Point-cloud accuracy on ETH3D (mean over 13 scenes), ScanNet++ (mean over 20 scenes of
150 frames) and AerialMegaDepth (mean over 8 sets of 150 aerial images). All rows share BLASt3R
tracks. BLASt3R initialization follows the BA depth basis, using native depth for the no-depth
baselines; our rows use the VGGT-$\Omega$ 416 checkpoint. F1 is reported at the indicated
distance thresholds; on AerialMegaDepth against the GT MVS depth, with one camera per image since
the focal length varies between frames. Pose is combined relative-pose recall where the maximum
of rotation and translation-direction error is at most 1 degree. Time is the mean reconstruction
time per scene in seconds on one A100 (80\,GB), excluding model loading. Pending entries are
marked with --; failed scenes score zero. Best and second-best results in each column are
highlighted.}
\label{tab:eth3d_scannetpp}
\small
\setlength{\tabcolsep}{2pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}l *{6}{c} @{\hspace{10pt}} *{6}{c} @{\hspace{10pt}} *{6}{c}@{}}
\toprule
& \multicolumn{6}{c}{\textbf{ETH3D}}
& \multicolumn{6}{c}{\textbf{ScanNet++}}
& \multicolumn{6}{c}{\textbf{AerialMegaDepth}} \\
\cmidrule(lr){2-7} \cmidrule(lr){8-13} \cmidrule(lr){14-19}
& \textbf{Pose} & \multicolumn{4}{c}{\textbf{Depth F1} (\%)} &
& \textbf{Pose} & \multicolumn{4}{c}{\textbf{Depth F1} (\%)} &
& \textbf{Pose} & \multicolumn{4}{c}{\textbf{Depth F1} (\%)} & \\
\cmidrule(lr){2-2}  \cmidrule(lr){3-6}
\cmidrule(lr){8-8}  \cmidrule(lr){9-12}
\cmidrule(lr){14-14}  \cmidrule(lr){15-18}
\textbf{Method}
& 1$^\circ$ & 1\,cm & 2\,cm & 5\,cm & 20\,cm & \textbf{Time}
& 1$^\circ$ & 1\,cm & 2\,cm & 5\,cm & 20\,cm & \textbf{Time}
& 1$^\circ$ & 0.5\,m & 1\,m & 2\,m & 5\,m & \textbf{Time} \\
\midrule
\multicolumn{19}{@{}l}{\textit{BLASt3R / BA type}} \\
\addlinespace[2pt]
"""
FOOTER = r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'out/final_416')
    table, notes = collect(root)
    (root / 'final_table.tex').write_text(HEADER + render(table) + FOOTER)
    cols = ['Pose', 'F1a', 'F1b', 'F1c', 'F1d', 'Time']
    md = []
    for dataset, (name, *_) in DATASETS.items():
        df = pd.DataFrame({label: table[dataset, cfg] if table[dataset, cfg] is not None else [np.nan] * 6
                           for label, cfg in ROWS}, index=cols).T
        md.append(f'### {name}\n' + df.to_markdown(floatfmt='.1f'))
    text = '\n\n'.join(md) + ('\n\n' + '\n'.join(notes) if notes else '') + '\n'
    (root / 'final_table.md').write_text(text)
    print(text)
    print(f'wrote {root / "final_table.tex"}')


if __name__ == '__main__':
    main()
