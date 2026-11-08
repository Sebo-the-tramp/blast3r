# Benchmarks

Reproducing the published results. None of this is needed to reconstruct your
own scenes; see the [README](../README.md) for that.

BLASt3R is evaluated on ETH3D and CO3D v2 offline, and on TUM-RGBD and
ETH3D-SLAM online. Each is rebuildable from its official download.

## Preparing the data

The two offline benchmarks need their download converted into the layout the
evaluation reads: one directory per scene, holding the evaluated images and a
`gt.pkl`. The images are symlinked, so this costs no disk space beyond the
ground truth.

```bash
# ETH3D: the official high-resolution multi-view download
python scripts/preprocessing/prepare_eth3d.py \
  --eth3d_root /path/to/ETH3D --out data/eth3d

# CO3D v2: the official download, 41 seen categories
python scripts/preprocessing/prepare_co3d.py \
  --co3d_root /path/to/co3d --out data/co3d
```

CO3D is evaluated on ten frames per scene, drawn by an RNG seeded from the scene
name, and the prepared directories record which frames those are. The scene set
is the `fewview_dev` test sequences of the 41 seen categories whose
`viewpoint_quality_score` exceeds `--min_quality` (0.5), 2080 sequences in all.

The two online benchmarks are used as distributed, with no preparation step.

## Running a benchmark

One script per benchmark, each running inference over every scene and then
evaluating. They take the data directory and an output directory:

```bash
scripts/xps/eth3d_offline.sh     data/eth3d          out/eth3d
scripts/xps/co3d_offline.sh      data/co3d           out/co3d
scripts/xps/tum_rgbd_online.sh   /path/to/TUM_RGBD   out/tum
scripts/xps/eth3d_slam_online.sh /path/to/ETH3D_slam out/eth3d_slam
```

The released checkpoints download from the Hugging Face Hub on first use. To
evaluate your own instead, pass them as two further arguments:
`<script> <data_dir> <output_dir> <matcher_ckpt> <depth_ckpt>`.

All four are safe to re-run: finished scenes are skipped. Each states its own
settings in a header comment, including the bundle-adjustment schedule, which
differs between benchmarks.

On ETH3D, averaged over the 13 scenes:

| Metric | Value |
|---|---|
| AUC@30 | 98.5 |
| RRA@5 | 100.0 |
| RTA@5 | 97.9 |

Expect small per-scene variation between machines. The bundle adjustment stops
when the loss stops decreasing, and that test can fall either side of the
threshold depending on the GPU and the CUDA build, moving individual scenes by a
few tenths of a point.

## Evaluating on your own

The benchmark scripts call the two evaluation scripts, which also run
standalone. Both print a table and write a CSV.

**Relative pose**, for offline reconstructions. Ground truth is one directory
per scene holding a `gt.pkl` with `poses` (Nx4x4 camera-to-world) and
`filenames`, as the preprocessing scripts write:

```bash
python scripts/eval_offline.py --gt_dir /path/to/gt --preds_dir /path/to/output
```

It reports AUC@30, RRA@5 and RTA@5 per scene plus an average. A scene with no
prediction is scored as zero and reported as such, as in the paper.

**Trajectory error**, for online reconstructions on TUM-RGBD or ETH3D-SLAM,
which share a file format:

```bash
python scripts/eval_online.py --data_root /path/to/TUM_RGBD --preds_dir /path/to/output \
  --benchmark tum_rgbd --interpolate
```

It reports absolute trajectory error after Sim(3) alignment, since monocular
reconstruction fixes scale only up to a constant. `--benchmark` selects the
published sequence list (`tum_rgbd` or `eth3d_slam`); `--sequences` overrides it.

`--frame_step` reconstructs a subset of the frames, so the trajectory is sampled
more coarsely than the ground truth. `--interpolate`, which the published
numbers use, interpolates it onto every ground-truth frame before measuring the
error. The alternative, `--allow_subset`, measures only the reconstructed frames
and reports the fraction covered.

## Local VGGT comparison scripts

`run.sh` contains the Bicycle/Courtyard reconstruction examples; `eval.sh`
contains the ETH3D comparison commands; `cache_eth3d.sh` builds their frozen
10-image dataset and VGGT caches. All load `scripts/experiment_paths.sh`
and run from the BLASt3R checkout, so they can be invoked from any directory:

```bash
bash third_party/blast3r/run.sh
bash third_party/blast3r/eval.sh
```

Configure paths with environment variables or create `experiments.local.sh` in
this checkout (gitignored). To use a different shell config file, set
`BLAST3R_CONFIG=/absolute/path/to/config.sh`. Config file assignments take
precedence over inherited environment variables; defaults fill unset or empty
values. Use `${VARIABLE:-default}` assignments in local config if you want to
preserve environment overrides. Quote paths containing spaces.

| Variable | Default on Berzelius |
|---|---|
| `BLAST3R_DATA_ROOT` | `/proj/vlarsson/datasets` |
| `BLAST3R_CHECKPOINT_ROOT` | `/proj/vlarsson/checkpoints` |
| `BLAST3R_VGGT_CHECKPOINT` | `$BLAST3R_CHECKPOINT_ROOT/vggt_omega_1b_512.pt` |
| `BLAST3R_MIP360_ROOT` | `$BLAST3R_DATA_ROOT/mip360` |
| `BLAST3R_ETH3D_ROOT` | `$BLAST3R_DATA_ROOT/eth3d-eval/eth3d_processed` |
| `BLAST3R_BICYCLE_IMAGES` | `$BLAST3R_MIP360_ROOT/bicycle/images_2` |
| `BLAST3R_COURTYARD_IMAGES` | `$BLAST3R_ETH3D_ROOT/courtyard/undistorted_images` if present; otherwise `courtyard/images/dslr_images_undistorted` under the same root |
| `BLAST3R_CACHE_ROOT` | `<checkout>/.cache` (exported for Python inference) |
| `BLAST3R_ETH3D_10` | `$BLAST3R_CACHE_ROOT/eth3d_10_seed0/scenes` |
| `BLAST3R_OUTPUT_ROOT` | `<checkout>/out` |
| `BLAST3R_EVAL_ROOT` | `<checkout>/outputs/evaluation/eth3d` |
| `BLAST3R_ETH3D_GT` | `$BLAST3R_EVAL_ROOT/gt` |
| `BLAST3R_PYTHON` | `<checkout>/.venv/bin/python` if executable, otherwise `python` from the active environment |

For Sebastian's machine, put these assignments in `experiments.local.sh`:

```bash
BLAST3R_DATA_ROOT=/home/se7882ca/Documents/scsv/datasets
BLAST3R_ETH3D_ROOT="$BLAST3R_DATA_ROOT/eth3d/multi_view_training_dslr_undistorted"
BLAST3R_VGGT_CHECKPOINT=/fast-storage/huggingface/hub/models--facebook--VGGT-Omega/snapshots/1041e80fc0e911235d3426b0a3d9a81075111a53/vggt_omega_1b_512.pt
```

Or override a path for one invocation:

```bash
BLAST3R_PYTHON=/path/to/env/bin/python bash run.sh
BLAST3R_EVAL_ROOT=/path/to/results bash eval.sh
```

These settings configure paths only. The reconstruction blocks in `eval.sh`
remain commented; its active commands evaluate existing predictions. `run.sh`
currently runs the Bicycle VGGT example.

`prepare_eth3d.py` and the cache builder accept both the official COLMAP layout
and the Berzelius processed layout (`undistorted_images/` and per-image
camera-from-world `poses/*.npy`). Processed poses are inverted for evaluation.

## Frozen 10-image ETH3D cache

From the BLASt3R checkout, with a Python environment containing PyTorch,
torchvision, the image-loading dependencies and VGGT-Omega:

```bash
bash cache_eth3d.sh
```

The builder uses the existing scene-name-dependent sampling algorithm with
`--seed 0`, selecting exactly 10 images for each of the 13 training scenes.
These are the same selections as running `--max_frames 10 --seed 0` on the
full prepared scene directories. Each selected image is copied once, so future
experiments read a frozen local dataset rather than select from a changing
source directory. Re-running resumes existing caches; a changed selection or
image content raises an error instead of silently replacing the experiment.
Use another `BLAST3R_ETH3D_10` directory for a different selection.

The default artifacts (all gitignored) are:

```text
.cache/
  eth3d_10_seed0/scenes/<scene>/
    <10 original image files>
    selection.json          # seed, source paths, filenames, SHA-256 hashes
    gt.pkl                  # selected camera-to-world GT, for evaluation only
    preprocessed.pth        # normalized/tokenized images at size 512
    vggt_cache.json          # filename-to-feature-cache mapping; paths relative to scene
  vggt_monodepth/<hash>.pth  # raw features, depth, confidence, intrinsics, PCA
```

Each VGGT cache entry retains the full **256-channel input to `dense_head.proj`**,
base log depth, confidence, pointmap, predicted intrinsics, and the whitened
PCA-15 correction basis. Raw features remain available for other PCA ranks.
PCA is fitted independently per image without GT. Neither GT intrinsics nor
poses are provided to inference; `gt.pkl` is read only by evaluation.

Offline image loading reuses `preprocessed.pth` when filenames, sizes, mtimes,
image size, crop aspect ratio and patch size match. Other preprocessing settings
fall back to normal image loading. VGGT cache keys include the processed image
content, supplied intrinsics if any, and checkpoint identity (resolved filename,
size and mtime). PCA-15 has its own version marker. Cache writes are atomic.
Warm VGGT caches skip checkpoint loading and network inference; PCA refinement
also skips the eigendecomposition.

The three 10-image comparison commands in `eval.sh` use
`BLAST3R_ETH3D_10` for both inference and GT. The full-scene examples continue
to use `BLAST3R_ETH3D_GT`. Standard BLAST3R reuses cached image preprocessing;
VGGT scale-only and PCA-15 runs also reuse the depth/feature predictions.

For a different machine, override the paths described above. To prepare only
images, GT and preprocessing without running VGGT:

```bash
bash cache_eth3d.sh --prepare_only
```

To cache one scene first:

```bash
bash cache_eth3d.sh --scenes courtyard
```

On this Berzelius checkout, the initial build used the existing EPO environment
and the VGGT-Omega source already present in `third_party/epo`. From the parent
SC3R repository, run on an allocated GPU node:

```bash
export PYTHONPATH="$PWD/third_party/blast3r/src:$PWD/third_party/epo/third_party/vggt-omega${PYTHONPATH:+:$PYTHONPATH}"
BLAST3R_PYTHON=/home/x_luddi/.conda/envs/epo/bin/python \
  bash third_party/blast3r/cache_eth3d.sh
```

This cache-only build does not require compiling BLAST3R's CUDA extensions or
installing its matcher/retrieval dependencies. Those remain necessary for full
reconstruction experiments.

## Eight-row pose initialization and depth BA experiment

`run_pose_matrix.sh` implements the eight rows of the pose-source table on the
frozen ten-image scenes. From the parent SC3R root:

```bash
bash third_party/blast3r/run_pose_matrix.sh --dry_run
# On an allocated GPU node:
bash third_party/blast3r/run_pose_matrix.sh
# Small integration run; keep these results in a separate output directory:
bash third_party/blast3r/run_pose_matrix.sh --scenes courtyard --max_iters 2 \
  --output outputs/evaluation/eth3d/pose_matrix_smoke
```

A batch launcher is supplied in the parent repository:
`sbatch jobs/job_blast3r_pose_matrix.sh` (submit from the SC3R root).
It requests one 40 GB A100 (`thin`), 8 CPU cores, 48 GB of **host RAM**, and
one hour. It runs all eight rows over all thirteen frozen scenes, continues
past explicitly recorded scene failures, and writes logs to
`<SC3R>/logs/blast3r_pose_matrix-<jobid>.out` and `.err`. It uses an absolute
working directory, so the submitted job remains independent of the interactive
allocation. Resubmitting the same command skips completed rows. To retry failed
rows as well, pass `--rerun` (this also reruns successful rows).

A measured Courtyard run on an A100 80 GB, with the full 100-iteration cap for
all eight rows, took **57.5 seconds**. Native depth, matching and initial states
were recomputed, while the already-built joint VGGT cache was reused. Peak
PyTorch allocated memory was **5.37 GiB**, and peak reserved memory **7.94 GiB**;
CUDA context and other driver allocations are additional. This is a measured
single-scene example, not a guarantee for every scene. Budget roughly **15–30
minutes for the thirteen-scene matrix**, excluding queue time; the one-hour
request provides margin. A 40 GB A100 is the recommended batch choice for this
cached ten-image, size-512 protocol. These measurements do not establish a
minimum VRAM requirement or cover larger image sets/resolutions. Request an
80 GB node explicitly if desired:

```bash
sbatch --constraint=fat jobs/job_blast3r_pose_matrix.sh
```

Override the wall time at submission if needed, e.g.
`sbatch --time=02:00:00 jobs/job_blast3r_pose_matrix.sh`.
The runner uses `<checkout>/.venv/bin/python` when present. All paths remain
configurable through `scripts/experiment_paths.sh` and the CLI. The additional
settings are `BLAST3R_MATCH_CHECKPOINT` (`naver/blast3r-matcher`),
`BLAST3R_DEPTH_CHECKPOINT` (`naver/blast3r-depth`), and `BLAST3R_MATRIX_OUTPUT`
(`<checkout>/outputs/evaluation/eth3d/pose_matrix`). Hugging Face checkpoints
are downloaded under `BLAST3R_CACHE_ROOT/huggingface` unless `HF_HOME` is set.

| Pose initialization | No BA | 2D-only BA | BLASt3R depth BA | VGGT feature depth BA |
|---|---|---|---|---|
| Joint VGGT-Omega | `vggt_no_ba` | `vggt_ba_none` | `vggt_ba_blast3r` | `vggt_ba_vggt` |
| BLASt3R tree (depth matched to BA) | `blast3r_no_ba` | `blast3r_ba_none` | `blast3r_ba_blast3r` | `blast3r_ba_vggt` |

Protocol:

- VGGT-Omega runs **jointly on all ten images**. Its predicted poses initialize
  the VGGT rows, and its jointly predicted depths/features supply the VGGT BA
  residuals in both pose-source groups. This is a separate cache from the older
  per-image monocular VGGT cache, which cannot supply multi-view poses.
- BLASt3R initialization uses matched pixels, relative similarity alignment and
  a maximum spanning tree, including scale propagation. **Its pointmaps follow
  the BA depth basis:** native BLASt3R for `blast3r_ba_blast3r`, joint VGGT for
  `blast3r_ba_vggt`. The no-BA and 2D-only BA rows use native BLASt3R as their
  baseline (`--baseline_init_depth vggt` changes that explicit baseline choice).
  `--blast3r_init_depth blast3r` or `vggt` can override the depth-matched policy
  for a separate experiment; the requested default is `residual`.
- Every row uses the **same cached BLASt3R tracks**, with all image pairs enabled
  for these ten-image scenes. GT is used only when computing metrics.
- Initial states are cached separately for joint VGGT poses, the native-depth
  BLASt3R tree, and the VGGT-depth BLASt3R tree. Sparse points are initialized by
  depth backprojection and averaging using the selected initializer. Each row
  receives a fresh copy of its selected K/poses/points. The VGGT-pose rows share
  their initial state; **the two BLASt3R depth-BA rows deliberately start from
  different states**, because initialization depths follow the residual basis.
  Their comparison measures initialization plus BA together, rather than an
  isolated basis change. Every result records `initial_depth_source` and the
  exact initial-state cache used.
- The default shares one intrinsics matrix per scene, initialized with square
  pixels and the median focal from the selected initializer.
  `--independent_intrinsics` keeps separate matrices. The existing solver
  optimizes `fx` and `fy` separately. All BA variants optimize intrinsics,
  poses and sparse points; the first
  camera pose is held fixed to select the coordinate frame.
- 2D-only BA disables depth coefficients and uses exactly zero depth weight.
  The solver preserves zero weights rather than clamping them back on.
- Native depth BA uses the fixed base log depth, a scale/bias, and 15 learned
  correction channels. VGGT depth BA uses the fixed joint base log depth,
  a scale/bias, and 15 per-image whitened PCA feature maps. Each therefore has
  16 adjustable depth parameters per image. Only the initial log-scale is fitted
  to the frozen geometry; the 15 shape coefficients start at zero.
- Default BA: one pose-refinement phase, at most 100 LM iterations, depth
  objective weight `1e-4`, Huber delta `0.5`, minimum loss decrease `1e-4`.
  There is no extra fixed-pose surface-stitching phase in this pose table.
- Default AUC is the **exact normalized integral of empirical recall** for the
  maximum of relative rotation and translation-direction errors, at 1, 5 and
  30 degrees. Means weight the scenes equally (45 pairs per ten-image scene).
  `--auc_mode histogram` reproduces the older evaluator's integer-degree-bin
  approximation; at one degree that approximation is simply recall below 1°.
  The legacy evaluator retains its original default. Translation-angle
  conventions are shared with the existing evaluator.

The output has `config.json`, `per_scene.csv`, `mean_auc.csv`, `mean_auc.tex`
and `summary.json`, plus `<row>/<scene>.pth` and `<row>/<scene>.json`. Partial
scene sets state their denominator explicitly. Means remain blank while any
requested scene is pending; explicit failures score zero and retain their error
messages. `--continue_on_error` records failures and continues; the default stops
on the first error. Existing rows are skipped; `--rerun` retries/replaces them.
Results from another configuration are rejected in the same output directory.

Native depth predictions, tracks, shared initial states and matching-support graphs
are cached under `BLAST3R_CACHE_ROOT/pose_matrix/<scene>/<input-identity>/`.
The cache identity includes input pixels/order, checkpoint identities, seed,
intrinsics convention and implementation identity. Iteration counts and depth
weights can change while reusing inference caches in a new output directory.
`vggt_ba_vggt` and `blast3r_ba_vggt` share the same joint VGGT feature prediction,
stored separately under `BLAST3R_CACHE_ROOT/vggt_joint/<sequence-identity>.pth`.
Joint feature capture concatenates all dense-head chunks (the Omega head decodes
eight frames at a time), preserving all ten frames and their order.
`match_graph.json` reports shared-track counts as a diagnostic for weak matching
support; these counts are not a measurement of ground-truth visual overlap.

Use `--rows vggt_no_ba blast3r_no_ba` to run only initial-pose baselines, or
`--report_only` to rebuild tables without running networks. If the uniformly
sampled scenes show poor VGGT initialization, evaluate a separately identified
high-overlap subset; keep its selection and table separate from this experiment.

The generated table uses `booktabs`, `amssymb`, and seven columns (`l ccc ccc`).

Before skipping existing results, the runner verifies every frozen image against
its selection SHA-256. The output protocol also records the selected image
identities and Python/PyTorch/CUDA versions, preventing stale results from being
reused after a dataset or runtime change.
