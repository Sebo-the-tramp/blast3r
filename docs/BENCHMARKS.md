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
