# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An anomaly-detection prototype for Cubii, a stainless-steel packaging-line
manufacturer, meant to visually flag defective containers on a production
line. Real Cubii factory images aren't available yet, so the pipeline
currently trains/evaluates on **PKU-GoodsAD's `food_box` category** (public
dataset, github.com/jianzhang96/GoodsAD) as a stand-in — real product-box
photos with labeled physical defects (`deformation`, `opened`,
`surface_damage`). The code is written so the exact same training path works
unmodified once real Cubii images exist (see README.md "Path to real Cubii
data").

Model: **PatchCore** (via `anomalib==2.6.0`) — an ImageNet-pretrained,
frozen `wide_resnet50_2` backbone used purely as a feature extractor (no
gradient descent, no fine-tuning). "Training" means: run every good/normal
training image through the backbone once, collect patch features, and
subsample them into a memory bank. Inference scores a new image by how far
its patch features are from anything in that memory bank. This is why
training is CPU-feasible (slow, but feasible) and doesn't require labeled
defect examples.

## Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train (current active path — see "Two training entry points" below)
python src/train_folder.py \
  --normal-dir data/GoodsAD/food_box/train/good \
  --abnormal-dir data/GoodsAD_merged/food_box/abnormal \
  --mask-dir data/GoodsAD_merged/food_box/masks \
  --normal-test-dir data/GoodsAD/food_box/test/good \
  --category food_box

# Inference on one image (auto-finds the newest checkpoint under results/ if --ckpt omitted)
python src/infer.py --image path/to/image.png
```

There is no lint config or test suite in this repo currently — don't assume
`pytest`/`ruff`/etc. are set up unless you add them yourself.

## Architecture

### Two training entry points — use `train_folder.py`, not `train.py`, for the current dataset

- **`src/train.py`** — config-driven (`configs/patchcore.yaml`), uses
  anomalib's official `MVTecAD`/`MVTecLOCO` datamodules, which expect the
  strict MVTec folder layout (`train/good`, `test/<defect_type>/`,
  `ground_truth/<defect_type>/`, one subfolder per defect type).
- **`src/train_folder.py`** — CLI-arg driven, uses anomalib's generic
  `Folder` datamodule: one flat `--abnormal-dir` + one flat `--mask-dir`
  instead of per-defect-type subfolders. This is also the intended long-term
  path once real Cubii data arrives (no MVTec-specific structure needed).

**`train.py` is currently broken for the GoodsAD `food_box` dataset**: anomalib's
`make_mvtec_ad_dataset` throws `ValueError: Must have equal len keys and
value when setting with an iterable` (a pandas/anomalib internal bug —
confirmed not a data issue: file counts and filename pairing were verified
exactly correct, and the identical code path works fine when run standalone
outside the library's own function call). Don't spend time re-debugging this
unless you have a specific new lead — the workaround (flatten the dataset
and use `train_folder.py`) is already in place and documented below.

Because of that bug, `data/GoodsAD_merged/food_box/{abnormal,masks}/` exists
as a manually flattened copy of the original `data/GoodsAD/food_box/{test,ground_truth}/<defect_type>/`
folders — same images, just reorganized (defect-type-prefixed filenames) so
`Folder` can consume them as two flat dirs instead of three per split. No
image content was altered. Don't regenerate this by hand again if it's
already present; if you need to rebuild it, filenames between `abnormal/`
and `masks/` must match exactly (same stem, matching mask by name).

### `src/backbone_cache.py` — must be imported and called before other anomalib imports

`patch_timm_for_local_weights()` monkeypatches `timm.create_model` so
pretrained backbone weights load from a locally cached file
(`weights/wide_resnet50_racm-8234f177.pth`, fetched from a GitHub release)
instead of Hugging Face Hub, which is unreachable in some sandboxed/offline
environments. This only works if it's called **before** anomalib's
timm-backed feature extractors are imported — see the import order in
`train.py`/`train_folder.py`/`infer.py` (`patch_timm_for_local_weights()` is
called immediately after importing it, before `from anomalib.models import
Patchcore`). Preserve that order if you touch these files.

### Known hyperparameter findings (don't re-run these experiments blind)

Baseline (`--coreset-sampling-ratio 0.1` default, `layer2`+`layer3`):
image_AUROC 0.8027, image_F1 0.80, pixel_AUROC 0.9746, pixel_F1 0.3482.

- Raising `--coreset-sampling-ratio` to 0.25 did **not** improve results
  (image_AUROC dropped slightly to 0.7825) and took ~3x longer to train —
  not worth it at this dataset scale.
- Adding `layer1` to `--layers` (in addition to `layer2`+`layer3`) caused an
  **out-of-memory crash** during coreset selection — `layer1`'s much higher
  spatial resolution blows up the memory needed for the greedy coreset
  selection step. Don't add `layer1` without significantly more RAM
  available than a typical CPU dev machine.
- `layer2`+`layer3` is also the standard PatchCore paper default, not an
  arbitrary choice — deeper (`layer4`) is too semantically abstract for
  fine-grained defect localization; shallower (`layer1`) is expensive as above.

### Data directories are gitignored

`data/`, `results/`, `venv/`, and model weight files (`*.ckpt`, `*.pt`,
`*.pth`) are all gitignored — they're large and either regenerable or
publicly downloadable, not project source. Trained checkpoints
(`results/Patchcore/<category>/v*/weights/lightning/model.ckpt`, ~355MB) are
not committed; regenerate via the training command above, or transfer the
file out-of-band (e.g. a GitHub Release asset) if an exact copy is needed.
