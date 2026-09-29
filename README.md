# Cubii CV-QC — Container Defect Detection (Prototype)

## What this is

Cubii manufactures stainless steel packaging lines for liquid food products
(milk, juice, etc.). This prototype is an early step toward automated visual
quality control on the production line: a camera looks at each container
(bottle/carton) as it comes off the line, and the system flags anything that
looks physically wrong with the **container itself** — dents, cracks, broken
seals, contamination, label defects — not the liquid inside.

We don't have real factory images yet (the planned Cubii factory visit fell
through), so this prototype trains and evaluates on **PKU-GoodsAD**'s
`food_box` category (github.com/jianzhang96/GoodsAD) as a stand-in: real
photos of packaged retail products on store shelves, with labeled physical
defects — `deformation`, `opened` (not properly sealed — matches the
"container not closed" defect case), and `surface_damage`. This validates the
full pipeline (training, evaluation, inference) before real Cubii line images
are available. Earlier iterations of this prototype used MVTec AD "bottle"
and MVTec LOCO AD "juice_bottle"; `src/train.py` + `configs/patchcore.yaml`
still support those (`data.dataset: mvtec_ad` / `mvtec_loco`), but the current
default dataset/training path is GoodsAD `food_box` via `src/train_folder.py`
(see below).

## Why PatchCore / anomaly detection (not classification)

A normal image classifier needs labeled examples of every defect type to
learn from. On a real production line:

- Defects are rare, so you rarely have enough labeled defective images.
- New/unseen defect types will show up that you never trained on.

**PatchCore** (via [anomalib](https://github.com/open-edge-platform/anomalib))
sidesteps this: it trains **only on defect-free ("good") images**, builds a
memory bank of typical local patch features (from a pretrained CNN backbone),
and at inference time flags any image region whose features are far from
anything in that memory bank. This is well suited to QC because "normal" is
easy to collect in bulk, and "abnormal" can be anything.

## Project layout

```
.
├── configs/
│   └── patchcore.yaml       # Model/data/trainer configuration
├── data/
│   └── MVTec_LOCO/
│       └── juice_bottle/    # MVTec LOCO AD "juice_bottle" category (not committed — see below)
├── src/
│   ├── train.py             # Train PatchCore using configs/patchcore.yaml + MVTec-style data
│   ├── train_folder.py      # Train PatchCore on a plain folder of "good" images
│   │                        # (closer to what real Cubii line data will look like)
│   └── infer.py             # Run a trained model on a single image, print/save the result
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Core dependency: `anomalib==2.6.0` (built on PyTorch + PyTorch Lightning).

## Dataset

Download PKU-GoodsAD's `food_box` category from
github.com/jianzhang96/GoodsAD and extract it so you end up with:

```
data/GoodsAD/food_box/
├── train/
│   └── good/                             # 432 defect-free images — this is all PatchCore trains on
├── test/
│   ├── good/                             # held-out normal images, for evaluation
│   ├── deformation/ opened/ surface_damage/   # defective test images, one folder per defect type
└── ground_truth/
    └── deformation/ opened/ surface_damage/   # matching pixel-level defect masks (same filenames)
```

anomalib's official MVTec-format loader (`src/train.py`) currently crashes on
this dataset with a `pandas`/`anomalib` internal bug (`ValueError: Must have
equal len keys and value when setting with an iterable` inside
`make_mvtec_ad_dataset`) — confirmed not a data problem (file counts and
filename pairing verified exactly correct; identical code works when run
standalone outside the library's own function). Training therefore goes
through `src/train_folder.py` instead, against a manually flattened copy of
the three defect-type folders:

```
data/GoodsAD_merged/food_box/
├── abnormal/   # all defective test images from all 3 types, defect-type-prefixed filenames
└── masks/      # matching ground-truth masks, same filenames (different extension)
```

(No images were altered — only reorganized into these two flat folders to
work around the loader bug above.)

The dataset is **not committed to git** (see `.gitignore`) — it's ~1.7GB and
is a public benchmark dataset, not project source code.

## Training

```bash
source venv/bin/activate
python src/train_folder.py \
  --normal-dir data/GoodsAD/food_box/train/good \
  --abnormal-dir data/GoodsAD_merged/food_box/abnormal \
  --mask-dir data/GoodsAD_merged/food_box/masks \
  --normal-test-dir data/GoodsAD/food_box/test/good \
  --category food_box
```

This fits PatchCore (runs every "good" training image through the frozen,
pretrained `wide_resnet50_2` backbone once and builds a memory bank of
normal-image patch features — there's no gradient descent, so no GPU is
required, though it is CPU-slow: ~1.5 hours on this dataset), then evaluates
against the abnormal + held-out normal test images and prints image-level and
pixel-level AUROC/F1.

Outputs (checkpoint, metrics, visualizations) land under `results/Patchcore/food_box/`.

### Current result (baseline: `--coreset-sampling-ratio 0.1` [default], `layer2`+`layer3`)

| Metric | Score |
|---|---|
| image_AUROC | 0.8027 |
| image_F1Score | 0.8000 |
| pixel_AUROC | 0.9746 |
| pixel_F1Score | 0.3482 |

In plain terms: ~80% accurate at the normal-vs-defective call; ~97% accurate
at localizing *where* a defect is once flagged. Raising
`--coreset-sampling-ratio` to 0.25 was tried and did **not** improve results
(image_AUROC dropped slightly to 0.7825) at ~3x the training time — not
recommended. Adding `layer1` to the feature layers was also tried and caused
an out-of-memory crash (much higher-resolution feature maps blow up memory
during coreset selection) — do not add `layer1` without significantly more
RAM headroom.

The trained checkpoint for the result above is available as a GitHub Release
asset (see repo Releases) rather than committed to git, since it's ~355MB.

## Inference on a single image

```bash
source venv/bin/activate
python src/infer.py --image path/to/some_test_image.png
```

Prints the anomaly score and predicted label (normal/anomalous), and saves a
heatmap visualization showing *where* the model thinks the defect is. Uses
the most recently modified checkpoint under `results/` unless `--ckpt` is
given explicitly.

## Path to real Cubii data

Once real line images are available: collect a folder of defect-free
container images (`good/`), no defective examples strictly required, and
point `src/train_folder.py` at it the same way as above — it's already the
active training path and doesn't require the MVTec-specific
train/test/ground_truth folder structure.
