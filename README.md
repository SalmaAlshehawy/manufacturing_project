# Cubii CV-QC — Container Defect Detection (Prototype)

## What this is

Cubii manufactures stainless steel packaging lines for liquid food products
(milk, juice, etc.). This prototype is an early step toward automated visual
quality control on the production line: a camera looks at each container
(bottle/carton) as it comes off the line, and the system flags anything that
looks physically wrong with the **container itself** — dents, cracks, broken
seals, contamination, label defects — not the liquid inside.

We don't have real factory images yet, so this prototype trains and evaluates
on the **MVTec LOCO AD "juice_bottle"** category as a stand-in: it's a
well-known public benchmark for exactly this kind of problem (defect-free
training images, mixed normal/defective test images) that also happens to be
a real juice bottle photographed from the side — closer to what a Cubii line
camera would actually see than MVTec AD's top-down "bottle" category (which
photographs straight down through the neck, a different but also realistic
inspection angle). Either way, this lets us validate the pipeline before real
Cubii line images are available; `configs/patchcore.yaml`'s `data.dataset`
field switches between the two (`mvtec_loco` or `mvtec_ad`).

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

Download the MVTec LOCO AD dataset (or just the `juice_bottle` category) from
the official source and extract it so you end up with:

```
data/MVTec_LOCO/juice_bottle/
├── train/
│   └── good/                # only defect-free images — this is all PatchCore trains on
├── test/
│   ├── good/
│   ├── logical_anomalies/   # e.g. wrong fill level, wrong label/fruit combination
│   └── structural_anomalies/ # e.g. cracks, contamination
└── ground_truth/
    ├── logical_anomalies/
    └── structural_anomalies/
```

To use the original MVTec AD "bottle" category instead, set `data.dataset:
mvtec_ad`, `data.root: data/MVTec`, `data.category: bottle` in
`configs/patchcore.yaml`.

The dataset is **not committed to git** (see `.gitignore`) — it's a few
hundred MB and is a public benchmark dataset, not project source code.

## Training

```bash
source venv/bin/activate
python src/train.py
```

This reads `configs/patchcore.yaml`, builds the matching anomalib datamodule
pointed at `data/MVTec_LOCO/juice_bottle`, fits PatchCore (this just means:
run all "good" training images through the backbone once and build the
memory bank — there's no gradient descent, so this is fast, even on CPU),
then evaluates against the test set and prints image-level and pixel-level
AUROC.

Outputs (checkpoint, threshold, metrics, visualizations) land under `results/`.

## Inference on a single image

```bash
source venv/bin/activate
python src/infer.py --image path/to/some_test_image.png
```

Prints the anomaly score and predicted label (normal/anomalous), and saves a
heatmap visualization showing *where* the model thinks the defect is.

## Path to real Cubii data

Once real line images are available: collect a folder of defect-free
container images (`good/`), no defective examples strictly required, and use
`src/train_folder.py` instead of `src/train.py` — it skips the MVTec-specific
folder structure (test/ground_truth splits) and trains directly on a plain
image folder, which is the realistic shape of early production data.
