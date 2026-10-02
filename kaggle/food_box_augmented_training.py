"""Self-contained Kaggle notebook script: PatchCore on GoodsAD food_box, with
brightness/contrast augmentation.

Not meant to be run as one file — copy each "# %% CELL" section into its own
Kaggle notebook cell, in order. Written to have NO dependency on this
project's Claude Code session: it only needs the GoodsAD food_box dataset
(as a Kaggle Dataset input) and internet access turned on (for pip installs
and the pretrained backbone weights download).

Setup before running:
  1. Upload the GoodsAD food_box dataset zip (the same ~1.7GB zip already
     used earlier in this project) as a new Kaggle Dataset. Kaggle will
     auto-extract it. Note the dataset's mount path, usually
     /kaggle/input/<your-dataset-name>/.
  2. Create a new Kaggle Notebook, attach that dataset, and under
     Settings -> turn ON internet access.
  3. Edit DATA_ROOT below (Cell 1) to match your dataset's actual mount path
     and folder layout (it should contain train/good, test/<defect_type>/,
     ground_truth/<defect_type>/ for the food_box category).
  4. Run cells in order. Expect ~1-1.5 hours on Kaggle's CPU, faster if you
     select a GPU runtime (not required -- PatchCore does not need one for
     a dataset this size, but it won't hurt).

Baseline to compare against (same dataset, no augmentation, trained
elsewhere): image_AUROC 0.8027, image_F1Score 0.8000, pixel_AUROC 0.9746,
pixel_F1Score 0.3482.
"""

# %% CELL 1 -- config: EDIT THIS to match your uploaded Kaggle dataset
import os
from pathlib import Path

# Path to the extracted GoodsAD food_box folder as Kaggle mounted it.
# Must contain: train/good, test/good, test/<defect_type>/, ground_truth/<defect_type>/
DATA_ROOT = Path("/kaggle/input/goodsad-food-box/food_box")

# Kaggle's /kaggle/input is read-only, so build the merged (flat
# abnormal/masks) layout under /kaggle/working instead.
MERGED_ROOT = Path("/kaggle/working/GoodsAD_merged/food_box")

RESULTS_DIR = Path("/kaggle/working/results")


# %% CELL 2 -- install dependencies (Kaggle already has torch; this adds anomalib)
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "anomalib[core]==2.6.0"],
    check=True,
)


# %% CELL 3 -- rebuild the flat abnormal/ + masks/ folders from the raw
# GoodsAD structure (same workaround used earlier for the anomalib
# make_mvtec_ad_dataset pandas bug -- using the Folder datamodule sidesteps
# it entirely, so this step is required either way)
import shutil

abnormal_dir = MERGED_ROOT / "abnormal"
masks_dir = MERGED_ROOT / "masks"
abnormal_dir.mkdir(parents=True, exist_ok=True)
masks_dir.mkdir(parents=True, exist_ok=True)

defect_types = ["deformation", "opened", "surface_damage"]
copied = 0
for defect_type in defect_types:
    src_test_dir = DATA_ROOT / "test" / defect_type
    src_mask_dir = DATA_ROOT / "ground_truth" / defect_type
    if not src_test_dir.is_dir():
        print(f"Skipping missing defect type folder: {src_test_dir}")
        continue
    for img_path in sorted(src_test_dir.iterdir()):
        dest_name = f"{defect_type}_{img_path.name}"
        shutil.copy2(img_path, abnormal_dir / dest_name)
        mask_path = src_mask_dir / img_path.name
        if not mask_path.is_file():
            # GoodsAD masks are typically .png even when images are .jpg
            mask_path = src_mask_dir / (img_path.stem + ".png")
        if mask_path.is_file():
            dest_mask_name = f"{defect_type}_{img_path.stem}.png"
            shutil.copy2(mask_path, masks_dir / dest_mask_name)
        copied += 1

print(f"Copied {copied} abnormal images into {abnormal_dir}")
print(f"Abnormal: {len(list(abnormal_dir.iterdir()))} files, "
      f"Masks: {len(list(masks_dir.iterdir()))} files")


# %% CELL 4 -- backbone weights: fetch the same ImageNet-pretrained
# wide_resnet50_2 weights used throughout this project, from their original
# GitHub release, and point timm at the local file (works the same as this
# project's src/backbone_cache.py, inlined here since Kaggle notebooks don't
# have access to this repo's other files)
import urllib.request

WEIGHTS_DIR = Path("/kaggle/working/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_URL = (
    "https://github.com/rwightman/pytorch-image-models/releases/"
    "download/v0.1-weights/wide_resnet50_racm-8234f177.pth"
)
WEIGHTS_PATH = WEIGHTS_DIR / "wide_resnet50_racm-8234f177.pth"

if not WEIGHTS_PATH.exists():
    print(f"Downloading backbone weights to {WEIGHTS_PATH} ...")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)

import timm

_original_create_model = timm.create_model


def _patched_create_model(model_name, *args, **kwargs):
    if kwargs.get("pretrained") and model_name == "wide_resnet50_2":
        kwargs.pop("pretrained_cfg", None)
        kwargs["pretrained_cfg_overlay"] = {"file": str(WEIGHTS_PATH)}
    return _original_create_model(model_name, *args, **kwargs)


timm.create_model = _patched_create_model


# %% CELL 5 -- train PatchCore with brightness/contrast augmentation
# (matches this project's src/train_folder.py --augment flag, inlined)
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
from torchvision.transforms import v2

# Kaggle/Colab notebooks can hit a RecursionError from tqdm's progress bar
# fighting with rich/ipykernel's Jupyter display hooks. We don't need the
# live bar, just the final metrics, so disable it outright.
import tqdm as tqdm_module

_original_tqdm_init = tqdm_module.tqdm.__init__


def _patched_tqdm_init(self, *args, **kwargs):
    kwargs["disable"] = True
    _original_tqdm_init(self, *args, **kwargs)


tqdm_module.tqdm.__init__ = _patched_tqdm_init

train_augmentations = v2.ColorJitter(brightness=0.2, contrast=0.2)

datamodule = Folder(
    name="food_box_augmented",
    normal_dir=DATA_ROOT / "train" / "good",
    abnormal_dir=abnormal_dir,
    mask_dir=masks_dir,
    normal_test_dir=DATA_ROOT / "test" / "good",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=2,
    train_augmentations=train_augmentations,
)

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    coreset_sampling_ratio=0.1,  # matches the no-augmentation baseline for a fair comparison
)

engine = Engine(default_root_dir=str(RESULTS_DIR))

print("Training PatchCore (augmented) on food_box ...")
test_results = engine.train(model=model, datamodule=datamodule)

print("\n=== Test results (augmented) ===")
for result in test_results:
    for key, value in result.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"{key}: {formatted}")

print("\n=== Compare against baseline (no augmentation) ===")
print("image_AUROC:   0.8027")
print("image_F1Score: 0.8000")
print("pixel_AUROC:   0.9746")
print("pixel_F1Score: 0.3482")

ckpt_path = engine.trainer.checkpoint_callback.best_model_path
print(f"\nCheckpoint saved to: {ckpt_path}")
