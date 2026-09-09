"""Train PatchCore on an MVTec-style dataset (MVTec AD or MVTec LOCO AD).

Usage:
    python src/train.py [--config configs/patchcore.yaml]

What this actually does:
    1. Loads config/data/model hyperparameters from the YAML config.
    2. Points the matching anomalib datamodule at data/<root>/<category>:
       - "mvtec_ad" expects train/good, test/<defect types>,
         ground_truth/<defect types> (e.g. the "bottle" category).
       - "mvtec_loco" expects the MVTec LOCO AD layout (e.g. "juice_bottle"),
         which anomalib reads natively via the same train/good structure.
    3. Builds a PatchCore model: a frozen, ImageNet-pretrained CNN backbone
       used purely as a feature extractor (no weights are updated).
    4. "Trains" — for PatchCore this means: run every training (good-only)
       image through the backbone once, collect local patch features, and
       subsample them (coreset selection) into a compact memory bank. There
       is no gradient descent, so this is much faster than typical deep
       learning training, even on CPU.
    5. Evaluates on the test split and prints image-level / pixel-level
       AUROC, then reports where the trained checkpoint was saved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from backbone_cache import patch_timm_for_local_weights

# Must happen before anomalib pulls in timm-backed feature extractors.
patch_timm_for_local_weights()

from anomalib.data import MVTecAD, MVTecLOCO  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.models import Patchcore  # noqa: E402

_DATAMODULES = {
    "mvtec_ad": MVTecAD,
    "mvtec_loco": MVTecLOCO,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/patchcore.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    trainer_cfg = cfg.get("trainer", {})

    data_root = Path(data_cfg["root"]) / data_cfg["category"]
    if not (data_root / "train" / "good").is_dir():
        raise SystemExit(
            f"Expected training images at {data_root / 'train' / 'good'}, "
            "but that folder doesn't exist. See README.md for the dataset "
            "layout this script expects."
        )

    dataset_kind = data_cfg.get("dataset", "mvtec_ad")
    if dataset_kind not in _DATAMODULES:
        raise SystemExit(
            f"Unknown data.dataset '{dataset_kind}' in config; "
            f"expected one of {list(_DATAMODULES)}."
        )

    datamodule = _DATAMODULES[dataset_kind](
        root=data_cfg["root"],
        category=data_cfg["category"],
        train_batch_size=data_cfg.get("train_batch_size", 32),
        eval_batch_size=data_cfg.get("eval_batch_size", 32),
        num_workers=data_cfg.get("num_workers", 4),
    )

    model = Patchcore(
        backbone=model_cfg.get("backbone", "wide_resnet50_2"),
        layers=model_cfg.get("layers", ["layer2", "layer3"]),
        pre_trained=model_cfg.get("pre_trained", True),
        coreset_sampling_ratio=model_cfg.get("coreset_sampling_ratio", 0.1),
        num_neighbors=model_cfg.get("num_neighbors", 9),
    )

    engine = Engine(default_root_dir=trainer_cfg.get("default_root_dir", "results"))

    print(f"Training PatchCore on category='{data_cfg['category']}' "
          f"(root={data_cfg['root']}) ...")
    test_results = engine.train(model=model, datamodule=datamodule)

    print("\n=== Test results ===")
    for result in test_results:
        for key, value in result.items():
            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
            print(f"{key}: {formatted}")

    ckpt_path = engine.trainer.checkpoint_callback.best_model_path
    print(f"\nCheckpoint saved to: {ckpt_path}")
    print("Run inference with:")
    print(f'  python src/infer.py --image <path/to/image.png> --ckpt "{ckpt_path}"')


if __name__ == "__main__":
    main()
