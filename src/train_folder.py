"""Train PatchCore on a plain folder of "good" images (no MVTec-style splits).

This is the script to use once real Cubii line images are available: unlike
train.py (which expects the MVTec AD train/test/ground_truth layout), this
only needs a folder of defect-free container images. You can optionally add
a folder of known-defective images too, for evaluation, but it isn't
required — PatchCore never trains on defective examples either way.

Usage:
    python src/train_folder.py --normal-dir data/cubii/good \\
        [--abnormal-dir data/cubii/defective] \\
        [--category cubii-line1]
"""

from __future__ import annotations

import argparse

from backbone_cache import patch_timm_for_local_weights

patch_timm_for_local_weights()

from anomalib.data import Folder  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.models import Patchcore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-dir", required=True,
                         help="Folder of defect-free ('good') images")
    parser.add_argument("--abnormal-dir", default=None,
                         help="Optional folder of known-defective images, for evaluation")
    parser.add_argument("--mask-dir", default=None,
                         help="Optional folder of pixel-level defect masks matching --abnormal-dir filenames")
    parser.add_argument("--normal-test-dir", default=None,
                         help="Optional separate folder of held-out normal images for evaluation")
    parser.add_argument("--category", default="cubii",
                         help="A name for this dataset, used to organize results/")
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--coreset-sampling-ratio", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    datamodule = Folder(
        name=args.category,
        normal_dir=args.normal_dir,
        abnormal_dir=args.abnormal_dir,
        mask_dir=args.mask_dir,
        normal_test_dir=args.normal_test_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = Patchcore(
        backbone=args.backbone,
        layers=args.layers,
        pre_trained=True,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
    )

    engine = Engine(default_root_dir="results")

    print(f"Training PatchCore on folder data: normal_dir={args.normal_dir}, "
          f"abnormal_dir={args.abnormal_dir}")
    test_results = engine.train(model=model, datamodule=datamodule)

    print("\n=== Test results ===")
    for result in test_results:
        for key, value in result.items():
            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
            print(f"{key}: {formatted}")

    ckpt_path = engine.trainer.checkpoint_callback.best_model_path
    print(f"\nCheckpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
