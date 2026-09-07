"""Run a trained PatchCore checkpoint on a single image.

Usage:
    python src/infer.py --image path/to/image.png [--ckpt path/to/model.ckpt]

If --ckpt is omitted, the most recently modified *.ckpt under results/ is
used. Prints the anomaly score and predicted label, and saves a heatmap
visualization next to the checkpoint (or to --out-dir if given) showing
*where* the model thinks the defect is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from backbone_cache import patch_timm_for_local_weights

patch_timm_for_local_weights()

from anomalib.engine import Engine  # noqa: E402
from anomalib.models import Patchcore  # noqa: E402


def find_latest_checkpoint(results_dir: str = "results") -> Path:
    candidates = sorted(Path(results_dir).rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            f"No checkpoint found under {results_dir}/. Run src/train.py first, "
            "or pass --ckpt explicitly."
        )
    return candidates[-1]


def save_heatmap(image_path: Path, anomaly_map: np.ndarray, pred_score: float,
                  pred_label: str, out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    amap = anomaly_map.squeeze()
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(amap, cmap="jet", alpha=0.5)
    axes[1].set_title(f"{pred_label} (score={pred_score:.3f})")
    axes[1].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the image to inspect")
    parser.add_argument("--ckpt", default=None, help="Path to a trained checkpoint (.ckpt)")
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--out-dir", default="results/inference")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest_checkpoint()
    print(f"Using checkpoint: {ckpt_path}")

    # pre_trained=False: we only need the right architecture here — the
    # actual learned weights (backbone + memory bank) come from the checkpoint.
    model = Patchcore(backbone=args.backbone, layers=args.layers, pre_trained=False)
    engine = Engine()

    predictions = engine.predict(model=model, data_path=str(image_path), ckpt_path=str(ckpt_path))
    if not predictions:
        raise SystemExit("No predictions returned.")

    batch = predictions[0]
    pred_score = float(batch.pred_score[0])
    pred_label = "ANOMALOUS" if bool(batch.pred_label[0]) else "NORMAL"
    anomaly_map = batch.anomaly_map[0].detach().cpu().numpy()

    print(f"\nImage:      {image_path}")
    print(f"Pred label: {pred_label}")
    print(f"Pred score: {pred_score:.4f}")

    out_path = Path(args.out_dir) / f"{image_path.stem}_result.png"
    save_heatmap(image_path, anomaly_map, pred_score, pred_label, out_path)
    print(f"Heatmap saved to: {out_path}")


if __name__ == "__main__":
    main()
