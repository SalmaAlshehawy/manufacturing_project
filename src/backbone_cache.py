"""Local caching for pretrained CNN backbone weights.

By default, anomalib's feature extractors load ImageNet-pretrained backbone
weights (e.g. wide_resnet50_2) through `timm`, which downloads them from the
Hugging Face Hub the first time a given backbone is used. That's an extra
network dependency at train/infer time, and in some environments (locked-down
CI runners, offline factory-floor machines, restricted egress policies)
huggingface.co simply isn't reachable.

This module downloads the backbone weights once from their original GitHub
release (the upstream pytorch-image-models project) into `weights/`, and
monkeypatches `timm.create_model` so pretrained backbones load from that local
file instead of hitting the network. If the download itself fails (no
internet at all), it falls back silently to timm's normal pretrained-download
behavior, so this is a pure optimization, not a requirement.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

# backbone name -> (upstream URL, local filename)
_BACKBONE_SOURCES: dict[str, tuple[str, str]] = {
    "wide_resnet50_2": (
        "https://github.com/rwightman/pytorch-image-models/releases/"
        "download/v0.1-weights/wide_resnet50_racm-8234f177.pth",
        "wide_resnet50_racm-8234f177.pth",
    ),
}


def _ensure_local_weights(backbone: str) -> Path | None:
    source = _BACKBONE_SOURCES.get(backbone)
    if source is None:
        return None

    url, filename = source
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / filename
    if dest.exists():
        return dest

    print(f"[backbone_cache] Downloading {backbone} weights to {dest} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: fall back to timm's own download
        print(f"[backbone_cache] Could not pre-fetch weights ({exc}); "
              f"falling back to timm's default download.")
        return None

    return dest


def patch_timm_for_local_weights() -> None:
    """Make timm.create_model use locally cached weights when available."""
    import timm

    original_create_model = timm.create_model

    def patched_create_model(model_name, *args, **kwargs):
        if kwargs.get("pretrained") and "pretrained_cfg_overlay" not in kwargs:
            local_path = _ensure_local_weights(model_name)
            if local_path is not None:
                kwargs.pop("pretrained_cfg", None)
                kwargs["pretrained_cfg_overlay"] = {"file": str(local_path)}
        return original_create_model(model_name, *args, **kwargs)

    timm.create_model = patched_create_model
