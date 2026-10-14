"""Convert the official WaveRep G1 detector weights to a local checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from deepfake_detection_archive.models.checkpoint import (
    save_checkpoint,
    verify_sha256,
)
from deepfake_detection_archive.models.waverep import (
    WaveRepConfig,
    WaveRepVideoClassifier,
)


G1_SHA256 = "1c7566b26c0b0afe3f5642dd4e454039328d59f512a5d815d5edaa10b544a897"
WAVEREP_COMMIT = "0fd6010759c14b572b7842a28fa9f85fe1ddd2fd"


def author_state(data: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Key rule of the author's demo/utils.py::create_model."""
    if "state_dict" in data:
        wrapped: dict[str, torch.Tensor] = data["state_dict"]
        return {
            key[6:]: value for key, value in wrapped.items() if key.startswith("model")
        }
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new checkpoint path")
    verify_sha256(args.weights, G1_SHA256)
    data = torch.load(args.weights, map_location="cpu", weights_only=True)
    model = WaveRepVideoClassifier(WaveRepConfig())
    model.backbone.load_state_dict(author_state(data), strict=True)
    descriptor = save_checkpoint(args.output, model)
    resources = {
        "waverep_commit": WAVEREP_COMMIT,
        "weights_sha256": G1_SHA256,
        "checkpoint_sha256": descriptor["sha256"],
        "preprocessing": "RGB, native CenterCrop(504) with zero pad, ImageNet mean/std",
        "frame_score": "end-to-end DINOv2 ViT-B/14 reg4 logit (author LLR)",
        "video_score": "mean raw frame logits; higher is fake",
        "license": "GRIP-UNINA non-profit research licence; no industrial use",
        "research_only": True,
    }
    args.output.with_suffix(".resources.json").write_text(
        json.dumps(resources, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(descriptor, sort_keys=True))


if __name__ == "__main__":
    main()
