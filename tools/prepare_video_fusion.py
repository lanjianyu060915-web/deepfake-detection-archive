"""Assemble the frozen issue #72 research candidate; no calibration or training."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from deepfake_detection_archive.models import (
    MediaFeatureClassifier,
    UFDVideoClassifier,
    VideoFusionClassifier,
    VideoFusionConfig,
    WaveRepVideoClassifier,
    load_checkpoint,
    save_checkpoint,
)


SOURCE_HASHES = {
    "clip": "81907988948ab62e4ab20adc33f7eff6bee14f1067cf47e1e712525e0010d7f7",
    "clip_head": "7dc0283f0975b2c373956e670138e4f870ed64fda78d2863183655af49037bdf",
    "waverep": "35659dbdec097790a82bfb1d9bfeb761215870ac05d7d693761eb8be15da4ce8",
}
# PR70 clip43_waverep_fusion: fit on original dev844, never AIGVDBench dev1880.
CALIBRATION = {
    "mean": [0.6499799112061048, -5.7335910216048065],
    "std": [3.2967671720293046, 2.5814849859556146],
    "coef": [1.3668461443051254, 0.5686207697389872],
    "intercept": 0.04395778566979127,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SOURCE_HASHES:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new checkpoint path")
    clip, _ = load_checkpoint(args.clip, expected_sha256=SOURCE_HASHES["clip"])
    head, _ = load_checkpoint(
        args.clip_head, expected_sha256=SOURCE_HASHES["clip_head"]
    )
    waverep, _ = load_checkpoint(args.waverep, expected_sha256=SOURCE_HASHES["waverep"])
    if not (
        isinstance(clip, UFDVideoClassifier)
        and isinstance(head, MediaFeatureClassifier)
        and isinstance(waverep, WaveRepVideoClassifier)
    ):
        raise TypeError("frozen source checkpoint architectures disagree")
    config = VideoFusionConfig(
        clip=asdict(clip.config),
        clip_head=asdict(head.config),
        waverep=asdict(waverep.config),
        **CALIBRATION,  # type: ignore[arg-type]
    )
    model = VideoFusionClassifier(config)
    model.clip_backbone.load_state_dict(clip.backbone.state_dict(), strict=True)
    model.clip_head.load_state_dict(head.state_dict(), strict=True)
    model.waverep.load_state_dict(waverep.state_dict(), strict=True)
    descriptor = save_checkpoint(args.output, model)
    args.output.with_suffix(".resources.json").write_text(
        json.dumps(
            {
                "source_sha256": SOURCE_HASHES,
                "calibration": "PR70 clip43_waverep_fusion; original dev844 only",
                "research_only": True,
                "competition_use": "unconfirmed; WaveRep non-profit license",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(descriptor, sort_keys=True))


if __name__ == "__main__":
    main()
