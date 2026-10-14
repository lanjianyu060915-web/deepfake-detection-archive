"""Convert the official UFD head and OpenAI CLIP archive to a local checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from deepfake_detection_archive.models.checkpoint import (
    save_checkpoint,
    verify_sha256,
)
from deepfake_detection_archive.models.ufd import UFDConfig, UFDVideoClassifier


CLIP_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
UFD_COMMIT = "030495aea3300a8b54c0ec37ec7fe1dd7e63c619"


def convert_visual_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map OpenAI's visual state to the equivalent transformers CLIP vision model.

    Attention Q/K/V are stored together by OpenAI and separately by transformers;
    the visual projection is transposed. Text weights are deliberately omitted.
    """
    names = {
        "vision_model.embeddings.class_embedding": "class_embedding",
        "vision_model.embeddings.patch_embedding.weight": "conv1.weight",
        "vision_model.embeddings.position_embedding.weight": "positional_embedding",
        "vision_model.pre_layrnorm.weight": "ln_pre.weight",
        "vision_model.pre_layrnorm.bias": "ln_pre.bias",
        "vision_model.post_layernorm.weight": "ln_post.weight",
        "vision_model.post_layernorm.bias": "ln_post.bias",
    }
    converted = {key: state[f"visual.{value}"] for key, value in names.items()}
    converted["visual_projection.weight"] = state["visual.proj"].T.contiguous()
    for index in range(24):
        source = f"visual.transformer.resblocks.{index}"
        target = f"vision_model.encoder.layers.{index}"
        for field in ("weight", "bias"):
            projections = state[f"{source}.attn.in_proj_{field}"].chunk(3, dim=0)
            for name, tensor in zip(("q", "k", "v"), projections, strict=True):
                converted[f"{target}.self_attn.{name}_proj.{field}"] = tensor
            for new, old in (
                ("self_attn.out_proj", "attn.out_proj"),
                ("layer_norm1", "ln_1"),
                ("layer_norm2", "ln_2"),
                ("mlp.fc1", "mlp.c_fc"),
                ("mlp.fc2", "mlp.c_proj"),
            ):
                converted[f"{target}.{new}.{field}"] = state[f"{source}.{old}.{field}"]
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--head-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new checkpoint path")
    verify_sha256(args.clip, CLIP_SHA256)
    verify_sha256(args.head, args.head_sha256)
    # Only this preparation command reads the trusted upstream TorchScript file.
    # Runtime inference reads our tensor-only, self-contained checkpoint.
    archive = torch.jit.load(  # type: ignore[no-untyped-call]
        str(args.clip), map_location="cpu"
    )
    state = cast(dict[str, Tensor], archive.state_dict())
    model = UFDVideoClassifier(UFDConfig())
    model.backbone.load_state_dict(convert_visual_state(state), strict=True)
    model.head.load_state_dict(
        torch.load(args.head, map_location="cpu", weights_only=True), strict=True
    )
    descriptor = save_checkpoint(args.output, model)
    resources = {
        "ufd_commit": UFD_COMMIT,
        "clip_sha256": CLIP_SHA256,
        "head_sha256": args.head_sha256,
        "checkpoint_sha256": descriptor["sha256"],
        "preprocessing": "RGB, native CenterCrop(224), CLIP mean/std; no resize",
        "frame_score": "unnormalized CLIP image projection -> official linear head",
        "video_score": "mean raw frame logits; higher is fake",
        "research_only": True,
    }
    args.output.with_suffix(".resources.json").write_text(
        json.dumps(resources, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(descriptor, sort_keys=True))


if __name__ == "__main__":
    main()
