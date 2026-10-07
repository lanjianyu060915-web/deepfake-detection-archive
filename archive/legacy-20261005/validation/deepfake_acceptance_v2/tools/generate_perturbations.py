#!/usr/bin/env python3
"""Create deterministic, label-preserving D-v2 image perturbations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from common import (
    GROUND_TRUTH_FIELDS,
    LABELS,
    image_dimensions,
    parse_bool,
    portable_relative,
    read_csv,
    require_exact_header,
    resolve_dataset_path,
    sha256_file,
    write_csv,
)


VARIANTS = (
    ("JPEG", "q90", ".jpg"),
    ("JPEG", "q70", ".jpg"),
    ("JPEG", "q50", ".jpg"),
    ("Scale", "075", ".png"),
    ("Scale", "050", ".png"),
    ("Blur", "sigma1", ".png"),
    ("Blur", "sigma2", ".png"),
    ("Brightness", "minus15", ".png"),
    ("Brightness", "plus15", ".png"),
)


def _save_variant(source: Image.Image, kind: str, level: str, destination: Path) -> None:
    image = source.convert("RGB")
    if kind == "JPEG":
        quality = int(level[1:])
        image.save(destination, format="JPEG", quality=quality, subsampling=2, optimize=False, progressive=False)
    elif kind == "Scale":
        scale = int(level) / 100.0
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image.resize(size, Image.Resampling.LANCZOS).save(destination, format="PNG", optimize=False)
    elif kind == "Blur":
        radius = float(level.removeprefix("sigma"))
        image.filter(ImageFilter.GaussianBlur(radius=radius)).save(destination, format="PNG", optimize=False)
    elif kind == "Brightness":
        factor = 0.85 if level == "minus15" else 1.15
        ImageEnhance.Brightness(image).enhance(factor).save(destination, format="PNG", optimize=False)
    else:
        raise AssertionError(kind)


def generate(
    manifest: Path,
    dataset_root: Path,
    output_root: Path,
    output_manifest: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    fields, rows = read_csv(manifest)
    require_exact_header(fields, GROUND_TRUTH_FIELDS, "ground truth")
    originals: list[dict[str, str]] = []
    for line_number, row in enumerate(rows, 2):
        if row["label"] not in LABELS:
            raise ValueError(f"line {line_number}: only trusted REAL/FAKE labels are allowed")
        if parse_bool(row["is_original"], f"line {line_number} is_original"):
            originals.append(row)
    if not originals:
        raise ValueError("manifest contains no trusted original samples")

    root = dataset_root.resolve()
    out = output_root.resolve()
    try:
        out.relative_to(root)
    except ValueError as exc:
        raise ValueError("output-root must be inside dataset-root") from exc
    if output_manifest.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite manifest: {output_manifest}")
    out.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, object]] = []
    output_paths: set[Path] = set()
    for row in originals:
        parent_id = row["sample_id"]
        source_path = resolve_dataset_path(root, row["file_path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"source image missing: {row['file_path']}")
        with Image.open(source_path) as opened:
            opened.load()
            source = opened.copy()
        for kind, level, suffix in VARIANTS:
            token = kind.lower()
            sample_id = f"{parent_id}__{token}_{level}"
            destination = out / f"{sample_id}{suffix}"
            if destination in output_paths:
                raise ValueError(f"output name collision: {destination.name}")
            output_paths.add(destination)
            if destination.exists() and not overwrite:
                raise FileExistsError(f"refusing to overwrite perturbation: {destination}")
            _save_variant(source, kind, level, destination)
            width, height = image_dimensions(destination)
            generated.append(
                {
                    "sample_id": sample_id,
                    "file_path": portable_relative(destination, root),
                    "label": row["label"],
                    "source": row["source"],
                    "generation_method": row["generation_method"],
                    "source_face_id": row["source_face_id"],
                    "target_face_id": row["target_face_id"],
                    "is_original": "false",
                    "parent_sample_id": parent_id,
                    "perturbation_type": kind,
                    "perturbation_level": level,
                    "width": width,
                    "height": height,
                    "sha256": sha256_file(destination),
                    "notes": "Deterministic derivative; truth label inherited from trusted parent.",
                }
            )

    write_csv(output_manifest, GROUND_TRUTH_FIELDS, [*originals, *generated])
    return {
        "status": "success",
        "original_count": len(originals),
        "generated_count": len(generated),
        "variants_per_original": len(VARIANTS),
        "manifest": str(output_manifest.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = generate(args.manifest, args.dataset_root, args.output_root, args.output_manifest, args.overwrite)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

