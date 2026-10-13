"""Build the frozen universal-audio (speech/music/environment) train/dev registry.

Merges three read-only domain registries into one single-root manifest. Music
and environment train rows come from the #32 mixed registry that already stores
media-root-relative paths; environment dev rows and ASVspoof train/dev rows are
re-rooted onto the same media root. Holdout rows are carried for registry
completeness only: training, selection and preparation never decode them.
No media is copied or modified; the tool only verifies that referenced bytes
exist under the supplied root.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    file_sha256,
    load_manifest,
    load_sources,
    write_manifest,
)
from deepfake_detection_archive.data.split import validate_splits
from deepfake_detection_archive.tracking import atomic_json, canonical_json


CONTENT_TYPES = ("environment", "music", "speech")
EXPECTED_SOURCES = (
    "aime-music-research",
    "asvspoof2019-la-official",
    "envsdd-urbansound8k-research",
)
MUSIC_PREFIX = "deepfake-audio-music-20261012/"
ENV_TRAIN_PREFIX = "deepfake-audio-expand-20261011/"
ENV_DEV_PREFIX = "deepfake-audio-expand-20261011/envsdd-urban/"
SPEECH_PREFIX = "ASVspoof2019_LA_"


def _rehosted(record: SampleRecord, path: str) -> SampleRecord:
    if record.status != "available" or record.label is None:
        raise ValueError(f"unavailable or unlabeled row: {record.sample_id}")
    if record.modality != "audio" or record.content_type not in CONTENT_TYPES:
        raise ValueError(f"unexpected media domain: {record.sample_id}")
    return replace(record, path=path)


def _music_or_env_train(record: SampleRecord) -> SampleRecord:
    if record.content_type == "music":
        if not record.path.startswith(MUSIC_PREFIX):
            raise ValueError(f"music row lacks shared-root prefix: {record.sample_id}")
        return _rehosted(record, record.path)
    if record.content_type == "environment":
        if record.split != "train":
            raise ValueError(
                f"environment train registry is train-only: {record.sample_id}"
            )
        if not record.path.startswith(ENV_TRAIN_PREFIX):
            raise ValueError(
                f"environment row lacks shared-root prefix: {record.sample_id}"
            )
        return _rehosted(record, record.path)
    raise ValueError(
        f"mixed registry only carries music/environment: {record.sample_id}"
    )


def _env_dev(record: SampleRecord) -> SampleRecord:
    if record.split != "dev" or not record.path.startswith("media/"):
        raise ValueError(
            f"environment dev registry expects split=dev media/: {record.sample_id}"
        )
    return _rehosted(record, ENV_DEV_PREFIX + record.path)


def _speech(record: SampleRecord) -> SampleRecord:
    if record.split not in ("train", "dev") or not record.path.startswith(
        SPEECH_PREFIX
    ):
        raise ValueError(
            f"speech registry expects ASVspoof train/dev paths: {record.sample_id}"
        )
    return _rehosted(record, record.path)


def _verify_media(records: list[SampleRecord], media_root: Path) -> dict[str, int]:
    checked = {"train": 0, "dev": 0, "holdout": 0}
    missing: list[str] = []
    for record in records:
        if record.split == "holdout":
            continue
        if not (media_root / record.path).is_file():
            missing.append(record.sample_id)
        else:
            checked[record.split] += 1
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(
            f"missing media files under root: {len(missing)} e.g. {preview}"
        )
    return checked


def build(  # noqa: PLR0913, PLR0917 -- explicit curation inputs
    mixed: Path,
    env_dev: Path,
    speech: Path,
    sources: tuple[Path, Path, Path],
    media_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    rows = (
        [_music_or_env_train(record) for record in load_manifest(mixed)]
        + [_env_dev(record) for record in load_manifest(env_dev)]
        + [_speech(record) for record in load_manifest(speech)]
    )
    identifiers = [record.sample_id for record in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("merged registry has duplicate sample IDs")
    errors = validate_splits(rows, require_class_coverage=False)
    if errors:
        raise ValueError(f"merged split validation failed: {'; '.join(errors)}")
    media = _verify_media(rows, media_root)
    rows.sort(key=lambda record: (record.split, record.content_type, record.sample_id))
    registries = {
        name: value
        for registry in sources
        for name, value in load_sources(registry).items()
    }
    if tuple(sorted(registries)) != EXPECTED_SOURCES:
        raise ValueError(f"unexpected source registry: {sorted(registries)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "manifest.jsonl"
    manifest_digest = write_manifest(rows, manifest_path)
    source_path = output_dir / "sources.jsonl"
    source_path.write_text(
        "".join(
            canonical_json(asdict(value)) + "\n"
            for _, value in sorted(registries.items())
        ),
        encoding="utf-8",
    )
    counts: dict[str, int] = Counter(
        f"{record.split}/{record.content_type}/label={record.label}" for record in rows
    )
    generators: dict[str, int] = Counter(
        f"{record.content_type}/{record.generator}" for record in rows
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "scope": (
            "speech/music/environment joint registry; holdout rows are registry "
            "metadata only and are never decoded"
        ),
        "inputs": {
            "mixed": str(mixed),
            "env_dev": str(env_dev),
            "speech": str(speech),
            "sources": [str(path) for path in sources],
            "media_root": str(media_root),
        },
        "total": len(rows),
        "counts": dict(sorted(counts.items())),
        "generators": dict(sorted(generators.items())),
        "media_verified": media,
        "manifest_sha256": manifest_digest,
        "sources_sha256": file_sha256(source_path),
        "sources": {
            name: {
                "license_id": value.license_id,
                "license_status": value.license_status,
                "allowed_uses": list(value.allowed_uses),
            }
            for name, value in sorted(registries.items())
        },
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", type=Path, required=True)
    parser.add_argument("--env-dev", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--mixed-sources", type=Path, required=True)
    parser.add_argument("--env-dev-sources", type=Path, required=True)
    parser.add_argument("--speech-sources", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    summary = build(
        arguments.mixed,
        arguments.env_dev,
        arguments.speech,
        (arguments.mixed_sources, arguments.env_dev_sources, arguments.speech_sources),
        arguments.media_root,
        arguments.output_dir,
    )
    print(summary["manifest_sha256"])
    print(summary["total"])


if __name__ == "__main__":
    main()
