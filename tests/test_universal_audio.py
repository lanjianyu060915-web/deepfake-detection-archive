"""Joint registries preserve split isolation without consuming holdout media."""

import json
import runpy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    SourceRecord,
    load_manifest,
    write_manifest,
)


MODULE: dict[str, Any] = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/prepare_universal_audio.py")
)


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Any]:
    domains = {
        "environment": "envsdd-urbansound8k-research",
        "music": "aime-music-research",
        "speech": "asvspoof2019-la-official",
    }
    media = tmp_path / "media"
    rows: dict[str, list[SampleRecord]] = {
        key: [] for key in ("mixed", "env_dev", "speech")
    }
    for domain, source in domains.items():
        for split in ("train", "dev", "holdout"):
            if split == "holdout" and domain != "music":
                continue
            for label in (0, 1):
                prefix = {
                    "music": MODULE["MUSIC_PREFIX"],
                    "environment": MODULE["ENV_TRAIN_PREFIX"]
                    if split == "train"
                    else "media/",
                    "speech": f"ASVspoof2019_LA_{split}/flac/",
                }[domain]
                row = SampleRecord(
                    f"{domain}-{split}-{label}",
                    f"{prefix}{split}-{label}.wav",
                    "audio",
                    label,
                    source,
                    content_type=domain,
                    group_id=f"{domain}:{split}",
                    split=split,
                )
                key = (
                    "env_dev"
                    if domain == "environment" and split == "dev"
                    else ("speech" if domain == "speech" else "mixed")
                )
                rows[key].append(row)
                if split != "holdout":
                    path = media / (
                        MODULE["ENV_DEV_PREFIX"] + row.path
                        if key == "env_dev"
                        else row.path
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(row.sample_id.encode())
    result: dict[str, Any] = {"media_root": media, "output_dir": tmp_path / "output"}
    for key, records in rows.items():
        path = tmp_path / f"{key}.jsonl"
        write_manifest(records, path)
        result[key] = path
    source_file = tmp_path / "sources.jsonl"
    source_file.write_text(
        "".join(
            json.dumps(
                asdict(
                    SourceRecord(
                        source,
                        "fixture",
                        "1",
                        "research",
                        "fixture",
                        "2026-10-13",
                        ("research",),
                    )
                )
            )
            + "\n"
            for source in sorted(domains.values())
        )
    )
    result["sources"] = (source_file, source_file, source_file)
    return result


def test_build_preserves_all_splits_without_holdout_files(
    inputs: dict[str, Any],
) -> None:
    summary = MODULE["build"](**inputs)
    rows = load_manifest(inputs["output_dir"] / "manifest.jsonl")
    assert len(rows) == 14
    assert summary["media_verified"] == {"train": 6, "dev": 6, "holdout": 0}
    assert {row.content_type for row in rows} == {"speech", "environment", "music"}
    before = (inputs["output_dir"] / "manifest.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        MODULE["build"](**inputs)
    assert (inputs["output_dir"] / "manifest.jsonl").read_bytes() == before


def test_joint_registry_rejects_cross_split_groups(inputs: dict[str, Any]) -> None:
    rows = load_manifest(inputs["speech"])
    rows = [replace(row, identity_id="shared-speaker") for row in rows]
    write_manifest(rows, inputs["speech"])
    with pytest.raises(ValueError, match="leakage"):
        MODULE["build"](**inputs)
    assert not inputs["output_dir"].exists()
