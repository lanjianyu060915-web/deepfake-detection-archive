"""Paired-source import behavior, independent of network and model accuracy."""

import csv
import hashlib
import io
import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from deepfake_detection_archive.data.manifest import load_manifest
from deepfake_detection_archive.data.split import validate_splits


MODULE: dict[str, Any] = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/prepare_doppelganger.py")
)
prepare = MODULE["prepare"]
select_pairs = MODULE["select_pairs"]


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = []
    for index in range(8):
        row = {
            "pair_id": str(index),
            "event_id": "RAIN",
            "event_name": "rain",
            "source_dataset": "FSD50K",
            "source_clip_id": str(index // 2),
            "real_uploader": f"recordist{index // 2}",
            "real_license_url": "http://creativecommons.org/publicdomain/zero/1.0/",
            "real_title": f"rain {index // 2}",
            "synthetic_license": "Stability AI Community License",
        }
        for kind, repo, revision in (
            ("real", MODULE["REAL_REPO"], MODULE["REAL_REVISION"]),
            ("synthetic", MODULE["REPO"], MODULE["SYNTHETIC_REVISION"]),
        ):
            path = f"{kind}/{index}.wav"
            row[f"{kind}_repo_id"] = repo
            row[f"{kind}_revision"] = revision
            row[f"{kind}_audio_path"] = path
            row[f"{kind}_audio_url"] = (
                f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
            )
            local = tmp_path / path
            local.parent.mkdir(exist_ok=True)
            rate = 24000 if kind == "real" else 44100
            # Different durations and channels exercise symmetric normalization.
            duration = 6 if kind == "real" else 5
            signal = np.sin(
                np.arange(rate * duration) * (300 + index * 13) / rate * 2 * np.pi
            ).astype(np.float32) * (0.2 if kind == "real" else 0.15)
            if kind == "real":
                signal = np.column_stack((signal, signal * 0.8))
            sf.write(local, signal, rate)
        rows.append(row)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue().encode()
    metadata = tmp_path / "metadata.csv"
    metadata.write_bytes(raw)
    info = {
        row["source_clip_id"]: {
            "title": row["real_title"],
            "license": row["real_license_url"],
            "uploader": row["real_uploader"],
            "description": "rain recorded outside",
            "tags": ["rain"],
        }
        for row in rows
    }
    info_raw = json.dumps(info).encode()
    (tmp_path / "source-info.json").write_bytes(info_raw)
    monkeypatch.setitem(
        select_pairs.__globals__,
        "SOURCE_INFO_SHA256",
        hashlib.sha256(info_raw).hexdigest(),
    )
    monkeypatch.setitem(
        select_pairs.__globals__, "METADATA_SHA256", hashlib.sha256(raw).hexdigest()
    )

    def download(url: str, path: Path) -> None:
        kind, filename = url.rsplit("/", 2)[-2:]
        shutil.copyfile(tmp_path / kind / filename, path)

    monkeypatch.setitem(prepare.__globals__, "_download", download)
    return metadata


def test_balanced_groups_and_symmetric_media(source: Path, tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    summary = prepare(source, source.with_name("source-info.json"), output, workers=2)
    records = load_manifest(output / "manifest.jsonl")
    assert len(records) == 16
    assert validate_splits(records) == []
    assert summary["audit_passed"] is True
    assert summary["competition_approved"] is False
    assert summary["source_decode_coverage"] == 1.0
    for split in ("train", "dev", "holdout"):
        labels = [record.label for record in records if record.split == split]
        assert labels.count(0) == labels.count(1) > 0
    for uploader in {record.group_id for record in records}:
        assert len({r.split for r in records if r.group_id == uploader}) == 1
    for record in records:
        info = sf.info(output / record.path)
        waveform, _ = sf.read(output / record.path)
        assert np.max(np.abs(waveform)) == pytest.approx(0.95, abs=1 / 32768)
        assert (info.samplerate, info.channels, info.frames, info.subtype) == (
            16000,
            1,
            64600,
            "PCM_16",
        )
        assert record.generator_version == "unknown"
    with pytest.raises(FileExistsError):
        prepare(source, source.with_name("source-info.json"), output)


def test_failed_or_short_pair_is_accounted_and_excluded(
    source: Path, tmp_path: Path
) -> None:
    (tmp_path / "synthetic/0.wav").write_bytes(b"broken audio")
    # Long enough for the final crop, but shorter than upstream conditioning.
    sf.write(tmp_path / "real/1.wav", np.ones(72000, dtype=np.float32), 16000)
    output = tmp_path / "prepared"
    summary = prepare(source, source.with_name("source-info.json"), output)
    inventory = load_manifest(output / "inventory.jsonl")
    records = load_manifest(output / "manifest.jsonl")
    assert len(inventory) == 16
    assert len(records) == 12
    assert all(record.original_id != "freesound:0" for record in records)
    assert summary["status_counts"]["decode_failed"] == 1
    assert summary["status_counts"]["excluded"] == 3
    assert summary["source_decode_coverage"] == 15 / 16
    provenance = [
        json.loads(line)
        for line in (output / "provenance.jsonl").read_text().splitlines()
    ]
    assert sum("error" in row for row in provenance) == 1


def test_fixed_metadata_and_deterministic_selection(source: Path) -> None:
    raw = source.read_bytes()
    first = select_pairs(
        raw, source.with_name("source-info.json").read_bytes(), per_event=3, seed=42
    )
    assert first == select_pairs(
        raw, source.with_name("source-info.json").read_bytes(), per_event=3, seed=42
    )
    assert len(first) == 3
    with pytest.raises(ValueError, match="pinned revision"):
        select_pairs(
            raw + b"\n", source.with_name("source-info.json").read_bytes(), 3, 42
        )


def test_unlicensed_rows_do_not_enter_selection(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = source.read_bytes().replace(b"publicdomain/zero/1.0", b"licenses/by-nc/4.0")
    monkeypatch.setitem(
        select_pairs.__globals__, "METADATA_SHA256", hashlib.sha256(raw).hexdigest()
    )
    with pytest.raises(ValueError, match="no eligible"):
        select_pairs(raw, source.with_name("source-info.json").read_bytes(), 40, 42)


def test_reviewed_exclusions_remove_both_labels(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(select_pairs.__globals__, "EXCLUDED_SOURCE_IDS", {"0"})
    output = tmp_path / "prepared"
    prepare(source, source.with_name("source-info.json"), output)
    inventory = load_manifest(output / "inventory.jsonl")
    assert len(inventory) == 12
    assert all(record.original_id != "freesound:0" for record in inventory)


def test_attribution_mismatch_is_rejected_before_download(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = source.with_name("source-info.json").read_bytes()
    info = info.replace(b"recordist0", b"unrelated_recordist")
    monkeypatch.setitem(
        select_pairs.__globals__, "SOURCE_INFO_SHA256", hashlib.sha256(info).hexdigest()
    )
    with pytest.raises(ValueError, match="attribution or license"):
        select_pairs(source.read_bytes(), info, 40, 42)
