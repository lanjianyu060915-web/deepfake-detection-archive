"""FakeSound import preserves missing originals and diagnoses actual WAV bytes."""

from __future__ import annotations

import io
import json
import runpy
import stat
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from deepfake_detection_archive.data.manifest import (
    file_sha256,
    load_manifest,
    load_sources,
)
from deepfake_detection_archive.data.split import validate_splits


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_fakesound.py"
MODULE: dict[str, Any] = runpy.run_path(str(SCRIPT))
prepare = MODULE["prepare_fakesound"]
STATE = prepare.__globals__


def wav_bytes(value: float, *, empty: bool = False) -> bytes:
    stream = io.BytesIO()
    sf.write(
        stream,
        np.full((0 if empty else 800, 2), value, dtype=np.float32),
        8000,
        format="WAV",
        subtype="FLOAT",
    )
    return stream.getvalue()


def make_archive(  # noqa: PLR0913 -- independent archive fixture variants
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bad_audio: bytes | None = None,
    extra: tuple[str | zipfile.ZipInfo, bytes] | None = None,
    mismatch: str | None = None,
    include_real: bool = False,
) -> Path:
    path = tmp_path / "FakeSound.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for index, bucket in enumerate(STATE["BUCKETS"]):
            base = "shared_clip" if bucket in ("easy", "hard") else bucket + "_clip"
            fake_id = base + "_0.01_0.09"
            rows = [
                {
                    "audio_id": audio_id,
                    "label": label,
                    "onset_offset": onset,
                    "filepath": f"FakeSound/deepfake_data/{bucket}/{audio_id}.wav",
                }
                for audio_id, label, onset in (
                    (fake_id, 1, "0.01_0.09"),
                    (base, 0, "0_0"),
                )
            ]
            if mismatch == "prefix":
                rows[0]["filepath"] = str(rows[0]["filepath"]).replace(
                    "deepfake_data", "other_data"
                )
            elif mismatch == "onset":
                rows[0]["onset_offset"] = "0.02_0.09"
            elif mismatch == "original":
                rows = rows[:1]
            archive.writestr(
                f"FakeSound/meta_data/{bucket}.json", json.dumps({"audios": rows})
            )
            archive.writestr(
                f"FakeSound/audio_data/{bucket}/{fake_id}.wav",
                bad_audio
                if bad_audio is not None and bucket == "easy"
                else wav_bytes(index * 0.1),
            )
            if include_real:
                archive.writestr(
                    f"FakeSound/audio_data/{bucket}/{base}.wav",
                    wav_bytes(index * 0.1 + 0.01),
                )
        if extra is not None:
            archive.writestr(*extra)
    monkeypatch.setitem(STATE, "ARCHIVE_BYTES", path.stat().st_size)
    monkeypatch.setitem(STATE, "ARCHIVE_SHA256", file_sha256(path))
    return path


def test_missing_originals_stay_in_manifest_and_difficulty_is_not_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = make_archive(tmp_path, monkeypatch)
    output = tmp_path / "diagnostic"
    summary = prepare(archive, output)
    records = load_manifest(output / "manifest.jsonl")
    assert len(records) == 8
    assert summary["counts"]["status"] == {"available": 4, "missing": 4}
    assert summary["decoded_label_counts"] == {"1": 4}
    assert summary["decode_coverage"] == 0.5
    assert summary["original_groups"] == 3
    assert summary["originals_crossing_upstream_buckets"] == {
        "shared_clip": ["easy", "hard"]
    }
    assert summary["formats"] == {"WAV/FLOAT/8000Hz/2ch": 4}
    assert summary["binary_eer_eligible"] is False
    conflict = summary["label_semantics_conflict"]
    assert conflict["readme_fake_raw_label"] == 0
    assert conflict["archive_manipulated_raw_label"] == 1
    assert conflict["internal_fake_label"] == 1
    assert all(record.split == "unassigned" for record in records)
    assert all(
        record.generator == record.identity_id == "unknown" for record in records
    )
    for record in records:
        if record.label == 0:
            assert record.status == "missing"
            assert record.failure_reason == "absent_from_zip"
            assert record.sha256 == record.decode_status == "unknown"
        else:
            assert record.decode_status == "ok"
            assert record.sample_rate == 8000
            assert record.duration_seconds == 0.1
            assert file_sha256(output / "media" / record.path) == record.sha256
            assert record.parent_id in {r.sample_id for r in records if r.label == 0}
    assert validate_splits(records, require_class_coverage=False) == []
    assigned = [
        replace(r, split="train" if ":easy:" in r.sample_id else "dev") for r in records
    ]
    assert any(
        "leakage" in e for e in validate_splits(assigned, require_class_coverage=False)
    )
    sources = load_sources(output / "sources.jsonl")
    assert all(source.license_status == "unresolved" for source in sources.values())
    assert all(source.allowed_uses == ("diagnostics",) for source in sources.values())
    provenance = [
        json.loads(line)
        for line in (output / "provenance.jsonl").read_text().splitlines()
    ]
    assert len(provenance) == 8
    assert {row["raw_label"] for row in provenance} == {0, 1}
    assert all(row["metadata_only"] == (row["raw_label"] == 0) for row in provenance)
    assert {row["upstream_bucket"] for row in provenance} == set(STATE["BUCKETS"])
    assert archive.exists()


@pytest.mark.parametrize(
    "bad_audio", [b"not a wav", wav_bytes(float("nan")), wav_bytes(0, empty=True)]
)
def test_decode_errors_are_recorded_without_reducing_coverage_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_audio: bytes
) -> None:
    archive = make_archive(tmp_path, monkeypatch, bad_audio=bad_audio)
    output = tmp_path / "diagnostic"
    summary = prepare(archive, output)
    records = load_manifest(output / "manifest.jsonl")
    assert summary["samples"] == 8
    assert summary["decode_coverage"] == 3 / 8
    failed = [record for record in records if record.status == "decode_failed"]
    assert len(failed) == 1
    assert failed[0].label == 1
    assert failed[0].decode_status == "failed"
    assert failed[0].failure_reason != "unknown"
    assert file_sha256(output / "media" / failed[0].path) == failed[0].sha256


def test_distribution_must_match_official_manipulated_only_description(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = make_archive(tmp_path, monkeypatch, include_real=True)
    with pytest.raises(ValueError, match="manipulated-only distribution"):
        prepare(archive, tmp_path / "diagnostic")
    assert not (tmp_path / "diagnostic").exists()


@pytest.mark.parametrize("field", ["ARCHIVE_BYTES", "ARCHIVE_SHA256"])
def test_source_mismatch_does_not_create_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    archive = make_archive(tmp_path, monkeypatch)
    monkeypatch.setitem(STATE, field, 1 if field == "ARCHIVE_BYTES" else "0" * 64)
    with pytest.raises(ValueError, match="pinned source"):
        prepare(archive, tmp_path / "diagnostic")
    assert not (tmp_path / "diagnostic").exists()


@pytest.mark.parametrize("mismatch", ["prefix", "onset", "original"])
def test_metadata_relationships_must_be_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    archive = make_archive(tmp_path, monkeypatch, mismatch=mismatch)
    with pytest.raises(ValueError, match=r"disagrees|onset_offset|original row"):
        prepare(archive, tmp_path / "diagnostic")
    assert not (tmp_path / "diagnostic").exists()


@pytest.mark.parametrize(
    "name", ["../escaped.wav", "/absolute.wav", "FakeSound/audio_data/unlisted.wav"]
)
def test_unsafe_or_unlabelled_members_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    archive = make_archive(tmp_path, monkeypatch, extra=(name, b"bad"))
    with pytest.raises(ValueError, match=r"unsafe|without metadata"):
        prepare(archive, tmp_path / "diagnostic")
    assert not (tmp_path / "diagnostic").exists()


def test_symlink_member_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = zipfile.ZipInfo("FakeSound/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = make_archive(tmp_path, monkeypatch, extra=(link, b"/outside"))
    with pytest.raises(ValueError, match="symlink"):
        prepare(archive, tmp_path / "diagnostic")


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_output_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symlink: bool
) -> None:
    archive = make_archive(tmp_path, monkeypatch)
    output = tmp_path / "diagnostic"
    if symlink:
        output.symlink_to(tmp_path / "absent")
    else:
        output.mkdir()
        (output / "keep.txt").write_text("existing result")
    with pytest.raises(FileExistsError, match="must not exist"):
        prepare(archive, output)
    if symlink:
        assert output.is_symlink()
    else:
        assert (output / "keep.txt").read_text() == "existing result"
