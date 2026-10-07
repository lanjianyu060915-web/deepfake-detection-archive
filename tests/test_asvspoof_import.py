"""Miniature protocol fixtures test import integrity, not audio or model quality."""

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

from deepfake_detection_archive.data.manifest import load_manifest, load_sources


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_asvspoof2019.py"
MODULE: dict[str, Any] = runpy.run_path(str(SCRIPT))
prepare = MODULE["prepare_asvspoof2019"]
PROTOCOLS: dict[str, str] = MODULE["PROTOCOLS"]
OFFICIAL_EXPECTED_COUNTS = MODULE["EXPECTED_COUNTS"]


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Only tests replace corpus-size constants; production has no bypass switch.
    monkeypatch.setitem(
        prepare.__globals__,
        "EXPECTED_COUNTS",
        {
            "train": {"speakers": 2, "real": 4, "fake": 4},
            "dev": {"speakers": 2, "real": 4, "fake": 4},
        },
    )
    root = tmp_path / "LA"
    for split, prefix, speakers in (
        ("train", "T", ("LA_0001", "LA_0002")),
        ("dev", "D", ("LA_0003", "LA_0004")),
    ):
        protocol = root / PROTOCOLS[split]
        protocol.parent.mkdir(parents=True, exist_ok=True)
        media = root / f"ASVspoof2019_LA_{split}" / "flac"
        media.mkdir(parents=True)
        rows = []
        for index in range(8):
            speaker = speakers[index // 4]
            sample_id = f"LA_{prefix}_{index:07d}"
            attack, label = ("-", "bonafide") if index % 4 < 2 else ("A01", "spoof")
            rows.append(f"{speaker} {sample_id} - {attack} {label}")
            # Intentionally not decodable: importer must not claim decoding.
            (media / f"{sample_id}.flac").write_bytes(
                f"fixture-{split}-{index}".encode()
            )
        protocol.write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "README.LA.txt").write_text(
        "Synthetic attribution fixture", encoding="utf-8"
    )
    (
        root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
    ).write_text("this eval protocol MUST NOT be parsed", encoding="utf-8")
    return root


def test_import_preserves_official_metadata_and_honest_unknowns(
    dataset: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "import"
    summary = prepare(dataset, destination)
    records = load_manifest(destination / "manifest.jsonl")
    assert len(records) == 16
    assert {r.split for r in records} == {"train", "dev"}
    assert all(r.path.startswith(f"ASVspoof2019_LA_{r.split}/flac/") for r in records)
    assert all(r.generator == ("human" if r.label == 0 else "A01") for r in records)
    assert all(
        r.identity_id == r.group_id and "speaker:LA_" in r.group_id for r in records
    )
    assert all(r.original_id == "unknown" and r.parent_id == "unknown" for r in records)
    assert all(
        r.decode_status == "unknown" and r.duration_seconds is None for r in records
    )
    assert all(r.sample_rate == 16000 and r.codec == "flac" for r in records)
    assert len({r.sha256 for r in records}) == 16
    assert summary["eval_accessed"] is False
    assert summary["competition_approved"] is False
    assert summary["decode_verified"] is False
    assert summary["official_counts_match"] is True
    assert summary["subset"] is False
    assert len(summary["protocol_sha256"]) == 2
    assert summary["attribution"]["local_files"][0]["path"] == "README.LA.txt"
    source = next(iter(load_sources(destination / "sources.jsonl").values()))
    assert source.license_id == "ODC-By-1.0"
    assert source.license_status == "approved"
    assert source.downloaded_at == "unknown"
    assert "training" in source.allowed_uses
    assert json.loads((destination / "import-summary.json").read_text()) == summary


def test_pilot_is_deterministic_and_speaker_stratified(
    dataset: Path, tmp_path: Path
) -> None:
    first = tmp_path / "pilot1"
    prepare(dataset, first, limit_per_class=2)
    for relative in PROTOCOLS.values():
        protocol = dataset / relative
        protocol.write_text(
            "\n".join(reversed(protocol.read_text().splitlines())) + "\n"
        )
    second = tmp_path / "pilot2"
    summary = prepare(dataset, second, limit_per_class=2)
    assert (first / "manifest.jsonl").read_bytes() == (
        second / "manifest.jsonl"
    ).read_bytes()
    assert summary["subset"] is True
    records = load_manifest(second / "manifest.jsonl")
    assert len(records) == 8
    for split in ("train", "dev"):
        for label in (0, 1):
            subset = [r for r in records if r.split == split and r.label == label]
            assert len({r.identity_id for r in subset}) == 2
    with pytest.raises(ValueError, match="cannot cover"):
        prepare(dataset, tmp_path / "too_small", limit_per_class=1)


def test_pilot_checks_missing_unselected_files(dataset: Path, tmp_path: Path) -> None:
    # Per-speaker sorted selection would skip item 7; complete validation cannot.
    (dataset / "ASVspoof2019_LA_train/flac/LA_T_0000007.flac").unlink()
    with pytest.raises(ValueError, match="required file missing"):
        prepare(dataset, tmp_path / "failed", limit_per_class=2)
    assert not (tmp_path / "failed").exists()


@pytest.mark.parametrize("violation", ["speaker", "hash", "label", "duplicate"])
def test_leakage_and_protocol_conflicts_fail(
    dataset: Path, tmp_path: Path, violation: str
) -> None:
    train = dataset / PROTOCOLS["train"]
    dev = dataset / PROTOCOLS["dev"]
    train_media = dataset / "ASVspoof2019_LA_train/flac"
    if violation == "speaker":
        dev.write_text(dev.read_text().replace("LA_0003", "LA_0001"))
        expected = "speaker crosses"
    elif violation == "hash":
        (dataset / "ASVspoof2019_LA_dev/flac/LA_D_0000000.flac").write_bytes(
            (train_media / "LA_T_0000000.flac").read_bytes()
        )
        expected = "bytes cross"
    elif violation == "label":
        (train_media / "LA_T_0000002.flac").write_bytes(
            (train_media / "LA_T_0000000.flac").read_bytes()
        )
        expected = "conflicting labels"
    else:
        train.write_text(train.read_text() + train.read_text().splitlines()[0] + "\n")
        expected = "duplicate protocol"
    with pytest.raises(ValueError, match=expected):
        prepare(dataset, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


@pytest.mark.parametrize(
    "bad_row",
    [
        "LA_0001 LA_T_0000000 - - real",
        "LA_0001 LA_T_0000000 - A07 spoof",
        "LA_0001 LA_T_0000000 - A01 bonafide",
        "LA_0001 ../escape - - bonafide",
        "LA_0001 LA_T_0000000 - bonafide",
    ],
)
def test_invalid_protocol_rejected(dataset: Path, tmp_path: Path, bad_row: str) -> None:
    protocol = dataset / PROTOCOLS["train"]
    rows = protocol.read_text().splitlines()
    rows[0] = bad_row
    protocol.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match=r"protocol|invalid|unknown|mismatch"):
        prepare(dataset, tmp_path / "failed")


def test_existing_artifacts_not_overwritten(dataset: Path, tmp_path: Path) -> None:
    destination = tmp_path / "import"
    prepare(dataset, destination)
    before = (destination / "manifest.jsonl").read_bytes()
    with pytest.raises(ValueError, match="already exist"):
        prepare(dataset, destination, limit_per_class=2)
    assert (destination / "manifest.jsonl").read_bytes() == before


@pytest.mark.parametrize("mismatch", ["truncated", "speakers", "production_counts"])
def test_incomplete_official_protocol_rejected_before_pilot(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    protocol = dataset / PROTOCOLS["train"]
    if mismatch == "truncated":
        # Both classes and all speakers survive, but a fake row is missing.
        protocol.write_text("\n".join(protocol.read_text().splitlines()[:-1]) + "\n")
    elif mismatch == "speakers":
        protocol.write_text(protocol.read_text().replace("LA_0002", "LA_0001"))
    else:
        # Actual production expectations reject the miniature fixture entirely.
        monkeypatch.setitem(
            prepare.__globals__, "EXPECTED_COUNTS", OFFICIAL_EXPECTED_COUNTS
        )
    destination = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="official train/dev counts mismatch"):
        prepare(dataset, destination, limit_per_class=2)
    assert not destination.exists()
