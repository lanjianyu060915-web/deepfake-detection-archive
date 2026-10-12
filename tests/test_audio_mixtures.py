"""Mixtures preserve labels, splits and symmetric nuisance signals."""

import json
import runpy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from deepfake_detection_archive.data.audit import audit_manifest
from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    SourceRecord,
    file_sha256,
    load_manifest,
    load_sources,
    write_manifest,
)


MODULE: dict[str, Any] = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/prepare_audio_mixtures.py")
)


def _fixture(tmp_path: Path) -> dict[str, Any]:
    music_root, speech_root = tmp_path / "music", tmp_path / "speech"
    music_root.mkdir()
    speech_root.mkdir()
    music, speech = [], []
    for split_index, split in enumerate(("train", "dev", "holdout")):
        for label in (0, 1):
            name = f"{split}-{label}"
            path = music_root / f"{name}.wav"
            if split != "holdout":
                signal = np.sin(
                    np.arange(64000)
                    * 2
                    * np.pi
                    * (200 + label * 80 + split_index * 13)
                    / 16000
                )
                sf.write(path, signal * (0.2 + label * 0.1), 16000)
            music.append(
                SampleRecord(
                    sample_id=name,
                    path=path.name,
                    modality="audio",
                    label=label,
                    source="music",
                    license_id="research",
                    content_type="music",
                    group_id=split,
                    sha256=file_sha256(path) if path.exists() else "unknown",
                    split=split,
                    decode_status="ok",
                    duration_seconds=4,
                    sample_rate=16000,
                )
            )
        name = f"speech-{split}"
        path = speech_root / f"{name}.wav"
        if split != "holdout":
            sf.write(
                path,
                np.sin(np.arange(96000) * 2 * np.pi * (500 + split_index * 23) / 16000)
                * 0.4,
                16000,
            )
        speech.append(
            SampleRecord(
                sample_id=name,
                path=path.name,
                modality="audio",
                label=0,
                source="speech",
                license_id="research",
                identity_id=name,
                group_id=name,
                split=split,
                sha256=file_sha256(path) if path.exists() else "unknown",
                decode_status="ok",
                duration_seconds=6,
                sample_rate=16000,
            )
        )
    for name, rows in (("music", music), ("speech", speech)):
        write_manifest(rows, tmp_path / f"{name}.jsonl")
        source = SourceRecord(
            source=name,
            url="https://example.org",
            version="test",
            license_id="research",
            license_url="https://example.org",
            downloaded_at="test",
            allowed_uses=("training", "research"),
            license_status="approved",
        )
        (tmp_path / f"{name}-sources.jsonl").write_text(
            json.dumps(asdict(source)) + "\n"
        )
    return {
        "music_manifest": tmp_path / "music.jsonl",
        "music_root": music_root,
        "music_sources": tmp_path / "music-sources.jsonl",
        "speech_manifest": tmp_path / "speech.jsonl",
        "speech_root": speech_root,
        "speech_sources": tmp_path / "speech-sources.jsonl",
        "output": tmp_path / "out",
    }


def test_symmetric_snr_replay_splits_and_determinism(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    result = MODULE["prepare"](**args)
    assert result["status_counts"] == {"available": 8}
    rows = load_manifest(args["output"] / "mixed-manifest.jsonl")
    originals = {r.sample_id: r for r in load_manifest(args["music_manifest"])}
    speech = {r.sample_id: r for r in load_manifest(args["speech_manifest"])}
    provenance = [
        json.loads(line)
        for line in (args["output"] / "provenance.jsonl").read_text().splitlines()
    ]
    for row, info in zip(
        rows, sorted(provenance, key=lambda p: p["sample_id"]), strict=True
    ):
        assert row.identity_id == f"speech-{row.split}"
        assert info["speech_crop_seconds"] == 1
        ratio = (
            info["music_gain"]
            * info["music_rms"]
            / (info["speech_gain"] * info["speech_rms"])
        )
        assert 20 * np.log10(ratio) == pytest.approx(info["snr_db"])
        original = originals[row.parent_id]
        voice = speech[info["speech_id"]]
        audio = sf.read(args["music_root"] / original.path)[0]
        background = sf.read(args["speech_root"] / voice.path)[0][16000:80000]
        expected = audio * info["music_gain"] + background * info["speech_gain"]
        actual = sf.read(args["output"] / row.path)[0]
        assert np.max(np.abs(actual)) <= 0.95004
        np.testing.assert_allclose(actual, expected, atol=1 / 32768)
    for split in ("train", "dev"):
        for snr in (-5, 5):
            matched = [
                p for p in provenance if p["split"] == split and p["snr_db"] == snr
            ]
            assert len(matched) == 2
            assert len({p["speech_id"] for p in matched}) == 1
    for row in load_manifest(args["output"] / "control-train-manifest.jsonl"):
        original = originals[row.parent_id]
        assert row.path == f"music/{original.path}"
        assert row.sha256 == original.sha256 == file_sha256(args["output"] / row.path)
        assert row.identity_id == original.identity_id
    assert all(r.split != "holdout" for r in rows)
    second = tmp_path / "again"
    MODULE["prepare"](**{**args, "output": second})
    assert (second / "mixed-manifest.jsonl").read_bytes() == (
        args["output"] / "mixed-manifest.jsonl"
    ).read_bytes()


def test_untrained_dev_dedup_and_train_rejected(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    originals = load_manifest(args["music_manifest"])
    real = next(r for r in originals if r.split == "dev" and r.label == 0)
    fake = next(r for r in originals if r.split == "dev" and r.label == 1)
    unseen = replace(fake, sample_id="unseen", generator="untrained")
    extra = tmp_path / "untrained.jsonl"
    write_manifest([real, unseen], extra)
    MODULE["prepare"](**args, untrained_manifest=extra)
    rows = load_manifest(args["output"] / "mixed-untrained-dev-manifest.jsonl")
    assert len(rows) == 4
    assert all(r.split == "dev" for r in rows)
    assert not any(
        r.generator == "untrained"
        for r in load_manifest(args["output"] / "mixed-manifest.jsonl")
    )
    write_manifest([replace(unseen, split="train")], extra)
    with pytest.raises(ValueError, match="dev only"):
        MODULE["prepare"](
            **{**args, "output": tmp_path / "invalid"}, untrained_manifest=extra
        )


@pytest.mark.parametrize("broken", ["missing", "corrupt", "silent"])
def test_media_failures_accounted(tmp_path: Path, broken: str) -> None:
    args = _fixture(tmp_path)
    path = args["music_root"] / "train-0.wav"
    if broken == "missing":
        path.unlink()
    elif broken == "corrupt":
        path.write_bytes(b"bad audio")
    else:
        sf.write(path, np.zeros(64000), 16000)
    result = MODULE["prepare"](**args)
    assert result["status_counts"] == {"available": 6, "decode_failed": 2}
    failed = [
        r
        for r in load_manifest(args["output"] / "mixed-manifest.jsonl")
        if r.status == "decode_failed"
    ]
    assert all(r.failure_reason != "unknown" and r.sha256 == "unknown" for r in failed)


def test_speaker_cross_split_rejected(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    speech = load_manifest(args["speech_manifest"])
    write_manifest(
        [replace(r, identity_id="same-speaker") for r in speech],
        args["speech_manifest"],
    )
    with pytest.raises(ValueError, match="leakage"):
        MODULE["prepare"](**args)
    assert not args["output"].exists()


def test_missing_speech_accounts_for_both_labels(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    (args["speech_root"] / "speech-train.wav").unlink()
    result = MODULE["prepare"](**args)
    assert result["status_counts"] == {"available": 4, "decode_failed": 4}
    rows = load_manifest(args["output"] / "mixed-manifest.jsonl")
    assert all(r.status == "decode_failed" for r in rows if r.split == "train")


def test_multiple_speech_choices_are_label_independent(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    speech = load_manifest(args["speech_manifest"])
    for original in tuple(speech):
        if original.split == "holdout":
            continue
        name = f"extra-{original.split}"
        path = args["speech_root"] / f"{name}.wav"
        sf.write(
            path,
            np.sin(np.arange(80000) * (0.13 if original.split == "train" else 0.21))
            * 0.3,
            16000,
        )
        speech.append(
            replace(
                original,
                sample_id=name,
                path=path.name,
                identity_id=name,
                group_id=name,
                sha256=file_sha256(path),
                duration_seconds=5,
            )
        )
    write_manifest(speech, args["speech_manifest"])
    result = MODULE["prepare"](**args)
    assert result["speech_pool_counts"] == {"train": 2, "dev": 2}
    rows = [
        json.loads(line)
        for line in (args["output"] / "provenance.jsonl").read_text().splitlines()
    ]
    for split in ("train", "dev"):
        for snr in (-5, 5):
            pairs = [
                row for row in rows if row["split"] == split and row["snr_db"] == snr
            ]
            assert len({row["speech_id"] for row in pairs}) == 1
            assert len({row["speech_crop_seconds"] for row in pairs}) == 1


def test_mixture_registry_passes_training_input_audit(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    MODULE["prepare"](**args)
    records = load_manifest(args["output"] / "mixed-manifest.jsonl")
    sources = load_sources(args["output"] / "sources.jsonl")
    assert sources[MODULE["SOURCE"]].license_id == "research AND research"
    assert all(row.license_id == sources[row.source].license_id for row in records)
    # Match experiments.py: selected training inputs are audited without requiring
    # a holdout partition; the preparer already validates original split relations.
    report = audit_manifest(
        [replace(row, split="unassigned") for row in records],
        args["output"],
        sources,
    )
    assert report.passed, report.issues
    assert len(report.trainable_ids) == len(records)


@pytest.mark.parametrize("parent", ["music", "speech"])
@pytest.mark.parametrize(
    ("status", "uses"),
    [
        ("denied", ("research", "training")),
        ("unresolved", ("research", "training")),
        ("approved", ("research", "evaluation")),
    ],
)
def test_parent_training_permission_required_before_output(
    tmp_path: Path, parent: str, status: str, uses: tuple[str, ...]
) -> None:
    args = _fixture(tmp_path)
    path = args[f"{parent}_sources"]
    source = replace(
        load_sources(path)[parent], license_status=status, allowed_uses=uses
    )
    path.write_text(json.dumps(asdict(source)) + "\n")
    with pytest.raises(ValueError, match="training permission not approved"):
        MODULE["prepare"](**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("parent", ["music", "speech"])
def test_parent_license_mismatch_rejected_before_output(
    tmp_path: Path, parent: str
) -> None:
    args = _fixture(tmp_path)
    path = args[f"{parent}_manifest"]
    write_manifest(
        [replace(row, license_id="different") for row in load_manifest(path)], path
    )
    with pytest.raises(ValueError, match="sample/source license mismatch"):
        MODULE["prepare"](**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("music_uses", [("training", "research"), ("training",)])
def test_real_parent_licenses_preserve_allowed_uses_intersection(
    tmp_path: Path, music_uses: tuple[str, ...]
) -> None:
    args = _fixture(tmp_path)
    licenses = {
        "music": "CC-BY-4.0 AND CC-BY-NC-SA-4.0 AND LicenseRef-MTG-CC-per-track",
        "speech": "ODC-By-1.0",
    }
    for parent, license_id in licenses.items():
        manifest = args[f"{parent}_manifest"]
        write_manifest(
            [replace(row, license_id=license_id) for row in load_manifest(manifest)],
            manifest,
        )
        path = args[f"{parent}_sources"]
        source = replace(
            load_sources(path)[parent],
            license_id=license_id,
            allowed_uses=(
                music_uses
                if parent == "music"
                else ("research", "training", "evaluation")
            ),
        )
        path.write_text(json.dumps(asdict(source)) + "\n")
    MODULE["prepare"](**args)
    derived = load_sources(args["output"] / "sources.jsonl")[MODULE["SOURCE"]]
    assert set(derived.allowed_uses) == set(music_uses)
    assert derived.license_id == f"{licenses['music']} AND {licenses['speech']}"
    assert derived.license_status == "approved"
    assert all(
        row.license_id == derived.license_id
        for row in load_manifest(args["output"] / "mixed-manifest.jsonl")
    )


def test_distinct_license_pairs_rejected_before_output(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    source = load_sources(args["music_sources"])["music"]
    other = replace(source, source="other-music", license_id="other-license")
    args["music_sources"].write_text(
        "".join(json.dumps(asdict(item)) + "\n" for item in (source, other))
    )
    music = load_manifest(args["music_manifest"])
    music[0] = replace(music[0], source=other.source, license_id=other.license_id)
    write_manifest(music, args["music_manifest"])
    with pytest.raises(ValueError, match="single music/speech license pair"):
        MODULE["prepare"](**args)
    assert not args["output"].exists()


def test_unused_holdout_sources_do_not_restrict_mixtures(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    for parent in ("music", "speech"):
        path = args[f"{parent}_manifest"]
        write_manifest(
            [
                replace(row, source="unregistered") if row.split == "holdout" else row
                for row in load_manifest(path)
            ],
            path,
        )
    result = MODULE["prepare"](**args)
    assert result["status_counts"] == {"available": 8}
