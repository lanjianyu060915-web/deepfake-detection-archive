"""Behavioral checks for provenance, byte integrity and split leakage."""

from dataclasses import replace
from pathlib import Path

import pytest

from deepfake_detection_archive.data.audit import audit_manifest, write_audit_report
from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    SourceRecord,
    file_sha256,
    load_manifest,
    load_sources,
    manifest_hash,
    sample_from_dict,
    write_manifest,
)
from deepfake_detection_archive.data.split import (
    assign_splits,
    connected_groups,
    validate_splits,
)


def sample(sample_id: str = "one", **changes: object) -> SampleRecord:
    values: dict[str, object] = {
        "sample_id": sample_id,
        "path": f"{sample_id}.wav",
        "modality": "audio",
        "label": 0,
        "source": "licensed",
    }
    values.update(changes)
    return sample_from_dict(values)


def licensed() -> dict[str, SourceRecord]:
    return {
        "licensed": SourceRecord(
            "licensed",
            "https://example.org/dataset",
            "v1",
            "CC0-1.0",
            "https://example.org/license",
            "2026-10-07",
            ("training", "evaluation"),
            "approved",
        )
    }


def test_manifest_round_trip_and_hash_is_order_independent(tmp_path: Path) -> None:
    records = [sample("b", label=1), sample("a", label=None)]
    path = tmp_path / "manifest.jsonl"
    checksum = write_manifest(records, path)
    assert checksum == manifest_hash(records[::-1]) == file_sha256(path)
    assert load_manifest(path) == records[::-1]
    assert manifest_hash([replace(records[0], split="dev"), records[1]]) != checksum


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "../secret"),
        ("path", "/absolute"),
        ("path", "C:/secret"),
        ("path", "x\\y"),
        ("path", "."),
        ("sha256", "ABC"),
        ("label", True),
        ("label", "fake"),
        ("label", 2),
        ("modality", "image"),
        ("sample_id", "unknown"),
        ("source", ""),
        ("duration_seconds", float("nan")),
        ("frame_rate", -1),
        ("sample_rate", 16000.5),
        ("status", "missing"),
        ("decode_status", "failed"),
        ("split", "validation"),
    ],
)
def test_schema_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=r"."):
        sample(**{field: value})


def test_manifest_rejects_duplicate_keys_ids_and_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"sample_id":"x","sample_id":"y"}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_manifest(path)
    write_manifest([sample()], path)
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_manifest(path)
    with pytest.raises(ValueError, match="unknown sample fields"):
        sample(typo="value")
    path.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        load_manifest(path)


def test_sources_strict_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(
        '{"source":"s","url":"u","version":"v1","license_id":"L",'
        '"license_url":"l","downloaded_at":"2026-10-07",'
        '"allowed_uses":["training"],"license_status":"approved"}\n'
    )
    assert load_sources(path)["s"].allowed_uses == ("training",)
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate source"):
        load_sources(path)


def test_transitive_groups_link_parent_identity_hash_and_duplicates() -> None:
    records = [
        sample("a", sha256="a" * 64),
        sample("b", sha256="a" * 64, identity_id="person"),
        sample("c", identity_id="person"),
        sample("d", parent_id="c"),
        sample("e"),
        sample("separate"),
    ]
    mapping = connected_groups(records, (("d", "e"),))
    assert {mapping[x] for x in "abcde"} == {"a"}
    assert mapping["separate"] == "separate"
    assert mapping == connected_groups(records[::-1], (("e", "d"),))
    with pytest.raises(ValueError, match="absent sample"):
        connected_groups(records, (("a", "absent"),))


def test_external_parent_siblings_and_source_scoping() -> None:
    records = [
        sample("a", parent_id="external"),
        sample("b", parent_id="external"),
        sample("c", original_id="entry"),
        sample("d", original_id="entry", source="other"),
    ]
    mapping = connected_groups(records)
    assert mapping["a"] == mapping["b"]
    assert mapping["c"] != mapping["d"]


def test_split_is_deterministic_balanced_and_keeps_derivatives_together() -> None:
    records = [
        sample(
            f"{modality}-{label}-{group}-{derivative}",
            label=label,
            modality=modality,
            group_id=f"{modality}-{label}-{group}",
        )
        for modality in ("audio", "video")
        for label in (0, 1)
        for group in range(20)
        for derivative in range(2)
    ]
    result = assign_splits(records, seed=37)
    reversed_result = assign_splits(records[::-1], seed=37)
    assert manifest_hash(result) == manifest_hash(reversed_result)
    assert validate_splits(result) == []
    for split, expected in (("train", 112), ("dev", 24), ("holdout", 24)):
        assert sum(r.split == split for r in result) == expected
    assert manifest_hash(result) != manifest_hash(assign_splits(records, seed=38))


def test_fixed_holdout_is_never_reassigned() -> None:
    records = [
        sample(f"{label}-{group}", label=label)
        for label in (0, 1)
        for group in range(6)
    ]
    records[0] = replace(records[0], split="holdout")
    result = assign_splits(records)
    assert result[0].split == "holdout"
    assert assign_splits(result, seed=999) == result


def test_split_reports_insufficient_groups_and_rejects_unlabelled_samples() -> None:
    with pytest.raises(ValueError, match="insufficient independent groups"):
        assign_splits([sample("real"), sample("fake", label=1)])
    with pytest.raises(ValueError, match="unlabelled"):
        assign_splits([sample(label=None)])
    with pytest.raises(ValueError, match="both classes"):
        assign_splits([sample()])
    with pytest.raises(ValueError, match="ratios"):
        assign_splits([sample()], ratios=(0.8, 0.2, 0.2))


def test_split_leakage_is_detected_before_assignment() -> None:
    records = [
        sample("a", group_id="g", split="train"),
        sample("b", group_id="g", split="dev"),
    ]
    assert validate_splits(records, require_class_coverage=False) == [
        "leakage: group a spans ['dev', 'train']"
    ]
    with pytest.raises(ValueError, match="existing split leakage"):
        assign_splits([*records, sample("fake", label=1)])


def test_audit_requires_actual_bytes_approved_license_and_decoding(
    tmp_path: Path,
) -> None:
    media = tmp_path / "one.wav"
    media.write_bytes(b"test-only bytes; decode proof mocked in this unit test")
    record = sample(sha256=file_sha256(media), license_id="CC0-1.0", decode_status="ok")
    report = audit_manifest([record], tmp_path, licensed())
    assert report.passed
    assert report.trainable_ids == ("one",)
    assert report.coverage["audio/unknown"]["status"] == "unverified"
    assert report.coverage["audio/music"]["samples"] == 0
    assert report.coverage["audio/music"]["status"] == "unverified"
    write_audit_report(report, tmp_path / "report.json")
    assert '"passed": true' in (tmp_path / "report.json").read_text()
    changed = replace(record, decode_status="unknown")
    assert not audit_manifest([changed], tmp_path, licensed()).passed
    sources = licensed()
    sources["licensed"] = replace(sources["licensed"], allowed_uses=("evaluation",))
    assert not audit_manifest([record], tmp_path, sources).passed
    media.write_bytes(b"changed")
    report = audit_manifest([record], tmp_path, licensed())
    assert any(i.code == "hash_mismatch" for i in report.issues)
    assert report.trainable_ids == ()


def test_audit_missing_entries_remain_in_denominators(tmp_path: Path) -> None:
    record = sample(status="missing", failure_reason="download not yet available")
    report = audit_manifest([record], tmp_path, {})
    assert report.total_samples == 1
    assert report.counts["status"] == {"missing": 1}
    assert not report.passed
    assert report.trainable_ids == ()
    assert {issue.code for issue in report.issues} >= {
        "missing_file",
        "unavailable",
        "unknown_hash",
        "unregistered_source",
    }


def test_audit_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"not a media file")
    (root / "one.wav").symlink_to(outside)
    report = audit_manifest([sample()], root, {})
    assert any(i.code == "path_escape" for i in report.issues)
    assert report.total_bytes == 0


def test_actual_hash_detects_hidden_duplicates_and_label_conflicts(
    tmp_path: Path,
) -> None:
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"same bytes")
    records = [sample("a", split="train"), sample("b", label=1, split="holdout")]
    report = audit_manifest(records, tmp_path, licensed())
    assert {i.code for i in report.issues} >= {"label_conflict", "actual_hash_leakage"}
    assert report.trainable_ids == ()
