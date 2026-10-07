"""Miniature ViF archives verify scope, provenance and source-video grouping."""

import hashlib
import json
import runpy
import stat
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from deepfake_detection_archive.data.manifest import (
    file_sha256,
    load_manifest,
    load_sources,
)
from deepfake_detection_archive.data.split import (
    connected_groups,
    statistical_groups,
    validate_splits,
)


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_vif_panda.py"
MODULE: dict[str, Any] = runpy.run_path(str(SCRIPT))
prepare = MODULE["prepare_panda"]
inspect = MODULE["inspect_archive"]
STATE = prepare.__globals__


def fixture_rows() -> list[dict[str, str]]:
    return [
        {
            "video_id": video_id,
            "video_path": f"source_videos/{dataset}/real/{model}/{video_id}.mp4",
            "aigc_video_path": f"source_videos/{dataset}/fake/{model}/{video_id}.mp4",
            "aigc_model": model,
        }
        for dataset, video_id, model in (
            ("Panda-70M", "ABCDEFGHIJK-0", "Model-A"),
            ("Panda-70M", "ABCDEFGHIJK-1", "Model-B"),
            ("Panda-70M", "0123456789_-0", "Model-A"),
            ("kinetics", "Kinetics_clip", "Model-B"),
        )
    ]


def archive_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: list[dict[str, str]] | None = None,
    extra: tuple[str | zipfile.ZipInfo, bytes] | None = None,
) -> Path:
    rows = fixture_rows() if rows is None else rows
    path = tmp_path / "source_videos.zip"
    metadata = json.dumps(rows).encode()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(STATE["NORMALIZED_PATH"], metadata)
        for row in rows:
            for key in ("video_path", "aigc_video_path"):
                archive.writestr(row[key], row[key].encode())
        if extra is not None:
            archive.writestr(*extra)
    monkeypatch.setitem(STATE, "ARCHIVE_BYTES", path.stat().st_size)
    monkeypatch.setitem(STATE, "ARCHIVE_SHA256", file_sha256(path))
    monkeypatch.setitem(
        STATE, "NORMALIZED_SHA256", hashlib.sha256(metadata).hexdigest()
    )
    monkeypatch.setitem(STATE, "EXPECTED_PAIRS", 4)
    monkeypatch.setitem(STATE, "EXPECTED_PANDA_PAIRS", 3)
    monkeypatch.setitem(STATE, "WORKSPACE_ROOT", tmp_path)
    return path


def evidence(tmp_path: Path) -> tuple[Path, list[Path]]:
    naming, license_file = tmp_path / "naming.txt", tmp_path / "license.txt"
    naming.write_text("synthetic reviewed naming evidence")
    license_file.write_text("synthetic noncommercial-research license evidence")
    (tmp_path / "source-map.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metadata_source_url": "https://example.org/primary-panda.csv",
                "metadata_sha256": file_sha256(naming),
                "clips": {
                    row["video_id"]: {
                        "url": "https://www.youtube.com/watch?v="
                        + (
                            "ZYXWVUTSRQP"
                            if "ABCDEFGHIJK" in row["video_id"]
                            else "9876543210_"
                        ),
                        "timestamp": [0.0, 5.0],
                    }
                    for row in fixture_rows()
                    if "/Panda-70M/" in row["video_path"]
                },
            }
        )
    )
    return naming, [license_file]


def test_import_only_panda_and_preserve_source_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = archive_fixture(tmp_path, monkeypatch)
    naming, licenses = evidence(tmp_path)
    original_open = zipfile.ZipFile.open
    opened = []

    def guarded_open(
        self: zipfile.ZipFile, name: Any, *args: Any, **kwargs: Any
    ) -> Any:
        filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
        assert "/kinetics/" not in filename, "Kinetics bodies must not be read"
        opened.append(filename)
        return original_open(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", guarded_open)
    output = tmp_path / "data/vif-panda-research"
    summary = prepare(
        archive,
        output,
        naming_evidence=naming,
        license_evidence=licenses,
        source_mapping=tmp_path / "source-map.json",
    )
    records = load_manifest(output / "manifest.jsonl")
    assert len(records) == 6
    assert summary["source_video_group_count"] == 2
    assert summary["competition_approved"] is False
    assert summary["media_decoded"] is False
    assert summary["kinetics_member_contents_read"] is False
    assert summary["redistribute_media_allowed"] is False
    assert {record.group_id for record in records if "ABCDEFGHIJK" in record.path} == {
        "panda70m:youtube:ZYXWVUTSRQP"
    }
    assert all(
        record.identity_id == "unknown"
        and record.decode_status == "unknown"
        and record.split == "unassigned"
        for record in records
    )
    assert all(record.generator == "unknown" for record in records if record.label == 0)
    assert {record.generator for record in records if record.label == 1} == {
        "Model-A",
        "Model-B",
    }
    assert all(
        file_sha256(output / "media" / record.path) == record.sha256
        for record in records
    )
    assert all("kinetics" not in path.as_posix() for path in output.rglob("*"))
    sources = load_sources(output / "sources.jsonl")
    assert all(
        source.license_status == "approved"
        and "local_noncommercial_computational_research" in source.allowed_uses
        for source in sources.values()
    )
    assert archive.exists()
    assert not list(output.parent.glob(".vif-panda-research.staging-*"))


def test_audit_only_never_opens_any_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = archive_fixture(tmp_path, monkeypatch)
    original_open = zipfile.ZipFile.open

    def metadata_only(
        self: zipfile.ZipFile, name: Any, *args: Any, **kwargs: Any
    ) -> Any:
        filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
        assert filename == STATE["NORMALIZED_PATH"]
        return original_open(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", metadata_only)
    summary = inspect(path)
    assert summary["panda_pair_count"] == 3
    assert summary["all_pair_count"] == 4


@pytest.mark.parametrize(
    "constant", ["ARCHIVE_BYTES", "ARCHIVE_SHA256", "NORMALIZED_SHA256"]
)
def test_checksum_failures_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, constant: str
) -> None:
    path = archive_fixture(tmp_path, monkeypatch)
    monkeypatch.setitem(STATE, constant, 1 if constant == "ARCHIVE_BYTES" else "0" * 64)
    with pytest.raises(ValueError, match=r"verification|checksum"):
        inspect(path)


@pytest.mark.parametrize(
    "extra", ["../escape.mp4", "/absolute", "source_videos/unused.mp4"]
)
def test_unsafe_or_unregistered_archive_members_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: str
) -> None:
    path = archive_fixture(tmp_path, monkeypatch, extra=(extra, b"bad"))
    with pytest.raises(ValueError, match=r"unsafe|differ"):
        inspect(path)


def test_symlink_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    info = zipfile.ZipInfo("source_videos/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    path = archive_fixture(tmp_path, monkeypatch, extra=(info, b"outside"))
    with pytest.raises(ValueError, match="symlink"):
        inspect(path)


def test_pair_roles_generator_and_path_must_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = fixture_rows()
    rows[0]["aigc_model"] = "different-generator"
    path = archive_fixture(tmp_path, monkeypatch, rows=rows)
    with pytest.raises(ValueError, match="disagree"):
        inspect(path)


def test_naming_mismatch_aborts_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = fixture_rows()
    for key in ("video_id", "video_path", "aigc_video_path"):
        rows[0][key] = rows[0][key].replace("ABCDEFGHIJK-0", "unknown-format")
    archive = archive_fixture(tmp_path, monkeypatch, rows=rows)
    naming, licenses = evidence(tmp_path)
    with pytest.raises(ValueError, match="cover exactly"):
        prepare(
            archive,
            tmp_path / "out",
            naming_evidence=naming,
            license_evidence=licenses,
            source_mapping=tmp_path / "source-map.json",
        )
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.staging-*"))


@pytest.mark.parametrize(
    "constant",
    ["CACHE_BUDGET", "ASSET_BUDGET", "MAX_MEMBER_BYTES", "MAX_EXTRACT_BYTES"],
)
def test_acquisition_and_extraction_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, constant: str
) -> None:
    archive = archive_fixture(tmp_path, monkeypatch)
    naming, licenses = evidence(tmp_path)
    monkeypatch.setitem(STATE, constant, 1)
    with pytest.raises(ValueError, match="budget"):
        prepare(
            archive,
            tmp_path / "out",
            naming_evidence=naming,
            license_evidence=licenses,
            source_mapping=tmp_path / "source-map.json",
        )
    assert not (tmp_path / "out").exists()


def test_no_overwrite_and_missing_review_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = archive_fixture(tmp_path, monkeypatch)
    naming, licenses = evidence(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(FileExistsError, match="must not exist"):
        prepare(
            archive,
            output,
            naming_evidence=naming,
            license_evidence=licenses,
            source_mapping=tmp_path / "source-map.json",
        )
    with pytest.raises(ValueError, match="evidence"):
        prepare(
            archive,
            tmp_path / "out2",
            naming_evidence=tmp_path / "absent",
            license_evidence=licenses,
            source_mapping=tmp_path / "source-map.json",
        )


def test_duplicate_members_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        archive = archive_fixture(
            tmp_path, monkeypatch, extra=(fixture_rows()[0]["video_path"], b"duplicate")
        )
    with pytest.raises(ValueError, match="duplicate"):
        inspect(archive)


def test_mapping_must_reference_actual_primary_evidence_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = archive_fixture(tmp_path, monkeypatch)
    naming, licenses = evidence(tmp_path)
    naming.write_text("changed source evidence")
    with pytest.raises(ValueError, match="metadata hash differs"):
        prepare(
            archive,
            tmp_path / "out",
            naming_evidence=naming,
            license_evidence=licenses,
            source_mapping=tmp_path / "source-map.json",
        )
    assert not (tmp_path / "out").exists()


def test_import_requires_mapping_even_when_license_evidence_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = archive_fixture(tmp_path, monkeypatch)
    naming, licenses = evidence(tmp_path)
    assert (
        MODULE["main"](
            [
                "--archive",
                str(archive),
                "--output",
                str(tmp_path / "out"),
                "--naming-evidence",
                str(naming),
                "--license-evidence",
                str(licenses[0]),
            ]
        )
        == 2
    )
    assert "explicit source mapping" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_exploratory_edges_do_not_claim_statistical_independence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = archive_fixture(tmp_path, monkeypatch)
    naming, licenses = evidence(tmp_path)
    output = tmp_path / "exploratory"
    summary = prepare(
        archive,
        output,
        naming_evidence=naming,
        license_evidence=licenses,
        exploratory=True,
    )
    records = load_manifest(output / "manifest.jsonl")
    assert summary["source_video_group_count"] == 0
    assert summary["suspected_component_count"] == 2
    assert summary["independent_group_ci_allowed"] is False
    assert summary["formal_promotion_allowed"] is False
    assert summary["local_research_only"] is True
    assert all(
        r.group_id == r.identity_id == r.original_id == "unknown" for r in records
    )
    assert set(statistical_groups(records).values()) == {"unknown"}
    linked = [r for r in records if "ABCDEFGHIJK" in r.path]
    assert len({connected_groups(records)[r.sample_id] for r in linked}) == 1
    assert {r.generator for r in linked if r.label == 1} == {"Model-A", "Model-B"}
    assert all(r.parent_id != "unknown" for r in records if r.label == 1)
    sidecar = json.loads((output / "grouping-evidence.json").read_text())
    assert sidecar["proves_common_original_recording"] is False
    assert sidecar["rule"] == "suspected_filename_component:v1"
    assert len(sidecar["components"]) == 2
    # Deliberately conflicting assignments to related clips must fail validation.
    assigned = [replace(r, split="train") for r in records]
    assigned[2] = replace(assigned[2], split="dev")
    assert any(
        "leakage" in e for e in validate_splits(assigned, require_class_coverage=False)
    )
    # An identical media hash bridges different suspected filename components.
    bridged = list(records)
    bridged[-1] = replace(bridged[-1], sha256=bridged[0].sha256)
    assert len(set(connected_groups(bridged).values())) == 1
    assert set(statistical_groups(bridged).values()) == {"unknown"}


def test_exploratory_and_verified_mapping_are_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = archive_fixture(tmp_path, monkeypatch)
    naming, licenses = evidence(tmp_path)
    with pytest.raises(ValueError, match="exclusively"):
        prepare(
            archive,
            tmp_path / "out",
            naming_evidence=naming,
            license_evidence=licenses,
            source_mapping=tmp_path / "source-map.json",
            exploratory=True,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("racing", [False, True])
def test_audit_never_overwrites_symlink_or_concurrently_created_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, racing: bool
) -> None:
    archive = archive_fixture(tmp_path, monkeypatch)
    report = tmp_path / "audit.json"
    if racing:

        def competing_inspect(path: Path) -> dict[str, object]:
            result: dict[str, object] = inspect(path)
            report.write_text("other process owns this report")
            return result

        monkeypatch.setitem(STATE, "inspect_archive", competing_inspect)
    else:
        report.symlink_to(tmp_path / "missing.json")
    assert (
        MODULE["main"](
            ["--archive", str(archive), "--output", str(report), "--audit-only"]
        )
        == 2
    )
    if racing:
        assert report.read_text() == "other process owns this report"
    else:
        assert report.is_symlink()
        assert not report.exists()
    assert not list(tmp_path.glob(".vif-audit-staging-*"))
