"""Synthetic ViF-Bench archives verify pairing, exclusion and import invariants."""

import json
import runpy
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    file_sha256,
    load_manifest,
)


TOOLS = Path(__file__).resolve().parents[1] / "tools"
# The importer runs as a script, so its directory provides the sibling helper
# module `prepare_vif_panda`; mirror that here.
sys.path.insert(0, str(TOOLS))
SCRIPT = TOOLS / "prepare_vif_bench.py"
MODULE: dict[str, Any] = runpy.run_path(str(SCRIPT))
prepare = MODULE["prepare"]
audit = MODULE["audit"]
STATE = prepare.__globals__
INDEX_PREFIX = STATE["INDEX_PREFIX"]


def members() -> dict[str, bytes]:
    return {
        "source_videos/Real/AAAAAAAAAAA-0.mp4": b"real-a",
        "source_videos/Real/BBBBBBBBBBB-1.mp4": b"real-b",
        "source_videos/Fake/Model-A/AAAAAAAAAAA-0.mp4": b"fake-a",
        "source_videos/Fake/Model-A/CCCCCCCCCCC-2.mp4": b"fake-unpaired",
        "source_videos/Fake/Model-B/BBBBBBBBBBB-1.mp4": b"fake-b",
        "source_videos/ladm_paths.json": json.dumps(
            {
                "Fake/Model-A": [
                    INDEX_PREFIX + "Fake/Model-A/AAAAAAAAAAA-0.mp4",
                    INDEX_PREFIX + "Fake/Model-A/CCCCCCCCCCC-2.mp4",
                ],
                "Fake/Model-B": [INDEX_PREFIX + "Fake/Model-B/BBBBBBBBBBB-1.mp4"],
                "Real": [
                    INDEX_PREFIX + "Real/AAAAAAAAAAA-0.mp4",
                    INDEX_PREFIX + "Real/BBBBBBBBBBB-1.mp4",
                ],
            }
        ).encode(),
    }


def fixture_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, bytes] | None = None,
) -> Path:
    path = tmp_path / "source_videos.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in (payload or members()).items():
            archive.writestr(name, content)
    monkeypatch.setitem(STATE, "ARCHIVE_BYTES", path.stat().st_size)
    monkeypatch.setitem(STATE, "ARCHIVE_SHA256", file_sha256(path))
    return path


def exclusion_manifest(tmp_path: Path, path_string: str) -> Path:
    record = SampleRecord(
        sample_id="vif:" + path_string,
        path=path_string,
        modality="video",
        label=0,
        source="fixture",
    )
    target = tmp_path / "exclusions.jsonl"
    target.write_text(
        json.dumps(record.to_dict(), sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def test_identical_names_pair_and_unmatched_fake_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = fixture_archive(tmp_path, monkeypatch)
    summary = prepare(archive, tmp_path / "import", [])
    rows = load_manifest(tmp_path / "import" / "manifest.jsonl")
    by_id = {row.sample_id: row for row in rows}

    assert summary["kept_real_count"] == 2
    assert len(rows) == 4
    assert {row.split for row in rows} == {"dev"}
    assert {row.group_id for row in rows} == {"unknown"}
    assert {row.original_id for row in rows} == {"unknown"}
    fake_a = by_id["vifbench:source_videos/Fake/Model-A/AAAAAAAAAAA-0.mp4"]
    assert fake_a.label == 1
    assert fake_a.generator == "Model-A"
    assert fake_a.parent_id == "vifbench:source_videos/Real/AAAAAAAAAAA-0.mp4"
    assert by_id[fake_a.parent_id].label == 0
    assert not any("CCCCCCCCCCC" in row.sample_id for row in rows), (
        "unmatched fake has no real counterpart"
    )
    assert summary["unpaired_fake_count"] == 1


def test_paired_fake_keeps_media_bytes_and_diagnostics_subsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = fixture_archive(tmp_path, monkeypatch)
    prepare(archive, tmp_path / "import", [])
    output = tmp_path / "import"
    rows = load_manifest(output / "manifest.jsonl")
    for row in rows:
        assert file_sha256(output / "media" / row.path) == row.sha256
    copied = output / "media" / "source_videos/Fake/Model-A/AAAAAAAAAAA-0.mp4"
    assert copied.read_bytes() == b"fake-a"

    index = json.loads((output / "diagnostics/index.json").read_text())
    assert set(index) == {"Model-A", "Model-B"}
    assert index["Model-A"]["real_count"] == 1
    assert index["Model-A"]["fake_count"] == 1
    subset = load_manifest(output / index["Model-A"]["path"])
    assert {row.label for row in subset} == {0, 1}
    assert {row.path for row in subset} == {
        "source_videos/Real/AAAAAAAAAAA-0.mp4",
        "source_videos/Fake/Model-A/AAAAAAAAAAA-0.mp4",
    }


def test_exclusion_removes_the_whole_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = fixture_archive(tmp_path, monkeypatch)
    excluded = exclusion_manifest(tmp_path, "source_videos/Real/AAAAAAAAAAA-0.mp4")
    summary = prepare(archive, tmp_path / "import", [excluded])
    rows = load_manifest(tmp_path / "import" / "manifest.jsonl")

    assert summary["excluded_real_count"] == 1
    assert summary["excluded_fake_count"] == 1
    assert summary["unpaired_fake_count"] == 1
    assert len(rows) == 2
    assert all("AAAAAAAAAAA" not in row.path for row in rows)
    assert {row.label for row in rows} == {0, 1}


def test_unexpected_member_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = members()
    payload["source_videos/Extra/notes.mp4"] = b"x"
    archive = fixture_archive(tmp_path, monkeypatch, payload)
    with pytest.raises(ValueError, match="unexpected archive member"):
        audit(archive)


def test_index_must_reference_present_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = members()
    payload["source_videos/ladm_paths.json"] = json.dumps(
        {"Fake/Model-A": [INDEX_PREFIX + "Fake/Model-A/DDDDDDDDDDD-3.mp4"]}
    ).encode()
    archive = fixture_archive(tmp_path, monkeypatch, payload)
    with pytest.raises(ValueError, match="references absent media"):
        audit(archive)


def test_wrong_archive_identity_and_existing_output_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = fixture_archive(tmp_path, monkeypatch)
    monkeypatch.setitem(STATE, "ARCHIVE_BYTES", archive.stat().st_size + 1)
    with pytest.raises(ValueError, match="fixed revision"):
        prepare(archive, tmp_path / "import", [])

    monkeypatch.setitem(STATE, "ARCHIVE_BYTES", archive.stat().st_size)
    target = tmp_path / "import"
    prepare(archive, target, [])
    with pytest.raises(FileExistsError):
        prepare(archive, target, [])


def test_audit_reports_layout_without_copying_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = fixture_archive(tmp_path, monkeypatch)
    summary = audit(archive)
    assert summary["real_count"] == 2
    assert summary["fake_count"] == 3
    assert summary["generators"] == {"Model-A": 2, "Model-B": 1}
    assert summary["paired_fake_count"] == {"Model-A": 1, "Model-B": 1}
    assert summary["index_declared_count"] == 5
    assert summary["index_matches_media"] is True
    assert summary["media_decoded"] is False
    assert list(tmp_path.glob("**/media")) == []


@pytest.mark.parametrize(
    ("widths", "expected"),
    [
        ([100, 200, 300, 400], {"increasing": 0.0, "decreasing": 1.0, "oracle": 0.0}),
        ([300, 400, 100, 200], {"increasing": 1.0, "decreasing": 0.0, "oracle": 0.0}),
        ([100, 100, 100, 100], {"increasing": 0.5, "decreasing": 0.5, "oracle": 0.5}),
        (
            [100, 200, 100, 100],
            {"increasing": 2 / 3, "decreasing": 1 / 3, "oracle": 1 / 3},
        ),
    ],
)
def test_width_baseline_uses_both_directions_and_interpolated_ties(
    widths: list[int], expected: dict[str, float]
) -> None:
    probe_module = runpy.run_path(
        str(
            TOOLS.parent
            / "docs/experiments/video-generalization-20261011/metadata-probe.py"
        )
    )
    rows = [
        {"width": width, "label": label}
        for width, label in zip(widths, [0, 0, 1, 1], strict=True)
    ]
    assert probe_module["width_eers"](rows) == pytest.approx(expected)
    assert probe_module["width_eers"]([{"width": 100, "label": 0}]) is None
