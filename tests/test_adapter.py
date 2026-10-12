"""Submission conversion must preserve order, refuse failures and declare direction."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from deepfake_detection_archive.contracts import ScoreRecord
from deepfake_detection_archive.submission.adapter import AdapterConfig, export_scores


def scores() -> list[ScoreRecord]:
    return [
        ScoreRecord("b", "audio", -2.5, "ok", 0.1, label=0),
        ScoreRecord("a", "audio", 7.0, "ok", 0.1, label=1),
    ]


def test_csv_explicit_columns_and_real_high_negates_unbounded_scores(
    tmp_path: Path,
) -> None:
    path = tmp_path / "submission.csv"
    config = AdapterConfig(
        sample_id_column="file", score_column="s", score_direction="real_high"
    )
    assert export_scores(scores(), path, config) == path
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{"file": "b", "s": "2.5"}, {"file": "a", "s": "-7.0"}]
    metadata = json.loads(path.with_suffix(".csv.metadata.json").read_text())
    assert metadata["official"] is False
    assert metadata["score_transform"] == "negate"


def test_jsonl_export_preserves_values_and_order_without_labels(tmp_path: Path) -> None:
    path = export_scores(scores(), tmp_path / "s.jsonl", AdapterConfig(format="jsonl"))
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"sample_id": "b", "score": -2.5},
        {"sample_id": "a", "score": 7.0},
    ]
    first = path.read_bytes()
    export_scores(
        [replace(row, label=1 - (row.label or 0)) for row in scores()],
        path,
        AdapterConfig(format="jsonl"),
    )
    assert path.read_bytes() == first


def test_unconfirmed_official_export_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    with pytest.raises(ValueError, match="confirmed platform"):
        export_scores(scores(), path, official=True)
    assert not path.exists()
    export_scores(scores(), path, AdapterConfig(official_confirmed=True), official=True)
    assert (
        json.loads(path.with_suffix(".csv.metadata.json").read_text())["official"]
        is True
    )


def test_failures_are_not_exported_or_replaced_by_truth(tmp_path: Path) -> None:
    failed = replace(
        scores()[0],
        score=None,
        status="error",
        error_code="decode",
        error_message="broken",
    )
    with pytest.raises(ValueError, match="failed samples"):
        export_scores([failed, scores()[1]], tmp_path / "s.csv")
    assert not (tmp_path / "s.csv").exists()


def test_adapter_config_and_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="direction"):
        AdapterConfig(score_direction="guess")
    with pytest.raises(ValueError, match="differ"):
        AdapterConfig(score_column="sample_id")
    with pytest.raises(ValueError, match="unique"):
        export_scores([scores()[0]] * 2, tmp_path / "s.csv")


@pytest.mark.parametrize("all_failed", [False, True])
def test_partial_export_accounts_for_every_id(tmp_path: Path, all_failed: bool) -> None:
    failed = ScoreRecord(
        "损坏.wav",
        "audio",
        None,
        "error",
        0.1,
        error_code="decode_error",
        error_message="broken",
    )
    rows = [failed] if all_failed else [scores()[0], failed, scores()[1]]
    path = export_scores(rows, tmp_path / "s.csv", allow_partial=True)
    with path.open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert [row["sample_id"] for row in exported] == ([] if all_failed else ["b", "a"])
    metadata = json.loads(path.with_suffix(".csv.metadata.json").read_text())
    assert metadata["total_count"] == len(rows)
    assert metadata["exported_count"] == len(exported)
    assert metadata["failed_count"] == 1
    assert metadata["failed_sample_ids"] == ["损坏.wav"]
    assert metadata["coverage"] == len(exported) / len(rows)


def test_official_partial_export_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="partial"):
        export_scores(
            scores(),
            tmp_path / "s.csv",
            AdapterConfig(official_confirmed=True),
            official=True,
            allow_partial=True,
        )
    assert not (tmp_path / "s.csv").exists()
