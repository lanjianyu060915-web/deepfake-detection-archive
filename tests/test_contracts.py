"""Reject ambiguous score/config inputs before they affect evaluation or resume."""

import json
from dataclasses import replace

import pytest

from deepfake_detection_archive.config import InferenceConfig, load_inference_config
from deepfake_detection_archive.contracts import ScoreRecord, load_scores
from deepfake_detection_archive.tracking import atomic_json, parse_json


def valid_config():
    return InferenceConfig(
        manifest="manifest.jsonl",
        data_root="data",
        output_dir="scores",
        modality="audio",
        checkpoint="candidate.pt",
        checkpoint_sha256="a" * 64,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("label", True),
        ("label", 1.0),
        ("score", float("nan")),
        ("score", float("inf")),
        ("score", True),
        ("elapsed_seconds", -1),
        ("elapsed_seconds", float("inf")),
        ("sample_id", " "),
        ("modality", "image"),
        ("error_message", "unexpected"),
    ],
)
def test_score_contract_rejects_invalid_success(field, value):
    score = ScoreRecord("sample", "audio", 0.5, "ok", 0.01, label=1)
    with pytest.raises(ValueError, match=r"must|finite"):
        replace(score, **{field: value})


@pytest.mark.parametrize("error", [None, "", "  ", 42])
def test_failed_score_requires_explicit_error(error):
    with pytest.raises(ValueError, match="explicit error"):
        ScoreRecord("sample", "video", None, "error", 0, error_code=error)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("checkpoint_sha256", 7),
        ("checkpoint_sha256", "A" * 64),
        ("device", "gpu"),
        ("model_kind", "video_features"),
        ("batch_size", True),
        ("segments", 0),
        ("seed", -1),
        ("audio", []),
        ("manifest", " "),
    ],
)
def test_inference_contract_rejects_ambiguous_values(field, value):
    with pytest.raises(
        ValueError, match=r"schema|SHA|device|disagree|integer|object|path"
    ):
        replace(valid_config(), **{field: value})


def test_json_rejects_duplicate_keys_nested_keys_and_nonfinite():
    for text in ('{"score":0,"score":1}', '{"audio":{"x":1,"x":2}}', "NaN", "Infinity"):
        with pytest.raises(ValueError, match=r"duplicate|non-finite"):
            parse_json(text)


def test_config_round_trip_and_unknown_fields(tmp_path):
    path = tmp_path / "config.json"
    atomic_json(path, valid_config().to_dict())
    assert load_inference_config(path) == valid_config()
    path.write_text(
        json.dumps({**valid_config().to_dict(), "misspelled_device": "cpu"})
    )
    with pytest.raises(ValueError, match="invalid inference"):
        load_inference_config(path)


def test_score_file_rejects_duplicate_ids_keys_and_nonobjects(tmp_path):
    path = tmp_path / "scores.jsonl"
    row = json.dumps(ScoreRecord("sample", "audio", 1, "ok", 0).to_dict())
    for text, match in (
        (row + "\n" + row, "unique"),
        ('{"sample_id":"a","sample_id":"b"}', "duplicate"),
        ("[]", "object"),
        ("", "nonempty"),
    ):
        path.write_text(text)
        with pytest.raises(ValueError, match=match):
            load_scores(path)
