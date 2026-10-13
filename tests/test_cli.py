"""CLI trust boundaries and a real-media/random-checkpoint integration probe."""

import json
import shutil
import sys
import wave
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from deepfake_detection_archive import cli
from deepfake_detection_archive.contracts import ScoreRecord
from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    file_sha256,
    load_manifest,
    write_manifest,
)
from deepfake_detection_archive.tracking import atomic_json, canonical_json


def records() -> list[SampleRecord]:
    return [
        SampleRecord("r", "r.wav", "audio", 0, "fixture", split="dev"),
        SampleRecord("f", "f.wav", "audio", 1, "fixture", split="dev"),
    ]


def write_scores(path: Path, samples: list[SampleRecord]) -> None:
    path.write_text(
        "".join(
            canonical_json(
                ScoreRecord(
                    sample.sample_id,
                    sample.modality,
                    float(sample.label or 0),
                    "ok",
                    0.1,
                    label=sample.label,
                    source=sample.source,
                    content_type=sample.content_type,
                    generator=sample.generator,
                    group_id="untrusted-" + sample.sample_id,
                ).to_dict()
            )
            + "\n"
            for sample in samples
        )
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("label", 1),
        ("modality", "video"),
        ("source", "other"),
        ("content_type", "speech"),
        ("generator", "guessed"),
    ],
)
def test_evaluate_rejects_score_metadata_disagreeing_with_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: object,
) -> None:
    samples = records()
    write_manifest(samples, tmp_path / "manifest.jsonl")
    write_scores(tmp_path / "scores.jsonl", samples)
    lines = (tmp_path / "scores.jsonl").read_text().splitlines()
    corrupted = json.loads(lines[0])
    corrupted[field] = value
    lines[0] = json.dumps(corrupted)
    (tmp_path / "scores.jsonl").write_text("\n".join(lines) + "\n")
    assert (
        cli.main(
            [
                "evaluate",
                "--scores",
                str(tmp_path / "scores.jsonl"),
                "--manifest",
                str(tmp_path / "manifest.jsonl"),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 2
    )
    assert "mismatch" in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("with_manifest", [True, False])
def test_evaluate_does_not_trust_score_claims_of_independent_groups(
    tmp_path: Path,
    with_manifest: bool,
) -> None:
    write_manifest(records(), tmp_path / "manifest.jsonl")
    write_scores(tmp_path / "scores.jsonl", records())
    command = [
        "evaluate",
        "--scores",
        str(tmp_path / "scores.jsonl"),
        "--output",
        str(tmp_path / "report.json"),
    ]
    if with_manifest:
        command.extend(["--manifest", str(tmp_path / "manifest.jsonl")])
    assert cli.main(command) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["group_count"] == 0
    assert report["missing_group_count"] == 2
    assert report["fullset_eer"] == 0


def test_split_preserves_unknown_statistical_group_ids(tmp_path: Path) -> None:
    samples = [
        SampleRecord(
            f"{label}-{index}", f"{label}-{index}.wav", "audio", label, "fixture"
        )
        for label in (0, 1)
        for index in range(4)
    ]
    write_manifest(samples, tmp_path / "input.jsonl")
    assert (
        cli.main(
            [
                "split",
                "--manifest",
                str(tmp_path / "input.jsonl"),
                "--output",
                str(tmp_path / "split.jsonl"),
            ]
        )
        == 0
    )
    assert all(
        sample.group_id == "unknown"
        for sample in load_manifest(tmp_path / "split.jsonl")
    )


def test_split_preserves_prompt_relations_without_claiming_independence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    samples = [
        SampleRecord(
            f"{label}-{index}",
            f"{label}-{index}.wav",
            "audio",
            label,
            "fixture",
            group_id=f"aime-prompt:{index}",
        )
        for label in (0, 1)
        for index in range(4)
    ]
    write_manifest(samples, tmp_path / "input.jsonl")
    assert (
        cli.main(
            [
                "split",
                "--manifest",
                str(tmp_path / "input.jsonl"),
                "--output",
                str(tmp_path / "split.jsonl"),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assigned = load_manifest(tmp_path / "split.jsonl")
    assert {r.sample_id: r.group_id for r in assigned} == {
        r.sample_id: r.group_id for r in samples
    }
    assert summary["leakage_components"] == 4
    assert summary["known_statistical_groups"] == 0
    for index in range(4):
        assert (
            len({r.split for r in assigned if r.group_id == f"aime-prompt:{index}"})
            == 1
        )


def test_evaluate_prompt_manifest_keeps_eer_but_not_independence_claims(
    tmp_path: Path,
) -> None:
    samples = [replace(r, group_id="aime-prompt:shared") for r in records()]
    write_manifest(samples, tmp_path / "manifest.jsonl")
    write_scores(tmp_path / "scores.jsonl", samples)
    assert (
        cli.main(
            [
                "evaluate",
                "--scores",
                str(tmp_path / "scores.jsonl"),
                "--manifest",
                str(tmp_path / "manifest.jsonl"),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["group_count"] == 0
    assert report["missing_group_count"] == 2
    assert report["fullset_eer"] == 0


def infer_config(tmp_path: Path) -> Path:
    write_manifest(records(), tmp_path / "manifest.jsonl")
    path = tmp_path / "infer.json"
    atomic_json(
        path,
        {
            "manifest": "manifest.jsonl",
            "data_root": ".",
            "output_dir": "run",
            "modality": "audio",
            "checkpoint": "random.pt",
            "checkpoint_sha256": "0" * 64,
            "device": "cpu",
        },
    )
    return path


@pytest.fixture
def mocked_scorer(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    called: list[str] = []

    class FakeScorer:
        def __init__(self, config: object, base: Path) -> None:
            self.base = base

        def __call__(self, record: SampleRecord) -> float:
            called.append(record.sample_id)
            if not (self.base / record.path).is_file():
                raise FileNotFoundError(record.path)
            return float(record.label or 0)

    module = ModuleType("deepfake_detection_archive.scoring")
    module.OfflineScorer = FakeScorer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        "deepfake_detection_archive.runtime.inference_runtime",
        lambda scorer: {"test": 1},
    )
    monkeypatch.setattr(cli, "code_identity", lambda: {"code_sha256": "test"})
    return called


@pytest.mark.parametrize("mutation", ["bytes", "removed", "created"])
def test_infer_resume_binds_actual_media_snapshot(
    tmp_path: Path,
    mocked_scorer: list[str],
    capsys: pytest.CaptureFixture[str],
    mutation: str,
) -> None:
    config = infer_config(tmp_path)
    (tmp_path / "r.wav").write_bytes(b"initial-real")
    if mutation != "created":
        (tmp_path / "f.wav").write_bytes(b"initial-fake")
    assert cli.main(["infer", "--config", str(config)]) == (
        2 if mutation == "created" else 0
    )
    assert mocked_scorer == ["f", "r"]
    original = (tmp_path / "run/scores.jsonl").read_bytes()
    metadata = json.loads((tmp_path / "run/metadata.json").read_text())
    assert len(metadata["identity"]["media_snapshot"]) == 2
    assert metadata["identity"]["media_snapshot_sha256"]
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["missing_group_count"] == 2
    if mutation == "bytes":
        (tmp_path / "r.wav").write_bytes(b"changed-real")
    elif mutation == "removed":
        (tmp_path / "f.wav").unlink()
    else:
        assert report["failure_count"] == 1
        assert report["fullset_eer"] is None
        (tmp_path / "f.wav").write_bytes(b"newly-available")
    assert cli.main(["infer", "--config", str(config), "--resume"]) == 2
    assert "identity changed" in capsys.readouterr().err
    assert mocked_scorer == ["f", "r"]
    assert (tmp_path / "run/scores.jsonl").read_bytes() == original


def test_failed_audit_writes_report_and_returns_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_manifest(records(), tmp_path / "manifest.jsonl")
    (tmp_path / "sources.jsonl").write_text("")
    config = tmp_path / "audit.json"
    atomic_json(
        config,
        {
            "manifest": "manifest.jsonl",
            "data_root": ".",
            "sources": "sources.jsonl",
            "output": "audit-report.json",
        },
    )
    assert cli.main(["audit", "--config", str(config)]) == 2
    assert json.loads(capsys.readouterr().out)["passed"] is False
    assert json.loads((tmp_path / "audit-report.json").read_text())["passed"] is False


def test_real_media_random_checkpoint_cli_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg required for real-media integration")
    from deepfake_detection_archive.models import (  # noqa: PLC0415 -- optional ML backend
        LogMelCNN,
        LogMelConfig,
        save_checkpoint,
    )

    monkeypatch.setattr(cli, "code_identity", lambda: {"code_sha256": "test"})
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = infer_config(tmp_path)
        samples = []
        for sample in records():
            waveform = (
                1000 * np.sin(np.arange(1600) * (0.1 + (sample.label or 0) * 0.03))
            ).astype("<i2")
            with wave.open(str(tmp_path / sample.path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(waveform.tobytes())
            samples.append(replace(sample, sha256=file_sha256(tmp_path / sample.path)))
        write_manifest(samples, tmp_path / "manifest.jsonl")
        descriptor = save_checkpoint(
            tmp_path / "random.pt",
            LogMelCNN(LogMelConfig(channels=4, hidden_dim=8, n_mels=8)),
        )
        value = json.loads(config.read_text())
        value.update(checkpoint_sha256=descriptor["sha256"], segment_samples=1600)
        atomic_json(config, value)
        assert cli.main(["infer", "--config", str(config)]) == 0
        assert cli.main(["infer", "--config", str(config), "--resume"]) == 0
        assert (
            cli.main(
                [
                    "evaluate",
                    "--scores",
                    str(tmp_path / "run/scores.jsonl"),
                    "--manifest",
                    str(tmp_path / "manifest.jsonl"),
                    "--output",
                    str(tmp_path / "evaluation.json"),
                ]
            )
            == 0
        )
        report = json.loads((tmp_path / "evaluation.json").read_text())
        assert report["failure_count"] == 0
        assert report["sample_count"] == 2
        assert report["manifest_metadata_verified"] is True
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("all_failed", [False, True])
def test_predict_exports_successes_with_failure_accounting_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocked_scorer: list[str],
    capsys: pytest.CaptureFixture[str],
    all_failed: bool,
) -> None:
    import csv  # noqa: PLC0415

    config = infer_config(tmp_path)
    for name in ("正常.wav", "损坏.wav"):
        (tmp_path / name).write_bytes(b"media")
    called: list[str] = []

    class MixedScorer:
        def __init__(self, config: object, base: Path) -> None:
            pass

        def __call__(self, record: SampleRecord) -> float:
            called.append(record.sample_id)
            if all_failed or record.path == "损坏.wav":
                raise ValueError("corrupt media")
            return 0.25

    monkeypatch.setattr(
        sys.modules["deepfake_detection_archive.scoring"], "OfflineScorer", MixedScorer
    )
    output = tmp_path / "prediction"
    command = [
        "predict",
        "--config",
        str(config),
        "--input",
        str(tmp_path),
        "--output",
        str(output),
    ]
    assert cli.main(command) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["scores_csv"] == str(output / "scores.csv")
    assert result["elapsed_seconds"] > 0
    with (output / "scores.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    saved = [
        json.loads(line) for line in (output / "scores.jsonl").read_text().splitlines()
    ]
    failed_ids = [row["sample_id"] for row in saved if row["status"] == "error"]
    assert [row["sample_id"] for row in rows] == [
        row["sample_id"] for row in saved if row["status"] == "ok"
    ]
    assert len(saved) == 2
    assert len(rows) == (0 if all_failed else 1)
    metadata = json.loads((output / "scores.csv.metadata.json").read_text())
    assert metadata["failed_sample_ids"] == failed_ids
    assert metadata["total_count"] == 2
    assert metadata["exported_count"] == len(rows)
    assert metadata["failed_count"] == len(failed_ids)
    assert metadata["coverage"] == len(rows) / 2
    assert json.loads((output / "report.json").read_text())["failure_count"] == len(
        failed_ids
    )
    original_csv = (output / "scores.csv").read_bytes()
    assert cli.main([*command, "--resume"]) == 2
    assert len(called) == 2
    assert (output / "scores.csv").read_bytes() == original_csv
    (tmp_path / "正常.wav").write_bytes(b"changed")
    assert cli.main([*command, "--resume"]) == 2
    assert (output / "scores.csv").read_bytes() == original_csv
    assert len(called) == 2


@pytest.mark.parametrize("kind", ["memory", "cuda_oom"])
@pytest.mark.parametrize("entrypoint", ["infer", "predict"])
@pytest.mark.usefixtures("mocked_scorer")
def test_inference_fatal_failure_aborts_without_completed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    entrypoint: str,
) -> None:
    torch = pytest.importorskip("torch")
    config = infer_config(tmp_path)
    for name in ("a.wav", "b.wav", "c.wav"):
        (tmp_path / name).write_bytes(b"media")
    write_manifest(
        [
            SampleRecord(name, name, "audio", None, "fixture")
            for name in ("a.wav", "b.wav", "c.wav")
        ],
        tmp_path / "manifest.jsonl",
    )
    called: list[str] = []

    class FatalScorer:
        def __init__(self, config: object, base: Path) -> None:
            pass

        def __call__(self, record: SampleRecord) -> float:
            called.append(record.path)
            if record.path == "b.wav":
                if kind == "memory":
                    raise MemoryError("exhausted")
                raise torch.OutOfMemoryError("CUDA exhausted")
            return 0.25

    monkeypatch.setattr(
        sys.modules["deepfake_detection_archive.scoring"], "OfflineScorer", FatalScorer
    )
    output = tmp_path / ("prediction" if entrypoint == "predict" else "run")
    command = [entrypoint, "--config", str(config)]
    if entrypoint == "predict":
        command.extend(["--input", str(tmp_path), "--output", str(output)])
    assert cli.main(command) == 2
    assert "exhausted" in capsys.readouterr().err
    assert called == ["a.wav", "b.wav"]
    assert len((output / "scores.jsonl").read_text().splitlines()) == 1
    assert not (output / "report.json").exists()
    assert not (output / "scores.csv").exists()


@pytest.mark.parametrize("missing", [0, 1, 2])
def test_infer_failure_exit_preserves_scores_and_report_on_resume(
    tmp_path: Path,
    mocked_scorer: list[str],
    capsys: pytest.CaptureFixture[str],
    missing: int,
) -> None:
    config = infer_config(tmp_path)
    for sample in records()[missing:]:
        (tmp_path / sample.path).write_bytes(b"media")
    command = ["infer", "--config", str(config)]
    expected_exit = 2 if missing else 0
    assert cli.main(command) == expected_exit
    result = json.loads(capsys.readouterr().out)
    assert result["samples"] == 2
    assert result["failures"] == missing
    assert result["elapsed_seconds"] > 0
    output = tmp_path / "run"
    original = (output / "scores.jsonl").read_bytes()
    scores = [json.loads(line) for line in original.splitlines()]
    assert len(scores) == 2
    assert sum(row["status"] == "error" for row in scores) == missing
    assert all(row["score"] is None for row in scores if row["status"] == "error")
    report = json.loads((output / "report.json").read_text())
    assert report["failure_count"] == missing
    if missing:
        assert report["fullset_eer"] is None
    assert cli.main([*command, "--resume"]) == expected_exit
    assert json.loads(capsys.readouterr().out)["failures"] == missing
    assert len(mocked_scorer) == 2
    assert (output / "scores.jsonl").read_bytes() == original
