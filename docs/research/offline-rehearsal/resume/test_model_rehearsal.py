"""Guard tests only: metadata fixtures and owned non-model subprocesses."""

import copy
import importlib.util
import json
import signal
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from deepfake_detection_archive.data.manifest import SampleRecord, SourceRecord
from deepfake_detection_archive.tracking import object_hash


@pytest.fixture
def probe() -> ModuleType:
    path = Path(__file__).with_name("model_rehearsal.py")
    spec = importlib.util.spec_from_file_location("rehearsal_guard_tests", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def history() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    experiment = {"best_epoch": 2, "best_dev_eer": 0.25}
    run = {**experiment, "epochs_completed": 3}
    rows = [
        {
            "epoch": number,
            "train_samples": 16,
            "dev_samples": 8,
            "optimizer_updates": 2,
            "dev_eer": eer,
            "selected": selected,
        }
        for number, eer, selected in ((1, 0.5, True), (2, 0.25, True), (3, 0.3, False))
    ]
    return experiment, run, rows


def test_history_checks_first_best_and_updates(probe: ModuleType, history: Any) -> None:
    assert probe.verify_history(*history, {"train": 16, "dev": 8}) == 4


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("run", "epochs_completed", 4),
        ("run", "best_epoch", True),
        ("run", "best_epoch", 0),
        ("experiment", "best_epoch", 1),
        ("experiment", "best_dev_eer", 0.2),
        ("row", "epoch", 2),
        ("row", "train_samples", 15),
        ("row", "dev_samples", 7),
        ("row", "optimizer_updates", -1),
        ("row", "optimizer_updates", True),
        ("row", "dev_eer", float("nan")),
        ("row", "dev_eer", 1.5),
        ("row", "selected", False),
    ],
)
def test_history_rejects_inconsistent_evidence(
    probe: ModuleType, history: Any, section: str, key: str, value: object
) -> None:
    experiment, run, rows = history
    target = {"experiment": experiment, "run": run, "row": rows[0]}[section]
    target[key] = value
    with pytest.raises(ValueError, match=r"epoch|Epoch|History|history|Invalid|metric"):
        probe.verify_history(experiment, run, rows, {"train": 16, "dev": 8})


def test_history_rejects_later_better_epoch_and_zero_updates(
    probe: ModuleType, history: Any
) -> None:
    experiment, run, rows = history
    rows[2].update(dev_eer=0.1, selected=True)
    with pytest.raises(ValueError, match="first minimum"):
        probe.verify_history(experiment, run, rows, {"train": 16, "dev": 8})
    rows[2].update(dev_eer=0.3, selected=False)
    for row in rows:
        row["optimizer_updates"] = 0
    with pytest.raises(ValueError, match="optimizer updates"):
        probe.verify_history(experiment, run, rows, {"train": 16, "dev": 8})


@pytest.fixture
def metadata_fixture(
    probe: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, history: Any
) -> dict[str, Any]:
    # No checkpoint is written or loaded. Every failure is before model loading.
    root = tmp_path / "experiment"
    job = {
        "experiment": str(root / "experiment.json"),
        "checkpoint": str(root / "training/best.pt"),
        "checkpoint_sha256": "b" * 64,
        "training_manifest": str(tmp_path / "manifest.jsonl"),
        "dev_manifest": str(tmp_path / "dev.jsonl"),
        "sources": str(tmp_path / "sources.jsonl"),
    }
    records = [
        SampleRecord(
            sample_id=str(index),
            path=f"{index}.wav",
            modality="audio",
            label=index % 2,
            source="fixture",
            split="train" if index < 16 else "dev",
        )
        for index in range(24)
    ]
    sources = {
        "fixture": SourceRecord(
            source="fixture",
            url="local:unit",
            version="1",
            license_id="test-only",
            license_url="local:unit",
            downloaded_at="2026-10-07",
            allowed_uses=("training",),
        )
    }
    config = {"encoder": {}, "modality": "audio"}
    provenance = {
        "weights_sha256": "a" * 64,
        "encoder_sha256": object_hash({}),
        "code_sha256": "c" * 64,
        "manifest_sha256": probe.manifest_hash(records),
        "data_sha256": probe.manifest_hash(records),
        "sources_sha256": "d" * 64,
    }
    experiment = {
        **history[0],
        "status": "complete",
        "checkpoint_sha256": "b" * 64,
        "checkpoint": "training/best.pt",
        "initial_checkpoint": {"sha256": "a" * 64},
        "provenance": provenance,
        "config": config,
        "config_sha256": object_hash(config),
        "code": {"code_sha256": "c" * 64},
        "train_count": 16,
        "dev_count": 8,
        "sources": {name: asdict(value) for name, value in sources.items()},
        "holdout_accessed": False,
    }
    run = {
        **history[1],
        "status": "complete",
        "checkpoint": {"sha256": "b" * 64},
        "provenance": copy.deepcopy(provenance),
        "provenance_sha256": object_hash(provenance),
        "config": {},
        "config_sha256": object_hash({}),
        "model_config": {},
        "model_config_sha256": object_hash({}),
        "selection_split": "dev",
        "score_direction": "higher_is_fake",
    }
    payloads = {
        "experiment.json": experiment,
        "run.json": run,
        "history.json": history[2],
    }
    monkeypatch.setattr(probe, "read_json", lambda path: payloads[Path(path).name])
    digests = {"best.pt": "b" * 64, "initial.pt": "a" * 64, "sources.jsonl": "d" * 64}
    monkeypatch.setattr(probe, "sha256", lambda path: digests[Path(path).name])
    manifests = {"manifest.jsonl": records, "dev.jsonl": records[16:]}
    monkeypatch.setattr(probe, "load_manifest", lambda path: manifests[path.name])
    monkeypatch.setattr(probe, "load_sources", lambda _path: sources)
    monkeypatch.setattr(probe, "verify_experiment_paths", lambda *_args: None)
    return {"job": job, "experiment": experiment, "run": run, "manifests": manifests}


def test_training_accepts_json_roundtrip_sources_before_history(
    probe: ModuleType,
    metadata_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = metadata_fixture["experiment"]
    experiment["sources"] = json.loads(json.dumps(experiment["sources"]))
    assert isinstance(experiment["sources"]["fixture"]["allowed_uses"], list)

    def stop_at_history(*_args: Any) -> None:
        raise RuntimeError("source validation reached history")

    monkeypatch.setattr(probe, "verify_history", stop_at_history)
    with pytest.raises(RuntimeError, match="source validation reached history"):
        probe.verify_training(metadata_fixture["job"])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("license_id", "different-license"),
        ("version", "different-version"),
        ("downloaded_at", "2026-10-08"),
        ("allowed_uses", ["competition"]),
    ],
)
def test_training_rejects_changed_json_source_fields(
    probe: ModuleType,
    metadata_fixture: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    experiment = metadata_fixture["experiment"]
    experiment["sources"] = json.loads(json.dumps(experiment["sources"]))
    experiment["sources"]["fixture"][key] = value
    with pytest.raises(ValueError, match="Source registry"):
        probe.verify_training(metadata_fixture["job"])


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("experiment", "status", "failed"),
        ("run", "status", "running"),
        ("experiment", "checkpoint_sha256", "f" * 64),
        ("experiment", "checkpoint", "other.pt"),
        ("experiment", "initial_checkpoint", {"sha256": "b" * 64}),
        ("run", "provenance_sha256", "0" * 64),
        ("experiment", "code", {"code_sha256": "f" * 64}),
        ("run", "config_sha256", "0" * 64),
        ("experiment", "sources", {}),
        ("experiment", "train_count", 15),
        ("experiment", "dev_count", 7),
        ("experiment", "holdout_accessed", True),
        ("run", "selection_split", "holdout"),
        ("run", "score_direction", "higher_is_real"),
        ("job", "checkpoint", "/not/this/best.pt"),
    ],
)
def test_training_rejects_detached_evidence_before_model_load(
    probe: ModuleType,
    metadata_fixture: dict[str, Any],
    section: str,
    key: str,
    value: Any,
) -> None:
    metadata_fixture[section][key] = value
    with pytest.raises(
        ValueError,
        match=r"complete|hash|path|checkpoint|provenance|identity|registry|counts|contract|best.pt",
    ):
        probe.verify_training(metadata_fixture["job"])


@pytest.mark.parametrize("key", ["manifest_sha256", "data_sha256", "sources_sha256"])
def test_training_rejects_consistently_forged_source_hash(
    probe: ModuleType, metadata_fixture: dict[str, Any], key: str
) -> None:
    for name in ("experiment", "run"):
        metadata_fixture[name]["provenance"][key] = "e" * 64
    metadata_fixture["run"]["provenance_sha256"] = object_hash(
        metadata_fixture["run"]["provenance"]
    )
    with pytest.raises(ValueError, match="manifest/data/source"):
        probe.verify_training(metadata_fixture["job"])


def test_training_rejects_dev_membership_and_order(
    probe: ModuleType, metadata_fixture: dict[str, Any]
) -> None:
    metadata_fixture["manifests"]["dev.jsonl"].reverse()
    with pytest.raises(ValueError, match="Dev manifest"):
        probe.verify_training(metadata_fixture["job"])


@pytest.mark.parametrize("configured_model", [{}, {"channels": 24}, {"channels": 32}])
def test_actual_experiment_model_config_matches_normalized_model(
    probe: ModuleType, tmp_path: Path, configured_model: dict[str, Any]
) -> None:
    from deepfake_detection_archive.experiments import (  # noqa: PLC0415
        ExperimentConfig,
    )
    from deepfake_detection_archive.models.checkpoint import (  # noqa: PLC0415
        create_model,
        model_spec,
    )
    from deepfake_detection_archive.training import TrainConfig  # noqa: PLC0415

    # Real configuration loading and model specification, without path-guard mocks,
    # checkpoint files, parameter updates, inference, or fabricated training assets.
    config = ExperimentConfig(
        manifest="manifest.jsonl",
        data_root="media",
        sources="sources.jsonl",
        output_dir="experiment",
        modality="audio",
        architecture="logmel_cnn_v1",
        model_config=configured_model,
        training={},
    )
    config_path = tmp_path / "experiment-config.json"
    config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    architecture, actual_config = model_spec(create_model("logmel_cnn_v1", {}))
    assert actual_config["channels"] == 24
    job = {
        "experiment_config": str(config_path),
        "experiment": str(tmp_path / "experiment/experiment.json"),
        "training_manifest": str(tmp_path / "manifest.jsonl"),
        "sources": str(tmp_path / "sources.jsonl"),
        "data_root": str(tmp_path / "media"),
    }
    run = {
        "architecture": architecture,
        "model_config": actual_config,
        "config": asdict(TrainConfig()),
    }
    if configured_model.get("channels", 24) != actual_config["channels"]:
        with pytest.raises(ValueError, match="model configuration"):
            probe.verify_experiment_paths(job, {"config": asdict(config)}, run)
    else:
        probe.verify_experiment_paths(job, {"config": asdict(config)}, run)


@pytest.fixture
def snapshot() -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    prefix = Path("/recorded/env")
    package = str(prefix / "lib/python3.12/site-packages/project/__init__.py")
    prior = {"python_executable": str(prefix / "bin/python"), "package_path": package}
    distributions = {
        f"package-{i}": {
            "version": "1",
            "root": str(prefix / "lib/python3.12/site-packages"),
        }
        for i in range(65)
    }
    current = {
        "prefix": str(prefix),
        "executable": prior["python_executable"],
        "package": package,
        "torch": "2.14.1+cpu",
        "cuda": None,
        "cuda_available": False,
        "distributions": distributions,
    }
    expected = [
        {"name": name, "version": item["version"]}
        for name, item in distributions.items()
    ]
    return current, prior, prefix, {"commands": [{"stdout": json.dumps(expected)}]}


def test_environment_snapshot_accepts_fixed_cpu_closure(
    probe: ModuleType, snapshot: Any
) -> None:
    probe.verify_environment_snapshot(*snapshot)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prefix", "/another/env"),
        ("executable", "/usr/bin/python"),
        ("package", "/checkout/src/project/__init__.py"),
        ("torch", "2.14.1+cu130"),
        ("cuda", "13.0"),
        ("cuda_available", True),
    ],
)
def test_environment_snapshot_rejects_wrong_runtime(
    probe: ModuleType, snapshot: Any, field: str, value: Any
) -> None:
    snapshot[0][field] = value
    with pytest.raises(ValueError, match=r"venv|site-packages|CPU|closure"):
        probe.verify_environment_snapshot(*snapshot)


@pytest.mark.parametrize("change", ["version", "root", "extra"])
def test_environment_snapshot_rejects_drift(
    probe: ModuleType, snapshot: Any, change: str
) -> None:
    distributions = snapshot[0]["distributions"]
    if change == "extra":
        distributions["injected"] = {"version": "1", "root": "/external/site-packages"}
    else:
        distributions["package-0"][change] = (
            "2" if change == "version" else "/external/site-packages"
        )
    with pytest.raises(ValueError, match=r"venv|site-packages|CPU|closure"):
        probe.verify_environment_snapshot(*snapshot)


@pytest.mark.parametrize(
    "name", ["job.json", "scripts/model_rehearsal.py", "assets/media.wav"]
)
def test_preparation_rejects_job_script_or_asset_mutation(
    probe: ModuleType, tmp_path: Path, name: str
) -> None:
    for item in ("job.json", "scripts/model_rehearsal.py", "assets/media.wav"):
        path = tmp_path / item
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            Path(str(probe.__file__)).read_bytes()
            if item.startswith("scripts")
            else b"fixture"
        )
    preparation = {
        "job_sha256": probe.sha256(tmp_path / "job.json"),
        "script_sha256": probe.sha256(tmp_path / "scripts/model_rehearsal.py"),
        "assets": {
            "assets/media.wav": {
                "size": 7,
                "sha256": probe.sha256(tmp_path / "assets/media.wav"),
            }
        },
    }
    (tmp_path / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match=r"changed|differs"):
        probe.verify_prepared(tmp_path, preparation)


@pytest.mark.parametrize(
    "state", ["State:\tR (running)", "State:", "", "State:\tS (sleeping)"]
)
def test_stopped_state_rejects_non_stopped(probe: ModuleType, state: str) -> None:
    with pytest.raises(ValueError, match="not observed stopped"):
        probe.stopped_state(state)


def test_prefix_rejects_rewritten_or_lost_rows(probe: ModuleType) -> None:
    before = b'{"sample_id":"a"}\n'
    assert probe.complete_lines(before + b'{"partial"') == before
    probe.require_prefix(before, before + b'{"sample_id":"b"}\n')
    for after in (b"", b'{"sample_id":"b"}\n', b'{"sample_id":"changed"}\n'):
        with pytest.raises(ValueError, match="prefix changed"):
            probe.require_prefix(before, after)


def test_owned_process_stop_is_observed_and_reaped(
    probe: ModuleType, tmp_path: Path
) -> None:
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    record: dict[str, Any] = {}
    try:
        process.send_signal(signal.SIGSTOP)
        status = probe.observe_stopped(process, timeout=2)
        assert "State:" in probe.stopped_state(status)
        assert f"Pid:\t{process.pid}\n" in status
        (tmp_path / "owned-status.txt").write_text(status)
    finally:
        probe.reap_owned(process, record, grace=0.1)
    assert record["reaped"]
    assert process.poll() is not None


def test_bounded_timeout_kills_term_ignoring_owned_process(
    probe: ModuleType, tmp_path: Path
) -> None:
    log = tmp_path / "timeout.json"
    command = [
        sys.executable,
        "-I",
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(30)",
    ]
    with pytest.raises(subprocess.TimeoutExpired):
        probe.bounded_command(command, log, timeout=0.3, cwd=tmp_path)
    result = json.loads(log.read_text())
    assert result["passed"] is False
    assert result["error_type"] == "TimeoutExpired"
    assert result["cleanup"]["signals"] == ["SIGCONT", "SIGTERM", "SIGKILL"]
    assert result["cleanup"]["reaped"]
    assert result["exit_code"] == -signal.SIGKILL
    assert result["stdout"] == "ready\n"
    assert not Path(f"/proc/{result['cleanup']['owned_pid']}").exists()


def test_bounded_missing_command_leaves_failure_report(
    probe: ModuleType, tmp_path: Path
) -> None:
    log = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        probe.bounded_command([str(tmp_path / "absent")], log, timeout=1, cwd=tmp_path)
    report = json.loads(log.read_text())
    assert report["passed"] is False
    assert report["error_type"] == "FileNotFoundError"
    assert "finished_at_utc" in report


def test_bounded_success_records_exit_and_reap(
    probe: ModuleType, tmp_path: Path
) -> None:
    result = probe.bounded_command(
        [sys.executable, "-I", "-c", "print('ok')"],
        tmp_path / "ok.json",
        timeout=5,
        cwd=tmp_path,
    )
    assert result["passed"]
    assert result["cleanup"]["reaped"]
    assert result["stdout"] == "ok\n"
