# ruff: noqa: S101, S603
# Deliberate verification assertions; only local fixed CLI argv and owned child PIDs.
"""Stage a real checkpoint/dev subset, then independently run isolated CPU checks.

Prepare: python -B model_rehearsal.py prepare JOB.json
Execute after Git-state freeze: python -B model_rehearsal.py execute STAGE
The caller supplies actual training evidence; no checkpoint is synthesized here.
"""

import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from deepfake_detection_archive.data.manifest import (
    file_sha256,
    load_manifest,
    load_sources,
    manifest_hash,
)
from deepfake_detection_archive.tracking import (
    atomic_json,
    code_identity,
    object_hash,
    parse_json,
)


if not __debug__:
    raise RuntimeError("Verification requires non-optimized Python")


def sha256(path: str | Path) -> str:
    return file_sha256(Path(path))


def read_json(path: str | Path) -> Any:
    return parse_json(Path(path).read_text())


def write_json(path: str | Path, value: object) -> None:
    atomic_json(Path(path), value)


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )


def prepare(job_path: str) -> None:  # noqa: PLR0915 -- ordered staging and verification steps
    job = read_json(job_path)
    stage = Path(job["stage"]).resolve()
    assert not stage.exists(), "Use a fresh model rehearsal directory"
    stage.mkdir(parents=True, exist_ok=False)
    root = Path(job["data_root"]).resolve()
    checkpoint = Path(job["checkpoint"]).resolve()
    assert sha256(checkpoint) == job["checkpoint_sha256"]
    training = verify_training(job)
    environment = verify_environment(job)
    if (
        training["provenance"]["code_sha256"]
        != environment["snapshot"]["code"]["code_sha256"]
    ):
        raise ValueError("Training code differs from installed offline model code")
    assert job["license_files"]
    rows = read_rows(job["dev_manifest"])
    assert rows
    assert all(r["split"] == "dev" for r in rows), "Dev-only manifest required"
    assert all(r["modality"] == "audio" for r in rows)
    assert len({r["sample_id"] for r in rows}) == len(rows)
    selected = []
    for label in (0, 1):
        candidates = sorted(
            (r for r in rows if type(r["label"]) is int and r["label"] == label),
            key=lambda r: r["sample_id"],
        )
        assert len(candidates) >= 12
        selected.extend(candidates[:12])
    selected.sort(key=lambda r: r["sample_id"])
    for directory in (
        "assets/media",
        "assets/licenses",
        "assets/provenance",
        "results",
        "scripts",
    ):
        (stage / directory).mkdir(parents=True, exist_ok=directory == "results")
    for row in selected:
        relative = Path(row["path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        source = (root / relative).resolve()
        assert source.is_relative_to(root)
        assert source.is_file()
        assert sha256(source) == row["sha256"], row["sample_id"]
        target = stage / "assets/media" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha256(target) == row["sha256"]
        else:
            shutil.copyfile(source, target)
        assert sha256(target) == row["sha256"]
    shutil.copyfile(checkpoint, stage / "assets/best.pt")
    assert sha256(stage / "assets/best.pt") == job["checkpoint_sha256"]
    shutil.copyfile(job["sources"], stage / "assets/sources.jsonl")
    provenance_files = training["evidence_paths"]
    for key, subdirectory in (
        ("license_files", "licenses"),
        ("provenance_files", "provenance"),
    ):
        for name, source in (
            provenance_files if key == "provenance_files" else job[key]
        ).items():
            assert Path(name).name == name
            target = stage / "assets" / subdirectory / name
            shutil.copyfile(source, target)
            assert sha256(target) == sha256(source)
    write_rows(stage / "assets/manifest.jsonl", selected)
    original = read_json(job["inference_config"])
    verify_preprocessing(original, training["experiment_config"])
    assert original["modality"] == "audio"
    assert original["model_kind"] == "audio_waveform"
    for name in ("reference", "offline", "resume"):
        config = dict(original)
        config.update(
            {
                "manifest": "manifest.jsonl",
                "data_root": "media",
                "checkpoint": "best.pt",
                "checkpoint_sha256": job["checkpoint_sha256"],
                "device": "cpu",
                "output_dir": f"../results/{name}",
            }
        )
        write_json(stage / "assets" / f"infer-{name}.json", config)
    copied_script = stage / "scripts/model_rehearsal.py"
    shutil.copyfile(Path(__file__), copied_script)
    shutil.copyfile(job_path, stage / "job.json")
    manifest = {
        str(p.relative_to(stage)): {"size": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted((stage / "assets").rglob("*"))
        if p.is_file()
    }
    report: dict[str, Any] = {
        "prepared_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "selection": "First 12 sample_id-sorted dev records per label, then sort union by sample_id; no model scores used",
        "selected_ids": [r["sample_id"] for r in selected],
        "selected_label_counts": {"real": 12, "fake": 12},
        "source_dev_manifest_sha256": sha256(job["dev_manifest"]),
        "source_sources_sha256": sha256(job["sources"]),
        "source_checkpoint_sha256": job["checkpoint_sha256"],
        "source_inference_config_sha256": sha256(job["inference_config"]),
        "script_sha256": sha256(copied_script),
        "assets": manifest,
        "job_sha256": sha256(stage / "job.json"),
        "training": training,
        "environment": environment,
        "model_inference_executed": False,
        "official_target_verified": False,
    }
    write_json(stage / "results/preparation.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def execute(stage_value: str) -> None:
    stage = Path(stage_value).resolve()
    job = read_json(stage / "job.json")
    preparation = read_json(stage / "results/preparation.json")
    verify_prepared(stage, preparation)
    parent_namespace = str(Path("/proc/self/ns/net").readlink())
    command = [
        "unshare",
        "--user",
        "--map-root-user",
        "--net",
        job["python"],
        "-I",
        str(stage / "scripts/model_rehearsal.py"),
        "isolated",
        str(stage),
        parent_namespace,
    ]
    result = bounded_command(
        command, stage / "results/namespace-launch.json", timeout=3600, cwd=stage
    )
    print(result["stdout"], end="")
    print(result["stderr"], end="", file=sys.stderr)
    if result["exit_code"] != 0:
        raise RuntimeError("Namespace rehearsal failed; see namespace-launch.json")


def isolated(stage_value: str, parent_namespace: str) -> None:  # noqa: PLR0915 -- one chronological probe
    stage = Path(stage_value).resolve()
    job = read_json(stage / "job.json")
    preparation = read_json(stage / "results/preparation.json")
    verify_prepared(stage, preparation)
    python = job["python"]
    namespace = str(Path("/proc/self/ns/net").readlink())
    assert namespace != parent_namespace
    report: dict[str, Any] = {
        "scope": "Local CPU real-detector dev-sample rehearsal; not official target or representative-500 acceptance",
        "parent_namespace": parent_namespace,
        "network_namespace": namespace,
        "route_table": Path("/proc/net/route").read_text(),
        "commands": [],
        "script_sha256": sha256(Path(__file__)),
        "official_target_verified": False,
    }
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=2).close()
    except OSError as exc:
        report["network_probe"] = {
            "external_tcp_failed": True,
            "exception": type(exc).__name__,
            "errno": exc.errno,
            "message": str(exc),
        }
    else:
        raise RuntimeError("External TCP unexpectedly reachable")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    env.update(
        {
            "UV_OFFLINE": "1",
            "UV_PYTHON_DOWNLOADS": "never",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "HF_HOME": str(stage / "empty-hf"),
            "XDG_CACHE_HOME": str(stage / "empty-xdg"),
        }
    )
    for directory in ("empty-hf", "empty-xdg"):
        (stage / directory).mkdir(exist_ok=False)
    report["explicit_environment"] = {
        k: env[k]
        for k in (
            "UV_OFFLINE",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "HF_HOME",
            "XDG_CACHE_HOME",
        )
    }
    report_path = stage / "results/model-probe.json"

    def save() -> None:
        write_json(report_path, report)

    def run(arguments: list[str], *, expect_success: bool = True) -> dict[str, Any]:
        command = [python, "-I", "-m", "deepfake_detection_archive", *arguments]
        log = stage / "results" / f"command-{len(report['commands']):02d}.json"
        try:
            result = bounded_command(command, log, timeout=300, cwd=stage, env=env)
        finally:
            if log.exists():
                report["commands"].append(read_json(log))
                save()
        assert (result["exit_code"] == 0) == expect_success, result["stderr"]
        return result

    def config(name: str) -> str:
        return str(stage / "assets" / f"infer-{name}.json")

    def score_path(name: str) -> Path:
        return stage / "results" / name / "scores.jsonl"

    def evaluate(name: str) -> None:
        run(
            [
                "evaluate",
                "--scores",
                str(score_path(name)),
                "--manifest",
                str(stage / "assets/manifest.jsonl"),
                "--output",
                str(stage / "results" / name / "evaluation.json"),
            ]
        )

    expected = [r["sample_id"] for r in read_rows(stage / "assets/manifest.jsonl")]

    def checked_scores(name: str) -> list[dict[str, Any]]:
        rows = read_rows(score_path(name))
        assert [r["sample_id"] for r in rows] == expected
        assert len({r["sample_id"] for r in rows}) == 24
        assert all(
            r["score"] is not None
            and math.isfinite(r["score"])
            and not r.get("error_code")
            for r in rows
        )
        return rows

    # Two complete fresh runs are retained before interruption is attempted.
    run(["infer", "--config", config("reference")])
    evaluate("reference")
    reference = checked_scores("reference")
    run(["infer", "--config", config("offline")])
    evaluate("offline")
    offline = checked_scores("offline")
    assert [r["score"] for r in reference] == [r["score"] for r in offline]
    report["reference_offline_max_score_difference"] = 0.0
    save()

    # The exact child belongs to this Popen. Freeze it before capturing a durable prefix.
    command = [
        python,
        "-I",
        "-m",
        "deepfake_detection_archive",
        "infer",
        "--config",
        config("resume"),
    ]
    started = datetime.datetime.now(datetime.UTC).isoformat()
    tick = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=stage,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Inference ended before a controlled interruption")
            path = score_path("resume")
            if path.exists() and path.read_bytes().count(b"\n") >= 1:
                process.send_signal(signal.SIGSTOP)
                state = observe_stopped(process, timeout=5)
                break
            time.sleep(0.005)
        else:
            raise TimeoutError("No score was persisted before interruption deadline")
        stopped_at = datetime.datetime.now(datetime.UTC).isoformat()
        (stage / "results/owned-stopped-status.txt").write_text(state)
        write_json(
            stage / "results/owned-stopped-state.json",
            {
                "pid": process.pid,
                "observed_at_utc": stopped_at,
                "proc_status": state,
                "state_line": stopped_state(state),
            },
        )
        before = score_path("resume").read_bytes()
        stopped_prefix = complete_lines(before)
        complete_count = before.count(b"\n")
        (stage / "results/interrupted-before-sigint.bin").write_bytes(before)
        assert 1 <= complete_count < 24, complete_count
        process.send_signal(signal.SIGINT)
        process.send_signal(signal.SIGCONT)
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode != 0
        after = score_path("resume").read_bytes()
        (stage / "results/interrupted-after-exit.bin").write_bytes(after)
        complete_prefix = complete_lines(after)
        require_prefix(stopped_prefix, complete_prefix)
        interrupted_rows = [json.loads(line) for line in complete_prefix.splitlines()]
        assert 1 <= len(interrupted_rows) < 24
        assert [r["sample_id"] for r in interrupted_rows] == expected[
            : len(interrupted_rows)
        ]
        report["controlled_interruption"] = {
            "command": command,
            "owned_pid": process.pid,
            "started_at_utc": started,
            "finished_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - tick,
            "exit_code": process.returncode,
            "signals": ["SIGSTOP", "SIGINT", "SIGCONT"],
            "rows_before_sigint": complete_count,
            "stopped_at_utc": stopped_at,
            "stopped_proc_status": state,
            "stopped_complete_prefix_sha256": hashlib.sha256(
                stopped_prefix
            ).hexdigest(),
            "exit_preserves_stopped_prefix": True,
            "complete_rows_after_exit": len(interrupted_rows),
            "persisted_prefix_sha256": hashlib.sha256(complete_prefix).hexdigest(),
            "stdout": stdout,
            "stderr": stderr,
        }
        save()
    finally:
        cleanup: dict[str, Any] = {}
        try:
            reap_owned(process, cleanup)
        finally:
            write_json(stage / "results/interruption-cleanup.json", cleanup)
    run(["infer", "--config", config("resume"), "--resume"])
    evaluate("resume")
    resumed = checked_scores("resume")
    require_prefix(complete_prefix, score_path("resume").read_bytes())
    require_prefix(stopped_prefix, score_path("resume").read_bytes())
    assert [r["score"] for r in reference] == [r["score"] for r in resumed]
    final_digest = sha256(score_path("resume"))
    run(["infer", "--config", config("resume"), "--resume"])
    assert sha256(score_path("resume")) == final_digest
    report.update(
        {
            "passed": True,
            "model_inference_executed": True,
            "sample_count": 24,
            "successful_rows_each_run": 24,
            "resume_prefix_preserved": True,
            "resume_duplicate_count": 0,
            "resume_missing_count": 0,
            "resume_max_score_difference": 0.0,
            "completed_resume_scores_unchanged": True,
            "finished_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        }
    )
    report["result_files"] = {
        str(p.relative_to(stage)): {"size": p.stat().st_size, "sha256": sha256(p)}
        for folder in ("reference", "offline", "resume")
        for p in sorted((stage / "results" / folder).rglob("*"))
        if p.is_file()
    }
    save()
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "passed",
                    "sample_count",
                    "resume_max_score_difference",
                    "official_target_verified",
                )
            }
        ),
        flush=True,
    )


# These immutable historical reports identify the already-installed environment.
# A rebuilt environment needs a new reviewed binding, not edited old reports.
INSTALLATION_SHA256 = "b11e625159ae88f0e11192c2b4d87a3ff7ad90aa2133ba5ef2016923fd51eecc"
WHEELHOUSE_SHA256 = "14a2d2cf31cef687258e7527f3050b630d83e27220449c646465eb2632a3f69a"


def complete_lines(value: bytes) -> bytes:
    return value[: value.rfind(b"\n") + 1]


def require_prefix(before: bytes, after: bytes) -> None:
    if not before or not before.endswith(b"\n") or not after.startswith(before):
        raise ValueError("Complete persisted score prefix changed")


def stopped_state(status: str) -> str:
    for line in status.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "State:" and parts[1] in {"T", "t"}:
            return line
    raise ValueError("Owned process was not observed stopped")


def observe_stopped(process: subprocess.Popen[str], *, timeout: float) -> str:
    """Observe this owned child's actual kernel state before reading score bytes."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Stopped-state timeout must be finite and positive")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Owned child exited before stopped-state observation")
        status = Path(f"/proc/{process.pid}/status").read_text()
        try:
            stopped_state(status)
        except ValueError:
            time.sleep(0.005)
        else:
            return status
    raise TimeoutError("Owned child did not enter a stopped state")


def reap_owned(
    process: subprocess.Popen[str], record: dict[str, Any], *, grace: float = 5.0
) -> tuple[str, str]:
    """Signal only this still-owned Popen; terminate, kill if needed, always reap."""
    record.update(owned_pid=process.pid, signals=[], reaped=False)
    if process.poll() is None:
        # A stopped child must continue before TERM can be delivered.
        process.send_signal(signal.SIGCONT)
        record["signals"].append("SIGCONT")
        process.terminate()
        record["signals"].append("SIGTERM")
    try:
        output = process.communicate(timeout=grace)
    except subprocess.TimeoutExpired:
        process.kill()
        record["signals"].append("SIGKILL")
        output = process.communicate(timeout=grace)
    record.update(reaped=True, exit_code=process.returncode)
    return output


def bounded_command(
    command: list[str],
    log: Path,
    *,
    timeout: float,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Command timeout must be finite and positive")
    record: dict[str, Any] = {
        "command": command,
        "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "timeout_seconds": timeout,
        "passed": False,
    }
    tick = time.monotonic()
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(timeout=timeout)
        record.update(
            stdout=stdout,
            stderr=stderr,
            exit_code=process.returncode,
            passed=process.returncode == 0,
        )
    except BaseException as exc:
        record.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        try:
            if process is not None:
                cleanup: dict[str, Any] = {}
                record["cleanup"] = cleanup
                stdout, stderr = reap_owned(process, cleanup)
                record.setdefault("stdout", stdout)
                record.setdefault("stderr", stderr)
                record["exit_code"] = process.returncode
        finally:
            record.update(
                finished_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                elapsed_seconds=time.monotonic() - tick,
            )
            write_json(log, record)
    return record


def verify_preprocessing(inference: dict[str, Any], config: dict[str, Any]) -> None:
    for key, default in (("audio", {}), ("segments", 1), ("segment_samples", 64600)):
        if inference.get(key, default) != config.get(key, default):
            raise ValueError(f"Inference preprocessing differs from training: {key}")
    if config["modality"] != "audio" or config.get("encoder"):
        raise ValueError("This rehearsal only supports waveform audio detectors")


def verify_experiment_paths(
    job: dict[str, Any], experiment: dict[str, Any], run: dict[str, Any]
) -> None:
    from deepfake_detection_archive.experiments import (  # noqa: PLC0415
        load_experiment_config,
    )
    from deepfake_detection_archive.models.checkpoint import (  # noqa: PLC0415
        create_model,
        model_spec,
    )
    from deepfake_detection_archive.training import TrainConfig  # noqa: PLC0415

    path = Path(job["experiment_config"]).resolve()
    config = load_experiment_config(path)
    if asdict(config) != experiment["config"]:
        raise ValueError(
            "Actual experiment configuration differs from completion record"
        )
    if (
        asdict(TrainConfig(**config.training)) != run["config"]
        or config.architecture != run["architecture"]
    ):
        raise ValueError("Trainer settings do not match experiment")
    if model_spec(create_model(config.architecture, config.model_config)) != (
        run["architecture"],
        run["model_config"],
    ):
        raise ValueError("Trainer model configuration does not match experiment")
    for key, configured in (
        ("training_manifest", config.manifest),
        ("sources", config.sources),
        ("data_root", config.data_root),
    ):
        if Path(job[key]).resolve() != (path.parent / configured).resolve():
            raise ValueError(f"Actual input path differs from experiment: {key}")
    if (
        Path(job["experiment"]).resolve().parent
        != (path.parent / config.output_dir).resolve()
    ):
        raise ValueError("Completion report is not in configured experiment directory")


def verify_training(job: dict[str, Any]) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915 -- ordered independent evidence checks
    experiment_path = Path(job["experiment"]).resolve()
    directory = experiment_path.parent
    run_path = directory / "training/run.json"
    history_path = directory / "training/history.json"
    initial = directory / "initial.pt"
    checkpoint = directory / "training/best.pt"
    experiment, run, history = (
        read_json(experiment_path),
        read_json(run_path),
        read_json(history_path),
    )
    if experiment["status"] != "complete" or run["status"] != "complete":
        raise ValueError("Both experiment and trainer must be complete")
    if (
        experiment["holdout_accessed"] is not False
        or run["selection_split"] != "dev"
        or run["score_direction"] != "higher_is_fake"
    ):
        raise ValueError("Training selection/holdout/score-direction contract differs")
    if Path(job["checkpoint"]).resolve() != checkpoint:
        raise ValueError("Checkpoint is not this experiment's best.pt")
    digest = sha256(checkpoint)
    if (
        not digest
        == job["checkpoint_sha256"]
        == experiment["checkpoint_sha256"]
        == run["checkpoint"]["sha256"]
    ):
        raise ValueError("Best checkpoint hashes disagree")
    if experiment["checkpoint"] != "training/best.pt":
        raise ValueError("Experiment checkpoint path differs")
    initial_digest = sha256(initial)
    if (
        initial_digest != experiment["initial_checkpoint"]["sha256"]
        or initial_digest == digest
    ):
        raise ValueError("Best checkpoint must differ from verified initial checkpoint")
    provenance = run["provenance"]
    if (
        provenance != experiment["provenance"]
        or object_hash(provenance) != run["provenance_sha256"]
    ):
        raise ValueError("Training provenance disagrees")
    if provenance["weights_sha256"] != initial_digest:
        raise ValueError("Initial weights do not match provenance")
    config = experiment["config"]
    verify_experiment_paths(job, experiment, run)
    for value, expected in (
        (config, experiment["config_sha256"]),
        (run["config"], run["config_sha256"]),
        (run["model_config"], run["model_config_sha256"]),
    ):
        if object_hash(value) != expected:
            raise ValueError("Recorded configuration hash mismatch")
    if provenance["encoder_sha256"] != object_hash(config["encoder"]):
        raise ValueError("Encoder provenance mismatch")
    if experiment["code"]["code_sha256"] != provenance["code_sha256"]:
        raise ValueError("Training source identity mismatch")
    records = load_manifest(Path(job["training_manifest"]))
    selected = [
        r
        for r in records
        if r.split in {"train", "dev"} and r.modality == config["modality"]
    ]
    dev = [r for r in selected if r.split == "dev"]
    supplied_dev = load_manifest(Path(job["dev_manifest"]))
    if [r.to_dict() for r in dev] != [r.to_dict() for r in supplied_dev]:
        raise ValueError(
            "Dev manifest must exactly match training dev members and order"
        )
    sources_path = Path(job["sources"])
    sources = load_sources(sources_path)
    if experiment["sources"] != {
        key: asdict(value)
        for key, value in sources.items()
        if key in {r.source for r in selected}
    }:
        raise ValueError("Source registry does not match actual experiment")
    checks = {
        "manifest_sha256": manifest_hash(records),
        "data_sha256": manifest_hash(selected),
        "sources_sha256": sha256(sources_path),
    }
    if any(provenance[key] != value for key, value in checks.items()):
        raise ValueError("Training manifest/data/source hashes disagree")
    counts = {
        split: sum(r.split == split for r in selected) for split in ("train", "dev")
    }
    if any(
        counts[split] <= 0 or experiment[f"{split}_count"] != counts[split]
        for split in counts
    ):
        raise ValueError("Actual train/dev counts disagree")
    updates = verify_history(experiment, run, history, counts)
    # Hash difference alone can result from serialization metadata. Verify parameters.
    import torch  # noqa: PLC0415 -- load tensor backend only for explicit validation

    from deepfake_detection_archive.models.checkpoint import (  # noqa: PLC0415
        load_checkpoint,
    )

    best_model, best_meta = load_checkpoint(checkpoint, expected_sha256=digest)
    initial_model, initial_meta = load_checkpoint(
        initial, expected_sha256=initial_digest
    )
    for field in ("architecture", "config", "score_direction"):
        if best_meta[field] != initial_meta[field]:
            raise ValueError("Initial/best checkpoint model contracts differ")
    if (
        best_meta["architecture"] != run["architecture"]
        or best_meta["config"] != run["model_config"]
    ):
        raise ValueError("Checkpoint and trainer model contracts differ")
    old_parameters = dict(initial_model.named_parameters())
    changed = [
        name
        for name, parameter in best_model.named_parameters()
        if not torch.equal(parameter, old_parameters[name])
    ]
    if not changed:
        raise ValueError("No learned parameter differs from initial checkpoint")
    paths = {
        "experiment.json": str(experiment_path),
        "run.json": str(run_path),
        "history.json": str(history_path),
        "training-manifest.jsonl": job["training_manifest"],
        "dev-manifest.jsonl": job["dev_manifest"],
        "sources.jsonl": str(sources_path),
        "initial.pt": str(initial),
        "experiment-config.json": job["experiment_config"],
    }
    return {
        "best_epoch": run["best_epoch"],
        "updates_through_best_epoch": updates,
        "changed_parameter_names": changed,
        "provenance": provenance,
        "experiment_config": config,
        "evidence_paths": paths,
        "evidence_sha256": {key: sha256(path) for key, path in paths.items()},
    }


def verify_history(
    experiment: dict[str, Any],
    run: dict[str, Any],
    history: list[dict[str, Any]],
    counts: dict[str, int],
) -> int:
    best = run["best_epoch"]
    epochs = run["epochs_completed"]
    if type(best) is not int or type(epochs) is not int or not 1 <= best <= epochs:
        raise ValueError("Invalid best/complete epoch")
    if best != experiment["best_epoch"] or epochs != len(history):
        raise ValueError("Epoch records disagree")
    updates = 0
    lowest_eer = math.inf
    selected_epoch = 0
    for number, epoch in enumerate(history, 1):
        if epoch["epoch"] != number or any(
            epoch[f"{s}_samples"] != counts[s] for s in counts
        ):
            raise ValueError("History epoch/sample counts disagree")
        eer = epoch["dev_eer"]
        if type(eer) not in (int, float) or not math.isfinite(eer) or not 0 <= eer <= 1:
            raise ValueError("Invalid history EER")
        improved = eer < lowest_eer
        if epoch["selected"] is not improved:
            raise ValueError(
                "History selection does not follow strict dev EER improvement"
            )
        if improved:
            lowest_eer, selected_epoch = eer, number
        count = epoch["optimizer_updates"]
        if type(count) is not int or count < 0:
            raise ValueError("Invalid optimizer updates")
        if number <= best:
            updates += count
    if best != selected_epoch:
        raise ValueError("Best epoch is not the first minimum dev EER epoch")
    if updates <= 0 or history[best - 1]["selected"] is not True:
        raise ValueError("Best checkpoint has no recorded optimizer updates")
    if (
        not history[best - 1]["dev_eer"]
        == run["best_dev_eer"]
        == experiment["best_dev_eer"]
    ):
        raise ValueError("Best epoch metric disagrees")
    return updates


def environment_snapshot() -> dict[str, Any]:
    import torch  # noqa: PLC0415 -- load tensor backend only for explicit validation

    import deepfake_detection_archive  # noqa: PLC0415

    distributions = {}
    for item in importlib.metadata.distributions():
        name = item.metadata["Name"].lower().replace("_", "-")
        if name in distributions:
            raise ValueError("Duplicate installed distribution")
        distributions[name] = {
            "version": item.version,
            "root": str(Path(str(item.locate_file(""))).resolve()),
        }
    return {
        "prefix": sys.prefix,
        "executable": sys.executable,
        "package": str(Path(deepfake_detection_archive.__file__).resolve()),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "distributions": distributions,
        "code": code_identity(),
    }


def verify_environment(job: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(job["installation_summary"]).resolve()
    if sha256(summary_path) != INSTALLATION_SHA256:
        raise ValueError("Historical installation summary is not the fixed report")
    summary = read_json(summary_path)
    report_dir = summary_path.parent
    for name, expected in summary["record_sha256"].items():
        if sha256(report_dir / name) != expected:
            raise ValueError(f"Historical installation evidence changed: {name}")
    inventory_path = report_dir / "cpu-wheelhouse.json"
    if sha256(inventory_path) != WHEELHOUSE_SHA256:
        raise ValueError("Historical wheel inventory changed")
    inventory = read_json(inventory_path)
    probe = read_json(report_dir / "cpu-installation-probe.json")
    prior = [
        json.loads(c["stdout"])
        for c in probe["commands"]
        if "python_executable" in c["stdout"]
    ]
    if len(prior) != 1:
        raise ValueError("Historical installed environment is ambiguous")
    python = Path(
        job["python"]
    ).absolute()  # Never resolve this symlink to system Python.
    if str(python) != prior[0]["python_executable"]:
        raise ValueError("Python path differs from the recorded CPU venv")
    prefix = python.parent.parent
    wheelhouse = prefix.parent / "wheelhouse"
    wheels = [*inventory["wheels"], summary["project_wheel"]]
    if len(wheels) != 65 or {p.name for p in wheelhouse.glob("*.whl")} != {
        w["filename"] for w in wheels
    }:
        raise ValueError("Wheelhouse is not the recorded 65 wheels")
    for wheel in wheels:
        path = wheelhouse / wheel["filename"]
        if path.stat().st_size != wheel["size"] or sha256(path) != wheel["sha256"]:
            raise ValueError(f"Wheel changed: {path.name}")
    stage = Path(job["stage"]).resolve()
    stage.mkdir(exist_ok=True)
    result = bounded_command(
        [str(python), "-I", str(Path(__file__).resolve()), "environment"],
        stage / "results/environment-check.json",
        timeout=60,
        cwd=stage,
    )
    if result["exit_code"] != 0:
        raise ValueError("CPU environment probe failed")
    current = json.loads(result["stdout"])
    verify_environment_snapshot(current, prior[0], prefix, probe)
    source_root = Path(job["source_root"]).resolve()
    project_wheel = wheelhouse / summary["project_wheel"]["filename"]
    with zipfile.ZipFile(project_wheel) as archive:
        members = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("deepfake_detection_archive/") and not name.endswith("/")
        }
    package = Path(current["package"]).parent
    installed = {
        f"deepfake_detection_archive/{p.relative_to(package).as_posix()}": sha256(p)
        for p in package.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    source = {
        f"deepfake_detection_archive/{p.relative_to(source_root).as_posix()}": sha256(p)
        for p in source_root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if members != installed or members != source:
        raise ValueError(
            "Source/installed package does not exactly match project wheel members"
        )
    return {
        "snapshot": current,
        "project_members": members,
        "installation_summary_sha256": sha256(summary_path),
        "wheelhouse_sha256": sha256(inventory_path),
        "wheel_count": len(wheels),
    }


def verify_environment_snapshot(
    current: dict[str, Any], prior: dict[str, Any], prefix: Path, probe: dict[str, Any]
) -> None:
    if (
        current["prefix"] != str(prefix)
        or current["executable"] != prior["python_executable"]
    ):
        raise ValueError("Actual sys.prefix/executable is not the installed CPU venv")
    if current["package"] != prior["package_path"] or not Path(
        current["package"]
    ).is_relative_to(prefix):
        raise ValueError("Package is not from recorded site-packages")
    if "site-packages" not in Path(current["package"]).parts:
        raise ValueError("Package missing site-packages ownership")
    if (
        current["torch"] != "2.14.1+cpu"
        or current["cuda"] is not None
        or current["cuda_available"]
    ):
        raise ValueError("Actual environment is not the pinned CPU build")
    expected = json.loads(probe["commands"][-1]["stdout"])
    expected_versions = {
        item["name"].lower().replace("_", "-"): item["version"] for item in expected
    }
    actual_versions = {
        name: value["version"] for name, value in current["distributions"].items()
    }
    if len(actual_versions) != 65 or expected_versions != actual_versions:
        raise ValueError("Installed distribution closure changed")
    if any(
        not Path(value["root"]).is_relative_to(prefix)
        or "site-packages" not in Path(value["root"]).parts
        for value in current["distributions"].values()
    ):
        raise ValueError("Installed distribution escapes the recorded venv")


def verify_prepared(stage: Path, preparation: dict[str, Any]) -> None:
    if sha256(stage / "job.json") != preparation["job_sha256"]:
        raise ValueError("Prepared job changed")
    if (
        sha256(stage / "scripts/model_rehearsal.py") != preparation["script_sha256"]
        or sha256(Path(__file__)) != preparation["script_sha256"]
    ):
        raise ValueError("Executing script differs from prepared script")
    for name, asset in preparation["assets"].items():
        path = (stage / name).resolve()
        if (
            not path.is_relative_to(stage)
            or path.stat().st_size != asset["size"]
            or sha256(path) != asset["sha256"]
        ):
            raise ValueError(f"Prepared asset changed: {name}")
    job = read_json(stage / "job.json")
    training = verify_training(job)
    environment = verify_environment(job)
    if training != preparation["training"] or environment != preparation["environment"]:
        raise ValueError("Prepared training/environment binding changed")
    if (
        training["provenance"]["code_sha256"]
        != environment["snapshot"]["code"]["code_sha256"]
    ):
        raise ValueError("Training code differs from installed offline model code")


def interrupted(_signum: int, _frame: object) -> None:
    raise InterruptedError("Rehearsal received termination signal")


def main() -> None:
    mode = sys.argv[1]
    if mode == "environment":
        print(json.dumps(environment_snapshot()), flush=True)
        return
    stage = (
        Path(read_json(sys.argv[2])["stage"]).resolve()
        if mode == "prepare"
        else Path(sys.argv[2]).resolve()
    )
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if mode == "prepare":
            prepare(sys.argv[2])
        elif mode == "execute":
            execute(sys.argv[2])
        elif mode == "isolated":
            isolated(sys.argv[2], sys.argv[3])
        else:
            raise ValueError("Expected prepare, execute, isolated or environment")  # noqa: TRY301 -- report failures at CLI boundary
    except BaseException as exc:
        write_json(
            stage / "results" / f"{mode}-failure.json",
            {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "observed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "model_inference_passed": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
