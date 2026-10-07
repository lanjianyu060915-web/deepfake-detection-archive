# ruff: noqa: S101, S603
# Deliberate verification assertions; only local fixed CLI argv and owned child PIDs.
"""Stage a real checkpoint/dev subset, then independently run isolated CPU checks.

Prepare: python -B model_rehearsal.py prepare JOB.json
Execute after Git-state freeze: python -B model_rehearsal.py execute STAGE
The caller supplies actual training evidence; no checkpoint is synthesized here.
"""

import datetime
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


if not __debug__:
    raise RuntimeError("Verification requires non-optimized Python")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_rows(path, rows):
    Path(path).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )


def prepare(job_path):  # noqa: PLR0915 -- ordered staging and verification steps
    job = read_json(job_path)
    stage = Path(job["stage"]).resolve()
    assert not stage.exists(), "Use a fresh model rehearsal directory"
    root = Path(job["data_root"]).resolve()
    checkpoint = Path(job["checkpoint"]).resolve()
    assert sha256(checkpoint) == job["checkpoint_sha256"]
    assert job["training_completed"] is True
    assert job["provenance_files"]
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
        (stage / directory).mkdir(parents=True, exist_ok=False)
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
    for key, subdirectory in (
        ("license_files", "licenses"),
        ("provenance_files", "provenance"),
    ):
        for name, source in job[key].items():
            assert Path(name).name == name
            target = stage / "assets" / subdirectory / name
            shutil.copyfile(source, target)
            assert sha256(target) == sha256(source)
    write_rows(stage / "assets/manifest.jsonl", selected)
    original = read_json(job["inference_config"])
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
    write_json(stage / "job.json", job)
    manifest = {
        str(p.relative_to(stage)): {"size": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted((stage / "assets").rglob("*"))
        if p.is_file()
    }
    report = {
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
        "model_inference_executed": False,
        "official_target_verified": False,
    }
    write_json(stage / "results/preparation.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def execute(stage):
    stage = Path(stage).resolve()
    job = read_json(stage / "job.json")
    preparation = read_json(stage / "results/preparation.json")
    assert sha256(stage / "scripts/model_rehearsal.py") == preparation["script_sha256"]
    for relative, asset in preparation["assets"].items():
        assert sha256(stage / relative) == asset["sha256"]
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
    started = datetime.datetime.now(datetime.UTC).isoformat()
    tick = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    write_json(
        stage / "results/namespace-launch.json",
        {
            "command": command,
            "started_at_utc": started,
            "finished_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - tick,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    raise SystemExit(result.returncode)


def isolated(stage, parent_namespace):  # noqa: C901, PLR0915 -- one chronological probe
    stage = Path(stage).resolve()
    job = read_json(stage / "job.json")
    python = job["python"]
    namespace = str(Path("/proc/self/ns/net").readlink())
    assert namespace != parent_namespace
    report = {
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

    def save():
        write_json(report_path, report)

    def run(arguments, *, expect_success=True):
        command = [python, "-I", "-m", "deepfake_detection_archive", *arguments]
        started = datetime.datetime.now(datetime.UTC).isoformat()
        tick = time.monotonic()
        result = subprocess.run(
            command, cwd=stage, env=env, capture_output=True, text=True, check=False
        )
        report["commands"].append(
            {
                "command": command,
                "started_at_utc": started,
                "finished_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "elapsed_seconds": time.monotonic() - tick,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        save()
        assert (result.returncode == 0) == expect_success, result.stderr
        return result

    def config(name):
        return str(stage / "assets" / f"infer-{name}.json")

    def score_path(name):
        return stage / "results" / name / "scores.jsonl"

    def evaluate(name):
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

    def checked_scores(name):
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
                stopped_deadline = time.monotonic() + 5
                while time.monotonic() < stopped_deadline:
                    state = Path(f"/proc/{process.pid}/status").read_text()
                    if any(
                        line.startswith("State:") and line.split()[1] in {"T", "t"}
                        for line in state.splitlines()
                    ):
                        break
                    time.sleep(0.005)
                else:
                    raise RuntimeError("Owned child did not stop")
                break
            time.sleep(0.005)
        else:
            raise TimeoutError("No score was persisted before interruption deadline")
        before = score_path("resume").read_bytes()
        complete_count = before.count(b"\n")
        assert 1 <= complete_count < 24, complete_count
        (stage / "results/interrupted-before-sigint.bin").write_bytes(before)
        process.send_signal(signal.SIGINT)
        process.send_signal(signal.SIGCONT)
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode != 0
        after = score_path("resume").read_bytes()
        complete_prefix = after[: after.rfind(b"\n") + 1]
        interrupted_rows = [json.loads(line) for line in complete_prefix.splitlines()]
        assert 1 <= len(interrupted_rows) < 24
        assert [r["sample_id"] for r in interrupted_rows] == expected[
            : len(interrupted_rows)
        ]
        (stage / "results/interrupted-after-exit.bin").write_bytes(after)
        report["controlled_interruption"] = {
            "command": command,
            "owned_pid": process.pid,
            "started_at_utc": started,
            "finished_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - tick,
            "exit_code": process.returncode,
            "signals": ["SIGSTOP", "SIGINT", "SIGCONT"],
            "rows_before_sigint": complete_count,
            "complete_rows_after_exit": len(interrupted_rows),
            "persisted_prefix_sha256": hashlib.sha256(complete_prefix).hexdigest(),
            "stdout": stdout,
            "stderr": stderr,
        }
        save()
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGCONT)
            process.terminate()
            process.communicate(timeout=30)
    run(["infer", "--config", config("resume"), "--resume"])
    evaluate("resume")
    resumed = checked_scores("resume")
    assert score_path("resume").read_bytes().startswith(complete_prefix)
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


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prepare":
        prepare(sys.argv[2])
    elif mode == "execute":
        execute(sys.argv[2])
    elif mode == "isolated":
        isolated(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("Expected prepare, execute, or isolated")
