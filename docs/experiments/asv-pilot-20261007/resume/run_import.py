"""Record the approved real ASV extraction/import/audit without training."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).parent
PYTHON = ROOT / ".venv/bin/python"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def run(name: str, command: list[str]) -> None:
    report = EVIDENCE / f"{name}-execution.json"
    if report.exists():
        raise FileExistsError(report)
    started = now()
    monotonic = time.perf_counter()
    env = dict(
        os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"
    )
    with (
        (EVIDENCE / f"{name}-stdout.txt").open("x") as stdout,
        (EVIDENCE / f"{name}-stderr.txt").open("x") as stderr,
    ):
        result = subprocess.run(  # noqa: S603 - fixed local commands below
            command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False
        )
    record = {
        "command": command,
        "cwd": str(ROOT),
        "started_at": started,
        "finished_at": now(),
        "seconds": time.perf_counter() - monotonic,
        "returncode": result.returncode,
        "script_sha256": digest(Path(command[1])),
        "runner_sha256": digest(Path(__file__)),
        "threads": {
            "OMP_NUM_THREADS": 2,
            "OPENBLAS_NUM_THREADS": 2,
            "MKL_NUM_THREADS": 2,
        },
        "git_head": subprocess.check_output(
            ["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "stdout_sha256": digest(EVIDENCE / f"{name}-stdout.txt"),
        "stderr_sha256": digest(EVIDENCE / f"{name}-stderr.txt"),
    }
    with report.open("x") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "stage": name,
                "returncode": result.returncode,
                "seconds": record["seconds"],
            }
        ),
        flush=True,
    )
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    for name in ("data/asvspoof2019-la", "data/asvspoof2019-la-pilot"):
        if (ROOT / name).exists():
            raise FileExistsError(ROOT / name)
    run(
        "finalize",
        [
            str(PYTHON),
            str(ROOT / "tools/finalize_asvspoof2019.py"),
            "--archive",
            "cache/downloads/asvspoof2019-LA.zip.part",
            "--output",
            "data/asvspoof2019-la",
        ],
    )
    acquisition = json.loads(
        (ROOT / "data/asvspoof2019-la/acquisition.json").read_text()
    )
    expected = {
        "archive_bytes": 7640952520,
        "archive_md5": "30c98f11d8b2bc21f2c257bfd78bb5c5",
        "archive_sha256": "208a7e4e3913f8c75ae1afd19bf32a5b29ae68435e9e30e23e5e98b6a155e4ec",
    }
    if any(acquisition[key] != value for key, value in expected.items()):
        raise ValueError("verified acquisition differs from approved archive")
    run(
        "import",
        [
            str(PYTHON),
            str(ROOT / "tools/prepare_asvspoof2019.py"),
            "--root",
            "data/asvspoof2019-la/LA",
            "--output-dir",
            "data/asvspoof2019-la-pilot",
            "--limit-per-class",
            "256",
        ],
    )
    run(
        "pilot",
        [
            str(PYTHON),
            str(ROOT / "docs/experiments/asv-pilot-20261007/source/prepare_pilot.py"),
            "--project-root",
            str(ROOT),
            "--downloaded-at",
            "2026-10-07T07:22:43.235315+00:00",
        ],
    )


if __name__ == "__main__":
    main()
