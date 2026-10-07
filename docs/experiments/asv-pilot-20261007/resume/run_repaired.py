"""Resume after root review without overwriting the extraction or failed import."""

from pathlib import Path

from run_import import PYTHON, ROOT, run


if __name__ == "__main__":
    if not (ROOT / "data/asvspoof2019-la/acquisition.json").is_file():
        raise FileNotFoundError("verified extraction is required")
    if (ROOT / "data/asvspoof2019-la-pilot").exists():
        raise FileExistsError("retain existing pilot artifacts")
    run(
        "import-repaired",
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
        "pilot-repaired",
        [
            str(PYTHON),
            str(Path(__file__).parent / "source/prepare_pilot.py"),
            "--project-root",
            str(ROOT),
            "--downloaded-at",
            "2026-10-07T07:22:43.235315+00:00",
        ],
    )
