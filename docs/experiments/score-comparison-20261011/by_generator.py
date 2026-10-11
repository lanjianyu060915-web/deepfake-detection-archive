"""Reproduce descriptive generator buckets using archived pilot scores only."""

from pathlib import Path

from deepfake_detection_archive.contracts import load_scores
from deepfake_detection_archive.evaluation.comparison import compare_records
from deepfake_detection_archive.tracking import atomic_json


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "docs/experiments/asv-pilot-quarantine-20261008/training"
CANDIDATES = ("logmel-seed43", "aasist-l-seed42", "aasist-l-seed43-retry1")


def main() -> None:
    baseline = load_scores(SOURCE / "logmel-seed42/dev-scores.jsonl")
    generators = sorted({row.generator for row in baseline if row.label == 1})
    comparisons = []
    for name in CANDIDATES:
        candidate = load_scores(SOURCE / name / "dev-scores.jsonl")
        for generator in generators:
            left = [r for r in baseline if r.label == 0 or r.generator == generator]
            right = [r for r in candidate if r.label == 0 or r.generator == generator]
            report = compare_records(left, right)
            comparisons.append(
                {
                    "candidate": name,
                    "generator": generator,
                    "real_count": report.baseline.real_count,
                    "fake_count": report.baseline.fake_count,
                    "baseline_eer": report.baseline.fullset_eer,
                    "candidate_eer": report.candidate.fullset_eer,
                    "delta_eer": report.fullset_delta_eer,
                    "common_coverage": report.common_coverage,
                }
            )
    atomic_json(
        ROOT / "outputs/score-comparison-20261011/by-generator.json",
        {
            "scope": "historical pilot; each generator uses all 256 real controls",
            "inputs": [f"{name}.json" for name in CANDIDATES],
            "comparisons": comparisons,
        },
    )


if __name__ == "__main__":
    main()
