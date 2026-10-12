"""Per-generator EER on the size-controlled matched view.

Every generator bucket keeps fakes paired with a registered real of identical
probed geometry, plus exactly those paired reals. Buckets are therefore balanced
and geometry-controlled, but may share reals across generators (a real can parent
up to four kept fakes). Generators with no surviving matched pair are reported by
count only; no EER is fabricated for them. EERs from ~10-60 samples per class are
coarse (one flipped sample moves EER by 1/n), so treat them as descriptive.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from deepfake_detection_archive.contracts import ScoreRecord, load_scores
from deepfake_detection_archive.evaluation.report import evaluate_records
from deepfake_detection_archive.tracking import atomic_json


def pair_reports(
    pairs: list[dict[str, Any]], scores: list[ScoreRecord]
) -> dict[str, Any]:
    """Evaluate complete generator buckets, including missing and failed scores."""
    members: dict[str, dict[str, int]] = defaultdict(dict)
    for pair in pairs:
        members[pair["generator"]][pair["fake_sample_id"]] = 1
        members[pair["generator"]][pair["real_sample_id"]] = 0
    by_id = {row.sample_id: row for row in scores}
    reports = {}
    for generator, ids in sorted(members.items()):
        bucket = []
        for sample_id, label in sorted(ids.items()):
            score = by_id.get(sample_id)
            if score is None:
                score = ScoreRecord(
                    sample_id,
                    "video",
                    None,
                    "error",
                    0.0,
                    label=label,
                    error_code="missing_score",
                    error_message="expected score absent",
                )
            elif score.label != label or score.modality != "video":
                raise ValueError(f"score/pair metadata mismatch: {sample_id}")
            bucket.append(score)
        reports[generator] = asdict(
            evaluate_records(bucket, expected_sample_ids=sorted(ids))
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = [
        json.loads(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = {}
    for scores_path in args.scores:
        report[scores_path.parent.name] = pair_reports(pairs, load_scores(scores_path))
    atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
