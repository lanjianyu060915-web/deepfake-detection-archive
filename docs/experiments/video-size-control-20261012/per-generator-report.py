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
import collections
import json
from pathlib import Path

from deepfake_detection_archive.evaluation.metrics import evaluate_binary

HERE = Path(__file__).resolve().parent


def load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = load_rows(args.pairs)
    members: dict[str, set[str]] = collections.defaultdict(set)
    for pair in pairs:
        generator = str(pair["generator"])
        members[generator].add(str(pair["fake_sample_id"]))
        members[generator].add(str(pair["real_sample_id"]))

    report: dict[str, object] = {"generators_with_pairs": {}, "scores": {}}
    for scores_path in args.scores:
        rows = {
            str(row["sample_id"]): row
            for row in load_rows(scores_path)
            if row.get("status") == "ok"
        }
        candidate: dict[str, object] = {}
        for generator, ids in sorted(members.items()):
            bucket = [rows[sample_id] for sample_id in sorted(ids) if sample_id in rows]
            labels = [int(row["label"]) for row in bucket]
            values = [float(row["score"]) for row in bucket]
            if len(set(labels)) != 2:
                candidate[generator] = {
                    "fake": labels.count(1),
                    "real": labels.count(0),
                    "eer": None,
                    "reason": "single_class",
                }
                continue
            metrics = evaluate_binary(labels, values)
            candidate[generator] = {
                "fake": labels.count(1),
                "real": labels.count(0),
                "eer": metrics.eer,
            }
        report["scores"][scores_path.parent.name] = candidate

    report["generators_with_pairs"] = {
        generator: {"fake": len(ids) // 2, "real": len(ids) // 2}
        for generator, ids in sorted(members.items())
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, candidate in report["scores"].items():
        print(name)
        for generator, bucket in candidate.items():
            print(f"  {generator:26s} {bucket}")


if __name__ == "__main__":
    main()
