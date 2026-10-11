"""Compare two score files by sample ID, keeping full and paired coverage."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from deepfake_detection_archive.contracts import load_scores
from deepfake_detection_archive.data.manifest import file_sha256
from deepfake_detection_archive.evaluation.comparison import compare_records
from deepfake_detection_archive.tracking import atomic_json, code_identity, utc_now


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-scores", type=Path, required=True)
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument(
        "--threshold", type=float, help="one frozen threshold for both runs"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    comparison = compare_records(
        load_scores(args.baseline_scores),
        load_scores(args.candidate_scores),
        frozen_threshold=args.threshold,
    )
    atomic_json(
        args.output,
        {
            "recorded_at": utc_now(),
            "inputs": {
                name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for name, path in (
                    ("baseline", args.baseline_scores),
                    ("candidate", args.candidate_scores),
                )
            },
            "code": code_identity(),
            "model_identity_verified": False,
            "manifest_metadata_verified": False,
            "comparison": asdict(comparison),
        },
    )
    print(
        json.dumps(
            {
                "fullset_delta_eer": comparison.fullset_delta_eer,
                "common_delta_eer": comparison.common_delta_eer,
                "common_coverage": comparison.common_coverage,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
