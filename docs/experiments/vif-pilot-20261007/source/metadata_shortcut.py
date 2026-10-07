"""Train-only metadata diagnostics for a locked video pilot; no media or holdout."""

from __future__ import annotations

import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from deepfake_detection_archive.data.manifest import file_sha256, load_manifest
from deepfake_detection_archive.evaluation.metrics import evaluate_binary


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "outputs/experiment-planning/vif-pilot"
OUTPUT = ROOT / "outputs/experiments/vif-metadata-shortcut-v1.json"
FEATURE_SETS = {
    "width_only": ("width",),
    "numeric_media": ("width", "height", "fps", "duration_seconds", "bit_rate"),
}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("diagnostic already exists; retain the original")
    records = load_manifest(WORK / "manifest.jsonl")
    audit = json.loads((WORK / "decode-audit.json").read_text())
    metadata = {row["sample_id"]: row for row in audit["samples"]}
    selected = [row for row in records if row.split in {"train", "dev"}]
    results: dict[str, object] = {}
    for name, fields in FEATURE_SETS.items():
        matrices, labels = {}, {}
        for split in ("train", "dev"):
            subset = [row for row in selected if row.split == split]
            values = []
            for row in subset:
                meta = metadata[row.sample_id]
                if meta["media_sha256"] != row.sha256:
                    raise ValueError("decode metadata differs from the locked bytes")
                values.append(
                    [
                        float(Fraction(meta["frame_rate"]))
                        if field == "fps"
                        else float(meta[field])
                        for field in fields
                    ]
                )
            matrices[split] = np.asarray(values, dtype=np.float64)
            labels[split] = np.asarray([row.label for row in subset], dtype=np.int64)
            if not np.isfinite(matrices[split]).all() or set(labels[split]) != {0, 1}:
                raise ValueError("missing/nonfinite metadata or binary class")
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        )
        model.fit(matrices["train"], labels["train"])
        scores = model.decision_function(matrices["dev"])
        results[name] = {
            "fields": list(fields),
            "eer": asdict(evaluate_binary(labels["dev"].tolist(), scores.tolist())),
            "train_count": len(labels["train"]),
            "dev_count": len(labels["dev"]),
            "dev_scores": [
                {"sample_id": row.sample_id, "label": row.label, "score": float(score)}
                for row, score in zip(
                    [row for row in selected if row.split == "dev"], scores, strict=True
                )
            ],
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "scope": "exploratory shortcut diagnosis; metadata-only, train fitted, dev reported",
                "models": results,
                "holdout_used": False,
                "independent_group_ci": False,
                "formal_candidate": False,
                "detector_scores_used_for_selection": False,
                "manifest_sha256": file_sha256(WORK / "manifest.jsonl"),
                "decode_audit_sha256": file_sha256(WORK / "decode-audit.json"),
                "script_sha256": file_sha256(Path(__file__)),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({name: result["eer"]["eer"] for name, result in results.items()}))


if __name__ == "__main__":
    main()
