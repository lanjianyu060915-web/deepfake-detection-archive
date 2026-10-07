"""Explicit local CSV/JSONL conversion; official status requires confirmed config."""

from __future__ import annotations

import csv
import io
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from deepfake_detection_archive.contracts import ScoreRecord
from deepfake_detection_archive.tracking import atomic_json, canonical_json, object_hash


@dataclass(frozen=True)
class AdapterConfig:
    format: str = "csv"
    sample_id_column: str = "sample_id"
    score_column: str = "score"
    score_direction: str = "fake_high"
    official_confirmed: bool = False

    def __post_init__(self) -> None:
        if self.format not in ("csv", "jsonl"):
            raise ValueError("format must be csv or jsonl")
        if self.score_direction not in ("fake_high", "real_high"):
            raise ValueError("score_direction must be fake_high or real_high")
        columns = (self.sample_id_column, self.score_column)
        if any(not isinstance(key, str) or not key.strip() for key in columns):
            raise ValueError("column names must be nonempty strings")
        if len(set(columns)) != 2:
            raise ValueError("sample ID and score columns must differ")
        if type(self.official_confirmed) is not bool:
            raise ValueError("official_confirmed must be boolean")


def _export_rows(
    scores: Sequence[ScoreRecord], config: AdapterConfig
) -> list[dict[str, str | float]]:
    ids = [score.sample_id for score in scores]
    if not scores or len(set(ids)) != len(ids):
        raise ValueError("export requires nonempty scores with unique sample IDs")
    failed = [score.sample_id for score in scores if score.status != "ok"]
    if failed:
        raise ValueError(
            f"cannot export failed samples without an approved fallback: {failed}"
        )
    rows: list[dict[str, str | float]] = []
    for score in scores:
        if score.score is None:
            raise ValueError("successful row has no score")
        # Scores are unconstrained real-valued detector outputs, not probabilities.
        value = score.score if config.score_direction == "fake_high" else -score.score
        rows.append(
            {config.sample_id_column: score.sample_id, config.score_column: value}
        )
    return rows


def export_scores(
    scores: Sequence[ScoreRecord],
    path: Path,
    config: AdapterConfig | None = None,
    *,
    official: bool = False,
) -> Path:
    """Export complete scores with metadata sidecar; label never affects conversion.

    real_high negates the raw detector score, preserving reverse ranking without
    assuming scores are calibrated probabilities. This does not certify the
    official score range or container schema until the platform contract arrives.
    """
    config = config or AdapterConfig()
    if official and not config.official_confirmed:
        raise ValueError(
            "official export requires an explicitly confirmed platform contract"
        )
    rows = _export_rows(scores, config)
    buffer = io.StringIO(newline="")
    if config.format == "csv":
        writer = csv.DictWriter(
            buffer, fieldnames=[config.sample_id_column, config.score_column]
        )
        writer.writeheader()
        writer.writerows(rows)
    else:
        buffer.writelines(canonical_json(row) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(buffer.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(
        path.with_suffix(path.suffix + ".metadata.json"),
        {
            "official": official,
            "official_confirmed": config.official_confirmed,
            "format": config.format,
            "score_direction": config.score_direction,
            "score_transform": "identity"
            if config.score_direction == "fake_high"
            else "negate",
            "sample_count": len(scores),
            "ordered_score_hash": object_hash([score.to_dict() for score in scores]),
        },
    )
    return path
