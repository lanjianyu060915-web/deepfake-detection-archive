"""Versioned score contracts; real=0, fake=1, larger scores mean more fake."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import cast

from .tracking import parse_json


@dataclass(frozen=True)
class ScoreRecord:
    sample_id: str
    modality: str
    score: float | None
    status: str
    elapsed_seconds: float
    label: int | None = None
    group_id: str = "unknown"
    source: str = "unknown"
    content_type: str = "unknown"
    generator: str = "unknown"
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        for key in ("sample_id", "source", "content_type", "generator", "group_id"):
            if (
                not isinstance(getattr(self, key), str)
                or not getattr(self, key).strip()
            ):
                raise ValueError(f"{key} must be a nonempty string")
        if self.modality not in ("audio", "video"):
            raise ValueError("modality must be audio or video")
        if self.label is not None and (
            type(self.label) is not int or self.label not in (0, 1)
        ):
            raise ValueError("label must be integer 0, 1 or null")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (float, int))
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be nonnegative finite")
        if self.status == "ok":
            if (
                isinstance(self.score, bool)
                or not isinstance(self.score, (float, int))
                or not math.isfinite(self.score)
                or self.error_code is not None
                or self.error_message is not None
            ):
                raise ValueError("successful scores must be finite without errors")
        elif self.status == "error":
            if self.score is not None or any(
                not isinstance(value, str) or not value.strip()
                for value in (self.error_code, self.error_message)
            ):
                raise ValueError("failed scores need null score and explicit error")
        else:
            raise ValueError("status must be ok or error")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def score_from_dict(value: dict[str, object]) -> ScoreRecord:
    if set(value) - {field.name for field in fields(ScoreRecord)}:
        raise ValueError("unknown score fields")
    try:
        return ScoreRecord(**value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"invalid score schema: {exc}") from exc


def load_scores(path: Path) -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = parse_json(line)
            if not isinstance(value, dict):
                raise ValueError("score must be an object")  # noqa: TRY004, TRY301 -- schema
            records.append(score_from_dict(cast(dict[str, object], value)))
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    ids = [record.sample_id for record in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError("scores must be nonempty with unique sample IDs")
    return records
