"""Context-local host wall-clock timings for the ordered inference runner."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal


InferenceStage = Literal[
    "input_validation", "decode_and_preprocess", "model_and_feature_scoring"
]


@dataclass
class InferenceStageTimings:
    """Accumulate disjoint scopes for this invocation, never historical rows."""

    seconds: dict[InferenceStage, float] = field(
        default_factory=lambda: {
            "input_validation": 0.0,
            "decode_and_preprocess": 0.0,
            "model_and_feature_scoring": 0.0,
        }
    )
    measured_scopes: int = 0


_active_timings: ContextVar[InferenceStageTimings | None] = ContextVar(
    "inference_stage_timings", default=None
)


@contextmanager
def collect_inference_stages(timings: InferenceStageTimings) -> Iterator[None]:
    token = _active_timings.set(timings)
    try:
        yield
    finally:
        _active_timings.reset(token)


@contextmanager
def inference_stage(stage: InferenceStage) -> Iterator[None]:
    """Time a non-nested scope, including failures, only inside inference.

    This adds no CUDA synchronization. Built-in scoring scopes include their
    existing blocking CPU result copies; they are not device-kernel timings.
    """
    timings = _active_timings.get()
    if timings is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        timings.seconds[stage] += time.perf_counter() - started
        timings.measured_scopes += 1
