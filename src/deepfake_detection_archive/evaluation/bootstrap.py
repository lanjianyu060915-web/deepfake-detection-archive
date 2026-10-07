"""Paired bootstrap resampling whole dependency groups within fixed strata."""

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from math import floor, isfinite

from .metrics import UndefinedMetricError, evaluate_binary, validate_inputs


@dataclass(frozen=True)
class BootstrapResult:
    delta_eer: float
    ci_lower: float | None
    ci_upper: float | None
    confidence: float
    valid_repeats: int
    attempts: int
    invalid_repeats: int
    invalid_fraction: float
    group_count: int
    real_group_count: int
    fake_group_count: int
    evidence_sufficient: bool
    insufficiency_reasons: tuple[str, ...]
    seed: int
    group_order: tuple[str, ...]
    strata: tuple[tuple[int, ...], ...]
    draw_sha256: str
    # Each row is one attempted draw, shared by baseline and candidate.
    # Disabled by default to bound memory; seed/order/strata/hash permit replay.
    draws: tuple[tuple[int, ...], ...] | None
    deltas: tuple[float, ...]


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _validate_settings(
    integers: dict[str, int], confidence: float, invalid_fraction: float, seed: int
) -> None:
    for name, value in integers.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if not isfinite(invalid_fraction) or not 0 <= invalid_fraction <= 1:
        raise ValueError("max_invalid_fraction must be between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


def paired_group_bootstrap(  # noqa: PLR0913 -- aligned vectors and explicit protocol knobs
    labels: Sequence[int],
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    groups: Sequence[str],
    sources: Sequence[str],
    *,
    seed: int = 0,
    target_repeats: int = 2000,
    max_attempts: int = 20000,
    confidence: float = 0.95,
    max_invalid_fraction: float = 0.05,
    min_groups: int = 20,
    min_groups_per_class: int = 10,
    retain_draws: bool = False,
) -> BootstrapResult:
    """Bootstrap candidate-minus-baseline EER using identical sampled groups.

    Strata are the group source-set and group label-set, established without
    using scores. A group spanning sources remains indivisible. Each stratum
    draws its original number of groups with replacement. No automatic sparse
    stratum merging is done. CI evidence sufficiency is only one promotion gate;
    multi-seed, source bucket, failure and throughput checks remain separate.
    """
    validate_inputs(labels, baseline_scores)
    validate_inputs(labels, candidate_scores)
    if len(groups) != len(labels) or len(sources) != len(labels):
        raise ValueError("groups and sources must match label length")
    if any(
        not isinstance(value, str) or not value.strip() for value in (*groups, *sources)
    ):
        raise ValueError("groups and sources must be nonempty strings")
    if unknown_groups := [
        index
        for index, group in enumerate(groups)
        if group.strip().casefold() == "unknown"
    ]:
        raise ValueError(
            f"bootstrap requires known independent groups; rows={unknown_groups}"
        )
    _validate_settings(
        {
            "target_repeats": target_repeats,
            "max_attempts": max_attempts,
            "min_groups": min_groups,
            "min_groups_per_class": min_groups_per_class,
        },
        confidence,
        max_invalid_fraction,
        seed,
    )
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)
    group_order = tuple(sorted(by_group))
    indices = tuple(tuple(by_group[group]) for group in group_order)
    by_stratum: dict[tuple[tuple[str, ...], tuple[int, ...]], list[int]] = {}
    real_groups = fake_groups = 0
    for group_index, rows in enumerate(indices):
        group_labels = tuple(sorted({labels[row] for row in rows}))
        group_sources = tuple(sorted({sources[row] for row in rows}))
        real_groups += int(0 in group_labels)
        fake_groups += int(1 in group_labels)
        by_stratum.setdefault((group_sources, group_labels), []).append(group_index)
    strata = tuple(tuple(by_stratum[key]) for key in sorted(by_stratum))
    rng = random.Random(seed)  # noqa: S311 -- reproducible statistical resampling
    draw_hash = hashlib.sha256()
    stored_draws: list[tuple[int, ...]] = []
    deltas: list[float] = []
    attempts = invalid = 0
    while len(deltas) < target_repeats and attempts < max_attempts:
        drawn = tuple(rng.choice(stratum) for stratum in strata for _ in stratum)
        draw_hash.update((json.dumps(drawn, separators=(",", ":")) + "\n").encode())
        if retain_draws:
            stored_draws.append(drawn)
        sampled = [row for group_index in drawn for row in indices[group_index]]
        sample_labels = [labels[row] for row in sampled]
        attempts += 1
        try:
            baseline = evaluate_binary(
                sample_labels, [baseline_scores[row] for row in sampled]
            ).eer
            candidate = evaluate_binary(
                sample_labels, [candidate_scores[row] for row in sampled]
            ).eer
        except UndefinedMetricError:
            invalid += 1
            continue
        deltas.append(candidate - baseline)
    invalid_fraction = invalid / attempts
    reasons = tuple(
        reason
        for fails, reason in (
            (len(group_order) < min_groups, "insufficient_independent_groups"),
            (
                min(real_groups, fake_groups) < min_groups_per_class,
                "insufficient_groups_per_class",
            ),
            (len(deltas) < target_repeats, "insufficient_valid_repeats"),
            (invalid_fraction > max_invalid_fraction, "excess_invalid_repeats"),
        )
        if fails
    )
    alpha = (1 - confidence) / 2
    return BootstrapResult(
        delta_eer=evaluate_binary(labels, candidate_scores).eer
        - evaluate_binary(labels, baseline_scores).eer,
        ci_lower=_percentile(deltas, alpha) if deltas else None,
        ci_upper=_percentile(deltas, 1 - alpha) if deltas else None,
        confidence=confidence,
        valid_repeats=len(deltas),
        attempts=attempts,
        invalid_repeats=invalid,
        invalid_fraction=invalid_fraction,
        group_count=len(group_order),
        real_group_count=real_groups,
        fake_group_count=fake_groups,
        evidence_sufficient=not reasons,
        insufficiency_reasons=tuple(reasons),
        seed=seed,
        group_order=group_order,
        strata=strata,
        draw_sha256=draw_hash.hexdigest(),
        draws=tuple(stored_draws) if retain_draws else None,
        deltas=tuple(deltas),
    )
