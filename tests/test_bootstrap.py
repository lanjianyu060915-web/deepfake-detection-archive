"""Bootstrap pairing, independent groups, strata, reproducibility and gates."""

import hashlib
import json

import pytest

from deepfake_detection_archive.evaluation.bootstrap import paired_group_bootstrap
from deepfake_detection_archive.evaluation.metrics import (
    BinaryMetrics,
    UndefinedMetricError,
    evaluate_binary,
)


def test_paired_identical_models_have_zero_width_ci_and_replayable_draws() -> None:
    labels = [0, 1] * 20
    scores = [float(index % 5) for index in range(40)]
    groups = [f"g{index // 2:02}" for index in range(40)]
    sources = ["source"] * 40
    result = paired_group_bootstrap(
        labels,
        scores,
        scores,
        groups,
        sources,
        target_repeats=30,
        retain_draws=True,
    )
    assert result.evidence_sufficient
    assert result.delta_eer == result.ci_lower == result.ci_upper == 0
    assert result.invalid_repeats == 0
    assert (
        result.group_count == result.real_group_count == result.fake_group_count == 20
    )
    assert result.draws is not None
    draw_hash = hashlib.sha256()
    for draw in result.draws:
        assert len(draw) == 20
        draw_hash.update((json.dumps(draw, separators=(",", ":")) + "\n").encode())
    assert draw_hash.hexdigest() == result.draw_sha256
    repeated = paired_group_bootstrap(
        labels,
        scores,
        scores,
        groups,
        sources,
        target_repeats=30,
        retain_draws=True,
    )
    assert repeated == result


def test_shared_group_draws_reproduce_deltas_without_splitting_media() -> None:
    labels = [0, 0, 1, 1, 0, 1]
    groups = ["real", "real", "fake", "fake", "mixed", "mixed"]
    sources = ["a", "a", "a", "a", "b", "c"]
    baseline, candidate = [1, 0, 0, 1, 0.2, 0.3], [0, 0.1, 0.5, 0.7, 0.8, 0.9]
    result = paired_group_bootstrap(
        labels,
        baseline,
        candidate,
        groups,
        sources,
        target_repeats=5,
        retain_draws=True,
    )
    assert result.draws is not None
    assert len(result.strata) == 3  # mixed-source group stays whole.
    for draw, delta in zip(result.draws, result.deltas, strict=True):
        rows = [
            row
            for group_index in draw
            for row, group in enumerate(groups)
            if group == result.group_order[group_index]
        ]
        drawn_labels = [labels[row] for row in rows]
        expected = evaluate_binary(drawn_labels, [candidate[i] for i in rows]).eer
        expected -= evaluate_binary(drawn_labels, [baseline[i] for i in rows]).eer
        assert delta == expected
    assert not result.evidence_sufficient
    assert "insufficient_independent_groups" in result.insufficiency_reasons
    assert "insufficient_groups_per_class" in result.insufficiency_reasons


def test_target_and_attempt_caps_are_real_not_report_only() -> None:
    result = paired_group_bootstrap(
        [0, 1],
        [0, 1],
        [1, 0],
        ["r", "f"],
        ["a", "a"],
        target_repeats=10,
        max_attempts=3,
        min_groups=2,
        min_groups_per_class=1,
    )
    assert result.attempts == result.valid_repeats == 3
    assert result.delta_eer == result.ci_lower == result.ci_upper == 1
    assert result.insufficiency_reasons == ("insufficient_valid_repeats",)


def test_class_coverage_gate_checks_groups_not_rows() -> None:
    labels = [0] * 100 + [1] * 19
    result = paired_group_bootstrap(
        labels,
        labels,
        labels,
        ["single-real"] * 100 + [f"fake-{index}" for index in range(19)],
        ["a"] * len(labels),
        target_repeats=2,
    )
    assert result.group_count == 20
    assert result.real_group_count == 1
    assert result.insufficiency_reasons == ("insufficient_groups_per_class",)


@pytest.mark.parametrize("group", ["", " ", "unknown", " UNKNOWN "])
def test_unknown_group_identity_cannot_supply_promotion_evidence(group: str) -> None:
    with pytest.raises(ValueError, match=r"nonempty|known independent groups"):
        paired_group_bootstrap(
            [0, 1],
            [0, 1],
            [1, 0],
            [group, "known"],
            ["source", "source"],
            target_repeats=1,
            min_groups=1,
            min_groups_per_class=1,
        )


def test_default_valid_repeat_requirement_is_2000() -> None:
    result = paired_group_bootstrap(
        [0, 1],
        [0, 1],
        [1, 0],
        ["r", "f"],
        ["a", "a"],
        min_groups=2,
        min_groups_per_class=1,
    )
    assert result.attempts == result.valid_repeats == 2000
    assert result.evidence_sufficient
    assert result.draws is None


@pytest.mark.parametrize(
    "option", ["target_repeats", "max_attempts", "min_groups", "min_groups_per_class"]
)
def test_invalid_integer_options(option: str) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        paired_group_bootstrap(
            [0, 1], [0, 1], [1, 0], ["r", "f"], ["a", "a"], **{option: 0}
        )


def test_invalid_metadata_and_scores() -> None:
    with pytest.raises(ValueError, match="match label length"):
        paired_group_bootstrap([0, 1], [0, 1], [1, 0], ["r"], ["a", "a"])
    with pytest.raises(ValueError, match="nonempty strings"):
        paired_group_bootstrap([0, 1], [0, 1], [1, 0], ["", "f"], ["a", "a"])
    with pytest.raises(ValueError, match="finite numbers"):
        paired_group_bootstrap(
            [0, 1], [0, 1], [float("nan"), 0], ["r", "f"], ["a", "a"]
        )


@pytest.mark.parametrize(("target", "sufficient"), [(10, False), (19, True)])
def test_undefined_repeats_excluded_and_excess_invalid_disqualifies(
    monkeypatch: pytest.MonkeyPatch,
    target: int,
    sufficient: bool,
) -> None:
    # Inject an undefined metric to exercise the protocol guard. Label-set
    # stratification normally ensures both classes occur in every valid draw.
    calls = 0

    def flaky_metric(labels: list[int], scores: list[float]) -> BinaryMetrics:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise UndefinedMetricError("undefined sampled metric")
        return evaluate_binary(labels, scores)

    monkeypatch.setattr(
        "deepfake_detection_archive.evaluation.bootstrap.evaluate_binary", flaky_metric
    )
    result = paired_group_bootstrap(
        [0, 1],
        [0, 1],
        [1, 0],
        ["r", "f"],
        ["a", "a"],
        target_repeats=target,
        min_groups=2,
        min_groups_per_class=1,
    )
    assert result.valid_repeats == target
    assert result.invalid_repeats == 1
    assert result.attempts == target + 1
    assert result.invalid_fraction == pytest.approx(1 / (target + 1))
    assert result.ci_lower == result.ci_upper == 1
    assert result.evidence_sufficient == sufficient
    assert result.insufficiency_reasons == (
        () if sufficient else ("excess_invalid_repeats",)
    )


def test_percentile_interpolates_between_ranked_paired_deltas() -> None:
    result = paired_group_bootstrap(
        [0, 1, 0, 1],
        [0.1, 0.9, 0.8, 0.2],
        [0.2, 0.8, 0.4, 0.6],
        ["a", "a", "b", "b"],
        ["x"] * 4,
        target_repeats=4,
        confidence=0.5,
        seed=0,
    )
    # Ranked deltas [-1,-1,-1,-.5], linear 25th/75th percentiles.
    assert result.deltas == (-1.0, -0.5, -1.0, -1.0)
    assert result.ci_lower == -1
    assert result.ci_upper == -0.875


def test_all_attempts_undefined_leave_null_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def undefined_draws(labels: list[int], scores: list[float]) -> BinaryMetrics:
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise UndefinedMetricError("undefined sampled metric")
        return evaluate_binary(labels, scores)

    monkeypatch.setattr(
        "deepfake_detection_archive.evaluation.bootstrap.evaluate_binary",
        undefined_draws,
    )
    result = paired_group_bootstrap(
        [0, 1],
        [0, 1],
        [1, 0],
        ["r", "f"],
        ["a", "a"],
        target_repeats=2,
        max_attempts=3,
        min_groups=2,
        min_groups_per_class=1,
    )
    assert result.attempts == result.invalid_repeats == 3
    assert result.valid_repeats == 0
    assert result.ci_lower is result.ci_upper is None
    assert result.insufficiency_reasons == (
        "insufficient_valid_repeats",
        "excess_invalid_repeats",
    )
