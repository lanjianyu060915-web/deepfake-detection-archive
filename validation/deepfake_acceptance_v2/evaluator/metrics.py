"""Dependency-free binary metrics for the D-v2 deepfake evaluator.

FAKE is always the positive class and REAL is always the negative class.
Undefined metrics are represented as JSON-safe ``None`` values; callers must
not silently replace an undefined value with zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import isfinite
from typing import Any


POSITIVE_LABEL = "FAKE"
NEGATIVE_LABEL = "REAL"
VALID_LABELS = frozenset({NEGATIVE_LABEL, POSITIVE_LABEL})


def _metric(numerator: int | float, denominator: int | float) -> dict[str, Any]:
    if denominator == 0:
        return {
            "value": None,
            "status": "NOT_COMPUTED_ZERO_DENOMINATOR",
            "numerator": numerator,
            "denominator": denominator,
        }
    return {
        "value": numerator / denominator,
        "status": "COMPUTED",
        "numerator": numerator,
        "denominator": denominator,
    }


def compute_binary_metrics(
    truth_and_prediction: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Compute a complete confusion matrix and derived metrics.

    The caller is responsible for converting every execution failure to the
    opposite effective prediction before calling this function. This makes the
    fixed-denominator invariant explicit: ``TP + FP + TN + FN == N``.
    """

    tp = fp = tn = fn = 0
    for truth, prediction in truth_and_prediction:
        if truth not in VALID_LABELS:
            raise ValueError(f"invalid ground-truth label: {truth!r}")
        if prediction not in VALID_LABELS:
            raise ValueError(f"invalid effective prediction: {prediction!r}")
        if truth == POSITIVE_LABEL and prediction == POSITIVE_LABEL:
            tp += 1
        elif truth == NEGATIVE_LABEL and prediction == POSITIVE_LABEL:
            fp += 1
        elif truth == NEGATIVE_LABEL and prediction == NEGATIVE_LABEL:
            tn += 1
        else:
            fn += 1

    n = tp + fp + tn + fn
    return {
        "N": n,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": _metric(tp + tn, n),
        "Precision": _metric(tp, tp + fp),
        "Recall": _metric(tp, tp + fn),
        "Sensitivity": _metric(tp, tp + fn),
        "Specificity": _metric(tn, tn + fp),
        "F1": _metric(2 * tp, 2 * tp + fp + fn),
    }


def compute_roc_auc(
    labels: Sequence[str],
    scores: Sequence[float | int | None],
    *,
    score_semantics: str,
    all_predictions_usable: bool,
) -> dict[str, Any]:
    """Conditionally compute ROC-AUC using the pairwise/rank definition.

    AUC is deliberately all-or-nothing. It is never calculated on a filtered
    subset because doing so would silently change the D-v2 denominator.
    """

    if score_semantics != "higher_is_more_fake":
        return {
            "value": None,
            "status": "NOT_COMPUTED",
            "reason": "score semantics were not explicitly confirmed as higher_is_more_fake",
            "N": len(labels),
        }
    if not all_predictions_usable:
        return {
            "value": None,
            "status": "NOT_COMPUTED",
            "reason": "one or more samples did not produce a usable prediction",
            "N": len(labels),
        }
    if len(labels) != len(scores):
        raise ValueError("labels and scores have different lengths")
    if any(label not in VALID_LABELS for label in labels):
        raise ValueError("ROC-AUC received an invalid ground-truth label")
    if any(
        score is None
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not isfinite(float(score))
        for score in scores
    ):
        return {
            "value": None,
            "status": "NOT_COMPUTED",
            "reason": "at least one sample has no finite numeric score",
            "N": len(labels),
        }

    positive_scores = [
        float(score)
        for label, score in zip(labels, scores, strict=True)
        if label == POSITIVE_LABEL
    ]
    negative_scores = [
        float(score)
        for label, score in zip(labels, scores, strict=True)
        if label == NEGATIVE_LABEL
    ]
    if not positive_scores or not negative_scores:
        return {
            "value": None,
            "status": "NOT_COMPUTED",
            "reason": "ground truth does not contain both REAL and FAKE",
            "N": len(labels),
        }

    wins = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    pair_count = len(positive_scores) * len(negative_scores)
    return {
        "value": wins / pair_count,
        "status": "COMPUTED",
        "reason": None,
        "N": len(labels),
        "positive_count": len(positive_scores),
        "negative_count": len(negative_scores),
        "pair_count": pair_count,
    }

