"""Selective prediction: the risk-coverage tradeoff behind calibrated abstention.

Given, for every question, a confidence signal and whether the committed answer was correct,
we can ask: if the system abstains on the least-confident questions, how does accuracy on the
answered remainder improve, and at what cost in coverage? Sorting by confidence and sweeping
the cutoff produces the risk-coverage curve. The headline result ("cut errors by X% by
abstaining on the riskiest Y%") is read directly off this curve.

This module is pure and dependency-free: it operates on ``(confidence, correct)`` pairs that
the QA harness produces, so the whole tradeoff is computed once, offline, with no extra LLM
calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class CoveragePoint:
    """One operating point on the risk-coverage curve."""

    coverage: float  # fraction of questions answered (not abstained)
    n_answered: int
    accuracy: float  # accuracy among answered
    risk: float  # error rate among answered (1 - accuracy)
    min_confidence: float  # lowest confidence still answered at this point


def risk_coverage_curve(items: Sequence[tuple[float, bool]]) -> list[CoveragePoint]:
    """Compute the risk-coverage curve.

    Args:
        items: ``(confidence, correct)`` for every question, forced to answer (no abstention).

    Returns:
        Points from highest coverage (answer everything) to lowest, as the confidence cutoff
        rises. Questions are answered in descending confidence order; ties keep input order.
    """
    n = len(items)
    if n == 0:
        return []
    ordered = sorted(items, key=lambda it: it[0], reverse=True)
    points: list[CoveragePoint] = []
    correct_so_far = 0
    for i, (conf, correct) in enumerate(ordered, start=1):
        correct_so_far += 1 if correct else 0
        accuracy = correct_so_far / i
        points.append(
            CoveragePoint(
                coverage=i / n,
                n_answered=i,
                accuracy=accuracy,
                risk=1.0 - accuracy,
                min_confidence=conf,
            )
        )
    return points


def summarize_abstention(
    items: Sequence[tuple[float, bool]],
    abstain_fraction: float,
) -> dict[str, float]:
    """Summarize the effect of abstaining on the riskiest ``abstain_fraction`` of questions.

    Returns baseline error (answer everything), the error after abstaining on the least
    confident fraction, and absolute/relative error reduction, plus the resulting coverage.
    """
    if not 0.0 <= abstain_fraction < 1.0:
        raise ValueError("abstain_fraction must be in [0, 1)")
    n = len(items)
    if n == 0:
        return {
            "baseline_error": 0.0,
            "kept_error": 0.0,
            "error_reduction_abs": 0.0,
            "error_reduction_rel": 0.0,
            "coverage": 0.0,
            "n_total": 0,
            "n_kept": 0,
        }
    ordered = sorted(items, key=lambda it: it[0], reverse=True)
    baseline_correct = sum(1 for _, c in ordered if c)
    baseline_error = 1.0 - baseline_correct / n

    n_keep = max(1, n - int(round(abstain_fraction * n)))
    kept = ordered[:n_keep]
    kept_correct = sum(1 for _, c in kept if c)
    kept_error = 1.0 - kept_correct / n_keep

    abs_red = baseline_error - kept_error
    rel_red = abs_red / baseline_error if baseline_error > 0 else 0.0
    return {
        "baseline_error": baseline_error,
        "kept_error": kept_error,
        "error_reduction_abs": abs_red,
        "error_reduction_rel": rel_red,
        "coverage": n_keep / n,
        "n_total": n,
        "n_kept": n_keep,
    }


def accuracy_at_coverage(items: Sequence[tuple[float, bool]], coverage: float) -> float:
    """Accuracy among the most-confident ``coverage`` fraction of questions."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    n = len(items)
    if n == 0:
        return 0.0
    ordered = sorted(items, key=lambda it: it[0], reverse=True)
    n_keep = max(1, int(round(coverage * n)))
    kept = ordered[:n_keep]
    return sum(1 for _, c in kept if c) / n_keep


def area_under_risk_coverage(items: Sequence[tuple[float, bool]]) -> float:
    """Mean risk across all coverage levels (lower is better).

    This is the discrete area under the risk-coverage curve, a single-number summary of how
    well the confidence signal ranks correct answers above incorrect ones.
    """
    points = risk_coverage_curve(items)
    if not points:
        return 0.0
    return sum(p.risk for p in points) / len(points)
