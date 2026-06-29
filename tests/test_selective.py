import math

from citemd.eval.selective import (
    accuracy_at_coverage,
    area_under_risk_coverage,
    risk_coverage_curve,
    summarize_abstention,
)

# Confidence-sorted scenario: the two wrong answers are the least confident, so abstaining
# on the riskiest fraction should remove errors first.
ITEMS = [
    (0.9, True),
    (0.8, True),
    (0.7, True),
    (0.6, True),
    (0.3, False),
    (0.1, False),
]


def test_risk_coverage_curve_is_monotone_in_coverage():
    pts = risk_coverage_curve(ITEMS)
    assert len(pts) == 6
    assert pts[0].coverage == 1 / 6
    assert pts[-1].coverage == 1.0
    # Full coverage: 4/6 correct -> risk = 1/3
    assert math.isclose(pts[-1].risk, 1 - 4 / 6)
    # Top-4 most confident are all correct -> zero risk at coverage 4/6
    p4 = [p for p in pts if p.n_answered == 4][0]
    assert p4.risk == 0.0


def test_summarize_abstention_reduces_error():
    s = summarize_abstention(ITEMS, abstain_fraction=1 / 3)  # drop riskiest 2
    assert s["n_kept"] == 4
    assert math.isclose(s["baseline_error"], 2 / 6)
    assert s["kept_error"] == 0.0
    assert math.isclose(s["error_reduction_rel"], 1.0)
    assert math.isclose(s["coverage"], 4 / 6)


def test_accuracy_at_coverage():
    assert accuracy_at_coverage(ITEMS, 0.5) == 1.0  # top 3 all correct
    assert math.isclose(accuracy_at_coverage(ITEMS, 1.0), 4 / 6)


def test_area_under_risk_coverage_lower_when_well_ranked():
    well = [(0.9, True), (0.8, True), (0.2, False)]
    poorly = [(0.9, False), (0.8, True), (0.2, True)]
    assert area_under_risk_coverage(well) < area_under_risk_coverage(poorly)


def test_empty_inputs():
    assert risk_coverage_curve([]) == []
    assert area_under_risk_coverage([]) == 0.0
    assert summarize_abstention([], 0.2)["n_total"] == 0
