import math

from citemd.eval.stats import calibration, wilson_interval


def test_wilson_interval_basics():
    p = wilson_interval(0, 0)
    assert p.value == 0.0 and p.lo == 0.0 and p.hi == 0.0
    p = wilson_interval(75, 100)
    assert math.isclose(p.value, 0.75)
    assert p.lo < 0.75 < p.hi
    assert 0.0 <= p.lo and p.hi <= 1.0
    # Interval narrows as n grows.
    wide = wilson_interval(7, 10)
    narrow = wilson_interval(700, 1000)
    assert (wide.hi - wide.lo) > (narrow.hi - narrow.lo)


def test_wilson_interval_clamps_at_extremes():
    p = wilson_interval(10, 10)  # 100%
    assert p.value == 1.0 and math.isclose(p.hi, 1.0) and p.lo < 1.0
    p = wilson_interval(0, 10)  # 0%
    assert p.value == 0.0 and math.isclose(p.lo, 0.0, abs_tol=1e-12) and p.hi > 0.0


def test_calibration_perfect_is_zero_ece():
    # Confidence exactly equals accuracy within each bin -> ECE 0.
    items = [(0.95, True)] * 19 + [(0.95, False)]  # bin ~0.9-1.0: conf 0.95, acc 0.95
    cal = calibration(items, n_bins=10)
    assert cal.n == 20
    assert cal.ece < 0.02


def test_calibration_detects_overconfidence():
    # High confidence, low accuracy -> large ECE.
    items = [(0.99, False)] * 10 + [(0.99, True)] * 2  # conf ~0.99, acc ~0.17
    cal = calibration(items, n_bins=10)
    assert cal.ece > 0.5


def test_calibration_empty():
    cal = calibration([], n_bins=10)
    assert cal.ece == 0.0 and cal.n == 0 and cal.bins == []
