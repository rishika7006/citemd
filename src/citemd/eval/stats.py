"""Small statistics helpers for reporting: confidence intervals and calibration.

Pure and dependency-free. These turn raw counts and ``(confidence, correct)`` pairs into
numbers that can be reported honestly: a proportion with a Wilson interval (so "within noise"
is explicit), and a calibration summary (expected calibration error plus reliability bins) so
a claim about confidence being meaningful is measured, not asserted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# 1.96 is the standard-normal quantile for a two-sided 95% interval.
Z_95 = 1.959963984540054


@dataclass
class Proportion:
    """A proportion k/n with a Wilson score confidence interval."""

    k: int
    n: int
    value: float
    lo: float
    hi: float

    def as_pct(self) -> str:
        return f"{self.value * 100:.1f}% (95% CI {self.lo * 100:.1f}-{self.hi * 100:.1f})"


def wilson_interval(k: int, n: int, z: float = Z_95) -> Proportion:
    """Wilson score interval for a binomial proportion.

    More accurate than the normal approximation for small n and proportions near 0 or 1, which
    is exactly the regime these evaluations run in.
    """
    if n <= 0:
        return Proportion(k=0, n=0, value=0.0, lo=0.0, hi=0.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return Proportion(k=k, n=n, value=p, lo=max(0.0, center - half), hi=min(1.0, center + half))


@dataclass
class ReliabilityBin:
    """One bin of a reliability diagram."""

    lo: float
    hi: float
    n: int
    mean_confidence: float
    accuracy: float


@dataclass
class Calibration:
    """Calibration summary: expected calibration error plus the reliability bins."""

    ece: float
    n: int
    bins: list[ReliabilityBin]


def calibration(items: Sequence[tuple[float, bool]], n_bins: int = 10) -> Calibration:
    """Expected calibration error and reliability bins over ``(confidence, correct)`` pairs.

    ECE is the sample-weighted mean gap between confidence and accuracy across equal-width
    confidence bins. Lower is better; 0 means the confidence equals the empirical accuracy in
    every bin. Reported as a measured property, not a guarantee of calibration.
    """
    n = len(items)
    if n == 0:
        return Calibration(ece=0.0, n=0, bins=[])
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins: list[ReliabilityBin] = []
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # Last bin is closed on the right so confidence == 1.0 lands somewhere.
        in_bin = [
            (c, ok)
            for c, ok in items
            if (lo <= c < hi) or (b == n_bins - 1 and c == hi)
        ]
        if not in_bin:
            bins.append(ReliabilityBin(lo=lo, hi=hi, n=0, mean_confidence=0.0, accuracy=0.0))
            continue
        m = len(in_bin)
        mean_conf = sum(c for c, _ in in_bin) / m
        acc = sum(1 for _, ok in in_bin if ok) / m
        ece += (m / n) * abs(mean_conf - acc)
        bins.append(ReliabilityBin(lo=lo, hi=hi, n=m, mean_confidence=mean_conf, accuracy=acc))
    return Calibration(ece=ece, n=n, bins=bins)
