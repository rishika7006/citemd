import pytest

# Charts need the optional viz extra; skip cleanly where matplotlib is absent (e.g. CI core).
pytest.importorskip("matplotlib")

from citemd.eval.charts import plot_ablation, plot_risk_coverage  # noqa: E402


def test_plot_risk_coverage_writes_file(tmp_path):
    items = [(0.9, True), (0.5, False), (0.3, True)]
    out = plot_risk_coverage(items, tmp_path / "rc.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_ablation_writes_file(tmp_path):
    rows = [
        {"arm": "vector", "n": 10, "accuracy": 0.6, "error": 0.4, "coverage": 1.0},
        {"arm": "hybrid", "n": 10, "accuracy": 0.7, "error": 0.3, "coverage": 1.0},
    ]
    out = plot_ablation(rows, tmp_path / "ab.png")
    assert out.exists() and out.stat().st_size > 0
