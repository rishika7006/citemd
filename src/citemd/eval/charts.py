"""Charts for the evaluation report (matplotlib, the optional ``viz`` extra).

Two figures back the headline: the risk-coverage curve (how error among answered questions
falls as the system abstains on the least-confident ones) and the ablation accuracy bars
(vector -> hybrid -> +rerank -> +abstention). Imports are lazy so the core stays light.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from citemd.eval.selective import risk_coverage_curve


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: write files, never open a window
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "Charts need the 'viz' extra. Install with: pip install 'citemd[viz]'"
        ) from exc
    return plt


def plot_risk_coverage(
    items: Sequence[tuple[float, bool]],
    out_path: str | Path,
    *,
    title: str = "Risk-coverage (abstaining on least-confident questions)",
) -> Path:
    """Plot error among answered questions versus coverage; save to ``out_path``."""
    plt = _require_matplotlib()
    points = risk_coverage_curve(items)
    xs = [p.coverage for p in points]
    ys = [p.risk for p in points]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker=".", linewidth=1)
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("error rate among answered")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_ablation(rows: Sequence[dict], out_path: str | Path) -> Path:
    """Bar chart of accuracy per ablation arm; save to ``out_path``."""
    plt = _require_matplotlib()
    names = [r["arm"] for r in rows]
    accs = [r["accuracy"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(names)), accs, color="#3b6ea5")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Ablation: accuracy by pipeline configuration")
    for i, acc in enumerate(accs):
        ax.text(i, acc + 0.01, f"{acc:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
