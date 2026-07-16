from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from SystemX.experiment.ablation.plot_results import (
    _save,
    fig_title,
)
from SystemX.sca.leakage import LEAKAGE_GOLD_BY_KEY, LeakageClass

logger = logging.getLogger(__name__)

_CLASS_STYLE: dict[str, tuple[str, str]] = {
    key: (LEAKAGE_GOLD_BY_KEY[key]["label"], LEAKAGE_GOLD_BY_KEY[key]["color"])
    for key in (
        LeakageClass.PREPROCESSING_BEFORE_SPLIT.value,
        LeakageClass.TARGET_LEAKAGE.value,
        LeakageClass.TEST_IN_TRAIN.value,
        LeakageClass.METRIC_ON_TRAIN.value,
        LeakageClass.NO_HOLDOUT_EVALUATION.value,
    )
}

def plot_prevalence(prevalence: dict, output_dir: Path, report: dict | None = None) -> Path:
    """Render the prevalence (× reliability) figure; return the PDF path."""
    per_class = prevalence.get("per_class", {})
    classes = sorted(_CLASS_STYLE, key=lambda c: per_class.get(c, {}).get("pct_notebooks", 0.0), reverse=True)
    labels = [_CLASS_STYLE[c][0] for c in classes]
    colors = [_CLASS_STYLE[c][1] for c in classes]
    pct = [per_class.get(c, {}).get("pct_notebooks", 0.0) for c in classes]
    counts = [per_class.get(c, {}).get("n_notebooks", 0) for c in classes]

    ncols = 2 if report else 1
    fig, axes = plt.subplots(1, ncols, figsize=(9.5 if report else 6.0, 3.4), squeeze=False)
    y = range(len(classes))

    ax = axes[0][0]
    ax.barh(list(y), pct, color=colors, edgecolor="white")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("% of parsed notebooks")
    xmax = max([*pct, 1.0])
    ax.set_xlim(0, xmax * 1.18)
    for i, (p, n) in enumerate(zip(pct, counts, strict=False)):
        ax.text(p + xmax * 0.01, i, f"{p:.1f}%  (n={n})", va="center", fontsize=8.5, color="#333")
    n_parsed = prevalence.get("n_parsed", 0)
    ax.set_title(f"Prevalence  (n={n_parsed:,} notebooks)")

    if report:
        rax = axes[0][1]
        pc = report.get("per_class", {})
        f1 = [pc.get(c, {}).get("f1", 0.0) for c in classes]
        rax.barh(list(y), f1, color=colors, edgecolor="white")
        rax.set_yticks(list(y))
        rax.set_yticklabels([])
        rax.invert_yaxis()
        rax.set_xlabel("detector F1 (benchmark)")
        rax.set_xlim(0, 1.05)
        for i, f in enumerate(f1):
            rax.text(min(f + 0.02, 1.0), i, f"{f:.2f}", va="center", fontsize=8.5, color="#333")
        rax.set_title("Detector reliability")

    title = fig_title("fig_leakage_prevalence", default="Static Leakage Prevalence and Detector Reliability")
    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, output_dir, "fig_leakage_prevalence")
    return output_dir / "fig_leakage_prevalence.pdf"

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plot leakage prevalence (× detector reliability).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prevalence", type=Path, help="Path to prevalence.json (from `analyze`).")
    src.add_argument("--analysis_dir", type=Path, help="Directory containing prevalence.json.")
    ap.add_argument("--report", type=Path, default=None, help="Optional evaluate report.json for the F1 panel.")
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prevalence_path = args.prevalence or (args.analysis_dir / "prevalence.json")
    prevalence = json.loads(prevalence_path.read_text())
    report = json.loads(args.report.read_text()) if args.report and args.report.exists() else None

    out = plot_prevalence(prevalence, args.output_dir, report)
    logger.info("Wrote %s (+ .png)", out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
