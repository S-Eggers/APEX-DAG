from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from SystemX.parser.graph_parser import GraphParser
from SystemX.sca.leakage import LeakageClass, analyze_leakage

from .run_analysis import _load_cells

logger = logging.getLogger(__name__)

_CLASSES = [c.value for c in LeakageClass]

def _predict(path: Path) -> set[str]:
    cells = _load_cells(path)
    graph = GraphParser().parse(cells).get_graph()
    return {f.leakage_class.value for f in analyze_leakage(graph)}

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

def evaluate(benchmark: Path) -> dict:
    manifest = json.loads((benchmark / "ground_truth.json").read_text())
    gt: dict[str, list[str]] = manifest["ground_truth"]
    nb_dir = benchmark / "notebooks"

    counts = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in _CLASSES}
    errors: list[str] = []
    per_notebook: list[dict] = []
    for name, gold_list in gt.items():
        path = nb_dir / name
        try:
            pred = _predict(path)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}")
            continue
        gold = set(gold_list)
        per_notebook.append({"notebook": name, "gold": sorted(gold), "pred": sorted(pred)})
        for c in _CLASSES:
            g, p = c in gold, c in pred
            key = "tp" if (g and p) else "fn" if g else "fp" if p else "tn"
            counts[c][key] += 1

    per_class = {}
    macro_f = []
    tot = {"tp": 0, "fp": 0, "fn": 0}
    for c in _CLASSES:
        cc = counts[c]
        p, r, f = _prf(cc["tp"], cc["fp"], cc["fn"])
        per_class[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), **cc}
        macro_f.append(f)
        for k in tot:
            tot[k] += cc[k]
    micro_p, micro_r, micro_f = _prf(tot["tp"], tot["fp"], tot["fn"])

    return {
        "n_notebooks": len(gt),
        "n_errors": len(errors),
        "errors": errors,
        "per_class": per_class,
        "macro_f1": round(sum(macro_f) / len(macro_f), 4) if macro_f else 0.0,
        "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f, 4)},
        "per_notebook": per_notebook,
    }

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate leakage detectors on a labelled benchmark.")
    ap.add_argument("--benchmark", required=True, type=Path, help="Dir with ground_truth.json + notebooks/.")
    ap.add_argument("--output", type=Path, default=None, help="Optional path to write the full report JSON.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = evaluate(args.benchmark)

    logger.info("\n== Leakage detector evaluation (%d notebooks, %d parse errors) ==",
                report["n_notebooks"], report["n_errors"])
    logger.info("  %-30s %8s %8s %8s", "class", "P", "R", "F1")
    for c, m in report["per_class"].items():
        logger.info("  %-30s %8.3f %8.3f %8.3f", c, m["precision"], m["recall"], m["f1"])
    logger.info("  %-30s %8.3f %8.3f %8.3f  (micro)",
                "OVERALL", report["micro"]["precision"], report["micro"]["recall"], report["micro"]["f1"])
    logger.info("  macro-F1: %.3f", report["macro_f1"])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        logger.info("\nWrote full report to %s", args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
