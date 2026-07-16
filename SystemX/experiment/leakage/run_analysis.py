from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import nbformat

from SystemX.parser.graph_parser import GraphParser
from SystemX.sca.leakage import LeakageClass, analyze_leakage

logger = logging.getLogger(__name__)

_ALL_CLASSES = [c.value for c in LeakageClass]

def _load_cells(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    return [dict(c) for c in nb.cells if c.get("cell_type") == "code"]

def analyze_notebook(path: Path) -> dict:
    """Parse one notebook and run all detectors."""
    try:
        cells = _load_cells(path)
        graph = GraphParser().parse(cells).get_graph()
        findings = [f.to_dict() for f in analyze_leakage(graph)]
        has_split = any(n.startswith(("call_train test split", "call_train_test_split")) for n in graph.nodes())
        return {"notebook": str(path), "status": "ok", "has_split": has_split, "findings": findings}
    except Exception as exc:
        logger.debug("parse failed for %s: %s", path, exc)
        return {"notebook": str(path), "status": f"error:{type(exc).__name__}", "has_split": False, "findings": []}

def summarize(records: list[dict]) -> dict:
    parsed = [r for r in records if r["status"] == "ok"]
    n_parsed = len(parsed)
    with_split = [r for r in parsed if r["has_split"]]
    class_counts: Counter = Counter()
    nb_with_class: Counter = Counter()
    nb_any = 0
    for r in parsed:
        classes = {f["leakage_class"] for f in r["findings"]}
        if classes:
            nb_any += 1
        for c in classes:
            nb_with_class[c] += 1
        for f in r["findings"]:
            class_counts[f["leakage_class"]] += 1

    def _pct(num: int, den: int) -> float:
        return round(100.0 * num / den, 2) if den else 0.0

    return {
        "n_notebooks": len(records),
        "n_parsed": n_parsed,
        "n_parse_errors": len(records) - n_parsed,
        "n_with_split": len(with_split),
        "pct_with_any_leakage": _pct(nb_any, n_parsed),
        "per_class": {
            c: {
                "n_findings": class_counts.get(c, 0),
                "n_notebooks": nb_with_class.get(c, 0),
                "pct_notebooks": _pct(nb_with_class.get(c, 0), n_parsed),
            }
            for c in _ALL_CLASSES
        },
    }

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Static leakage prevalence over a notebook corpus.")
    ap.add_argument("--notebooks", required=True, type=Path, help="Directory of .ipynb files (searched recursively).")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0, help="Cap notebooks processed (0 = all).")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = sorted(args.notebooks.rglob("*.ipynb"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        logger.error("No .ipynb files found under %s", args.notebooks)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    findings_path = args.output_dir / "findings.jsonl"
    records: list[dict] = []
    with open(findings_path, "w", encoding="utf-8") as fh:
        for i, p in enumerate(paths, 1):
            rec = analyze_notebook(p)
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            if i % 200 == 0:
                logger.info("  processed %d/%d notebooks", i, len(paths))

    summary = summarize(records)
    with open(args.output_dir / "prevalence.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("\n== Leakage prevalence (%d parsed / %d total) ==", summary["n_parsed"], summary["n_notebooks"])
    logger.info("  any leakage: %.2f%% of parsed notebooks", summary["pct_with_any_leakage"])
    for c, s in summary["per_class"].items():
        logger.info("  %-28s %6d findings  %6.2f%% of notebooks", c, s["n_findings"], s["pct_notebooks"])
    logger.info("\nWrote %s and prevalence.json", findings_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
