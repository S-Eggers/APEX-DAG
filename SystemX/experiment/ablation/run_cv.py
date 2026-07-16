#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from SystemX.experiment.ablation.datasets import (
    GENERALIZATION_DATASETS,
    resolve_dataset,
)
from SystemX.experiment.ablation.splits import make_folds
from SystemX.experiment.evaluation.ablation_variant import AblationVariant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python3"
_PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

def _run(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("Command failed (exit %d): %s", result.returncode, " ".join(cmd))
        sys.exit(result.returncode)

def _write_list(path: Path, paths: list[Path]) -> None:
    path.write_text("\n".join(p.name for p in paths) + "\n")

def _mean_std(values: list[float]) -> dict[str, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }

def aggregate(fold_results: list[dict], output_path: Path) -> None:
    """Merge per-fold result dicts into mean ± std per variant/metric."""
    variants = sorted({v for fr in fold_results for v in fr})
    agg: dict[str, dict] = {}

    for variant in variants:
        per_fold = [fr[variant] for fr in fold_results if variant in fr]
        live = [r for r in per_fold if not r.get("skipped")]
        if not live:
            agg[variant] = {"skipped": True, "skip_reason": per_fold[0].get("skip_reason", "") if per_fold else "", "n_folds": 0}
            continue

        def _collect(getter) -> dict[str, float]:
            return _mean_std([getter(r) for r in live])

        metric_names = ("precision", "recall", "f1", "macro_f1")
        global_block = {m: _collect(lambda r, m=m: r.get("global", {}).get(m)) for m in metric_names}
        raw_block = {m: _collect(lambda r, m=m: r.get("raw", {}).get("global", {}).get(m)) for m in metric_names}

        class_names = sorted({c for r in live for c in r.get("per_class", {})})
        per_class = {}
        for cls in class_names:
            per_class[cls] = {
                "f1": _collect(lambda r, c=cls: r.get("per_class", {}).get(c, {}).get("f1")),
                "precision": _collect(lambda r, c=cls: r.get("per_class", {}).get(c, {}).get("precision")),
                "recall": _collect(lambda r, c=cls: r.get("per_class", {}).get(c, {}).get("recall")),
                "support": sum(r.get("per_class", {}).get(cls, {}).get("support", 0) for r in live),
            }

        timing_block = {
            "mean_s": _collect(lambda r: r.get("timing", {}).get("mean_s")),
            "std_s": _collect(lambda r: r.get("timing", {}).get("std_s")),
        }
        per_nb = [e for r in live for e in r.get("timing_per_notebook", [])]

        refiner_ann = {
            "leakage": sum(r.get("refiner_annotations", {}).get("leakage", 0) for r in live),
            "dead_code": sum(r.get("refiner_annotations", {}).get("dead_code", 0) for r in live),
        }

        agg[variant] = {
            "skipped": False,
            "n_folds": len(live),
            "global": global_block,
            "raw": {"global": raw_block},
            "per_class": per_class,
            "timing": timing_block,
            "timing_per_notebook": per_nb,
            "refiner_annotations": refiner_ann,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict = {}
    if output_path.exists():
        try:
            merged = json.loads(output_path.read_text())
        except Exception:
            merged = {}
    merged.update(agg)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    logger.info("Aggregated CV results -> %s (%d variants total, %d from this CV).", output_path, len(merged), len(agg))

def aggregate_tuples(fold_results: list[dict], output_path: Path) -> None:
    """Merge per-fold lineage-tuple result dicts into mean ± std per variant."""
    tuple_types = ("<D, D>", "<M, D>", "<D, Empty>")
    metric_names = ("precision", "recall", "f1")
    variants = sorted({v for fr in fold_results for v in fr})
    agg: dict[str, dict] = {}

    for variant in variants:
        per_fold = [fr[variant] for fr in fold_results if variant in fr]
        live = [r for r in per_fold if not r.get("skipped")]
        if not live:
            agg[variant] = {"skipped": True, "skip_reason": per_fold[0].get("skip_reason", "") if per_fold else "", "n_folds": 0}
            continue

        def _collect(getter) -> dict[str, float]:
            return _mean_std([getter(r) for r in live])

        global_block = {m: _collect(lambda r, m=m: r.get("global", {}).get(m)) for m in metric_names}
        per_type = {
            t: {m: _collect(lambda r, t=t, m=m: r.get("per_type", {}).get(t, {}).get(m)) for m in metric_names}
            for t in tuple_types
        }

        agg[variant] = {
            "skipped": False,
            "n_folds": len(live),
            "global": global_block,
            "per_type": per_type,
            "n_gold_tuples": sum(r.get("n_gold_tuples", 0) for r in live),
            "n_pred_tuples": sum(r.get("n_pred_tuples", 0) for r in live),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict = {}
    if output_path.exists():
        try:
            merged = json.loads(output_path.read_text())
        except Exception:
            merged = {}
    merged.update(agg)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    logger.info("Aggregated tuple CV results -> %s (%d variants total).", output_path, len(merged))

def _train_cmd(annotations_dir: Path, ckpt_dir: Path, args, train_list: Path | None = None, balancing: bool = False) -> list[str]:
    cmd = [
        _PYTHON, str(_HERE / "train_all.py"),
        "--annotations_dir", str(annotations_dir),
        "--output_dir", str(ckpt_dir),
        "--epochs_hgt", str(args.epochs_hgt),
        "--epochs_mlp", str(args.epochs_mlp),
    ]
    if train_list is not None:
        cmd += ["--train_list", str(train_list)]
    if args.hgt_features:
        cmd += ["--features", args.hgt_features]
    if balancing:
        cmd.append("--balancing")
    if args.force:
        cmd.append("--force")
    return cmd

def _eval_cmd(raw_dir: Path, annotations_dir: Path, ckpt_dir: Path, results_path: Path, args, eval_list: Path | None = None) -> list[str]:
    cmd = [
        _PYTHON, str(_HERE / "evaluate_all.py"),
        "--raw_dir", str(raw_dir),
        "--annotations_dir", str(annotations_dir),
        "--config_path", str(args.config_path),
        "--manifest_path", str(ckpt_dir / "manifest.json"),
        "--output_path", str(results_path),
    ]
    if eval_list is not None:
        cmd += ["--eval_list", str(eval_list)]
    if args.variants:
        cmd += ["--variants", *args.variants]
    if args.force:
        cmd.append("--force")
    return cmd

def _tuple_eval_cmd(annotations_dir: Path, ckpt_dir: Path, results_path: Path, args, eval_list: Path | None = None) -> list[str]:
    cmd = [
        _PYTHON, str(_HERE / "evaluate_tuples.py"),
        "--annotations_dir", str(annotations_dir),
        "--manifest_path", str(ckpt_dir / "manifest.json"),
        "--output_path", str(results_path),
    ]
    if eval_list is not None:
        cmd += ["--eval_list", str(eval_list)]
    if args.variants:
        cmd += ["--variants", *args.variants]
    if args.force:
        cmd.append("--force")
    return cmd

def run_cv_for_dataset(annotations_dir: Path, raw_dir: Path, output_root: Path, results_path: Path, args,
                       tuple_results_path: Path | None = None) -> None:
    """Run matched k-fold CV (train+eval on the same dataset) and aggregate."""
    output_root.mkdir(parents=True, exist_ok=True)

    ann_paths = sorted(annotations_dir.glob("*.json"))
    folds = make_folds(ann_paths, n_splits=args.folds, seed=args.seed)
    logger.info("Built %d folds over %d graphs (seed=%d).", len(folds), len(ann_paths), args.seed)

    fold_results: list[dict] = []
    tuple_fold_results: list[dict] = []

    for k, (train_paths, test_paths) in enumerate(folds):
        fold_dir = output_root / f"fold_{k}"
        ckpt_dir = fold_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        train_list = fold_dir / "train_list.txt"
        eval_list = fold_dir / "eval_list.txt"
        _write_list(train_list, train_paths)
        _write_list(eval_list, test_paths)
        fold_results_path = fold_dir / "results.json"

        logger.info("=" * 64)
        logger.info("FOLD %d/%d  train=%d  test=%d", k + 1, len(folds), len(train_paths), len(test_paths))
        logger.info("=" * 64)

        if not args.skip_training:
            _run(_train_cmd(annotations_dir, ckpt_dir, args, train_list=train_list, balancing=args.balancing))

        if not args.skip_evaluation:
            _run(_eval_cmd(raw_dir, annotations_dir, ckpt_dir, fold_results_path, args, eval_list=eval_list))

        if fold_results_path.exists():
            with open(fold_results_path) as f:
                fold_results.append(json.load(f))

        if args.tuples:
            tuple_fold_path = fold_dir / "tuple_results.json"
            if not args.skip_evaluation:
                _run(_tuple_eval_cmd(annotations_dir, ckpt_dir, tuple_fold_path, args, eval_list=eval_list))
            if tuple_fold_path.exists():
                with open(tuple_fold_path) as f:
                    tuple_fold_results.append(json.load(f))

    if fold_results:
        aggregate(fold_results, results_path)
    else:
        logger.warning("No fold results found - nothing to aggregate.")

    if args.tuples and tuple_fold_results and tuple_results_path is not None:
        aggregate_tuples(tuple_fold_results, tuple_results_path)
    elif args.tuples and not tuple_fold_results:
        logger.warning("No per-fold tuple results found - nothing to aggregate.")

def run_generalization(gen_root: Path, results_dir: Path, args) -> None:
    """Train on the full train dataset and evaluate on the full *other* dataset."""
    gen_datasets = getattr(args, "gen_datasets", None) or GENERALIZATION_DATASETS
    for train_ds in gen_datasets:
        train_ann, _train_raw = resolve_dataset(train_ds)
        ckpt_dir = gen_root / train_ds / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 64)
        logger.info("GENERALIZATION - training on full %s", train_ds)
        logger.info("=" * 64)
        if not args.skip_training:
            _run(_train_cmd(train_ann, ckpt_dir, args, train_list=None))

        for eval_ds in gen_datasets:
            if eval_ds == train_ds:
                continue
            eval_ann, eval_raw = resolve_dataset(eval_ds)
            results_path = results_dir / f"gen_{train_ds.removesuffix('_cleaned')}__{eval_ds.removesuffix('_cleaned')}.json"
            logger.info("GENERALIZATION - eval %s-trained model on full %s -> %s", train_ds, eval_ds, results_path.name)
            if not args.skip_evaluation:
                _run(_eval_cmd(eval_raw, eval_ann, ckpt_dir, results_path, args, eval_list=None))

def main() -> None:
    parser = argparse.ArgumentParser(description="k-fold CV driver for the SystemX ablation study")
    parser.add_argument("--datasets", nargs="*", default=["jetbrains", "github", "combined"],
                        help="Datasets for matched train+eval CV (default: all three). Use the *_cleaned "
                             "variants to run on the cleaned gold, e.g. combined_cleaned jetbrains_cleaned github_cleaned.")
    parser.add_argument("--gen_datasets", nargs="*", default=None,
                        help="Datasets for the cross-dataset generalization pass (default: jetbrains github). "
                             "For a cleaned run pass e.g. jetbrains_cleaned github_cleaned.")
    parser.add_argument("--generalization", dest="generalization", action="store_true", default=True,
                        help="Also run cross-dataset train->eval generalization (default: on).")
    parser.add_argument("--no-generalization", dest="generalization", action="store_false")
    parser.add_argument("--balancing", action="store_true", help="Also train+eval the class-imbalance ablation variants (under-sampling / class-weighting) on each matched dataset.")
    parser.add_argument("--tuples", dest="tuples", action="store_true", default=True,
                        help="Also run the lineage-tuple extraction eval per fold and aggregate to tuple_results_<dataset>.json (default: on).")
    parser.add_argument("--no-tuples", dest="tuples", action="store_false")
    parser.add_argument("--config_path", default="systemx-jupyter/models/config/default_reversed.yaml")
    parser.add_argument("--output_root", default="output/cv", help="Per-dataset CV outputs go to <output_root>/<dataset>/.")
    parser.add_argument("--gen_root", default="output/gen", help="Generalization checkpoints go to <gen_root>/<train_ds>/.")
    parser.add_argument("--results_dir", default="output/results", help="Aggregated results: cv_results_<dataset>.json and gen_<a>__<b>.json.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs_hgt", type=int, default=80)
    parser.add_argument("--epochs_mlp", type=int, default=50)
    parser.add_argument("--hgt_features", default=None, help="Comma-separated HGT presets to train (e.g. 'standard' for a fast headline-only run). Default: all.")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    args = parser.parse_args()

    if args.variants is None:
        args.variants = [v.value for v in AblationVariant if not v.value.startswith("llm")]
        logger.info("No --variants given: running %d non-LLM variants (LLMs come from the sample splice).", len(args.variants))

    output_root = Path(args.output_root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        annotations_dir, raw_dir = resolve_dataset(dataset)
        ds_name = dataset.removesuffix("_cleaned")
        results_path = results_dir / f"cv_results_{ds_name}.json"
        tuple_results_path = results_dir / f"tuple_results_{ds_name}.json" if args.tuples else None
        logger.info("#" * 64)
        logger.info("DATASET %s  (matched k-fold CV)  ann=%s", dataset, annotations_dir)
        logger.info("#" * 64)
        run_cv_for_dataset(annotations_dir, raw_dir, output_root / dataset, results_path, args,
                           tuple_results_path=tuple_results_path)

    if args.generalization:
        run_generalization(Path(args.gen_root), results_dir, args)

if __name__ == "__main__":
    main()
