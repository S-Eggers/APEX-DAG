#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import logging
import warnings
from typing import Any

import networkx as nx

from SystemX.experiment.ablation.evaluate_all import (
    VARIANT_CHECKPOINT_KEY,
    _GraphWrapper,
    _build_labeler_and_refiner,
)
from SystemX.experiment.evaluation.ablation_variant import (
    CONFIG_REGISTRY,
    AblationVariant,
    LabelerType,
)
from SystemX.experiment.evaluation.metrics import ConfusionMatrix
from SystemX.nn.training.v2.data_utils import annotation_to_networkx
from SystemX.sca.lineage_projector import BipartiteLineageProjector
from SystemX.sca.refinement.factory import create_default_refiner
from SystemX.serializer.tuple_utils import extract_tuples_from_projected
from SystemX.util.logger import configure_systemx_logger

configure_systemx_logger()
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=SyntaxWarning)

TUPLE_TYPES = ("<D, D>", "<M, D>", "<D, Empty>")

Tuple = tuple[str, str, str]

VAMSA_PAPER_VARIANT = "vamsa_paper_baseline"

def _strip_labels(g: nx.MultiDiGraph) -> None:
    for nid in g.nodes:
        g.nodes[nid].pop("domain_label", None)
        g.nodes[nid].pop("predicted_label", None)

def _project_tuples(g: nx.MultiDiGraph, refiner) -> set[Tuple]:
    """Run refiner -> projector -> tuple extraction on a labeled graph copy."""
    wrapper = _GraphWrapper(g)
    try:
        refiner.refine(wrapper)
    except Exception as exc:
        logger.debug("Refiner failed during tuple projection: %s", exc)
    macro = BipartiteLineageProjector().project(g)
    return {(t["tuple_type"], str(t["subject_id"]), str(t["object_id"])) for t in extract_tuples_from_projected(macro)}

def _gold_tuples(base: nx.MultiDiGraph, gold_labels: dict[str, int]) -> set[Tuple]:
    """Gold tuples: project the gold-labeled graph through the standard refiner."""
    g = base.copy()
    _strip_labels(g)
    nx.set_node_attributes(g, gold_labels, name="predicted_label")
    return _project_tuples(g, create_default_refiner())

def _pred_tuples(base: nx.MultiDiGraph, labeler, refiner) -> set[Tuple]:
    """Predicted tuples: label with the variant's labeler, then its own refiner."""
    g = base.copy()
    _strip_labels(g)
    labeler.apply_labels(_GraphWrapper(g))
    return _project_tuples(g, refiner)

class _TupleAccumulator:
    """Accumulates exact tuple-set precision/recall/F1, global and per type."""

    def __init__(self) -> None:
        self.global_cm = ConfusionMatrix()
        self.type_cms: dict[str, ConfusionMatrix] = {t: ConfusionMatrix() for t in TUPLE_TYPES}
        self.n_graphs = 0
        self.n_gold = 0
        self.n_pred = 0
        self.failures = 0

    def update(self, pred: set[Tuple], gold: set[Tuple]) -> None:
        self.n_graphs += 1
        self.n_gold += len(gold)
        self.n_pred += len(pred)

        tp = pred & gold
        fp = pred - gold
        fn = gold - pred

        self.global_cm.tp += len(tp)
        self.global_cm.fp += len(fp)
        self.global_cm.fn += len(fn)

        for t in TUPLE_TYPES:
            cm = self.type_cms[t]
            cm.tp += sum(1 for x in tp if x[0] == t)
            cm.fp += sum(1 for x in fp if x[0] == t)
            cm.fn += sum(1 for x in fn if x[0] == t)

    @staticmethod
    def _cm_dict(cm: ConfusionMatrix) -> dict[str, Any]:
        return {
            "precision": cm.precision,
            "recall": cm.recall,
            "f1": cm.f1_score,
            "tp": cm.tp,
            "fp": cm.fp,
            "fn": cm.fn,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "global": self._cm_dict(self.global_cm),
            "per_type": {t: self._cm_dict(cm) for t, cm in self.type_cms.items()},
            "n_graphs": self.n_graphs,
            "n_gold_tuples": self.n_gold,
            "n_pred_tuples": self.n_pred,
            "failures": self.failures,
            "skipped": False,
            "skip_reason": "",
        }

def evaluate_variant_tuples(
    variant: AblationVariant,
    v2_checkpoint_path: str | None,
    eval_paths: list[Path],
    gold_cache: dict[Path, set[Tuple]],
) -> dict[str, Any]:
    """Score one variant's tuple extraction against the gold tuples."""
    try:
        labeler, refiner = _build_labeler_and_refiner(variant, v2_checkpoint_path)
    except Exception as exc:
        logger.error("Labeler/refiner build failed for %s: %s", variant, exc)
        return {
            "global": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0},
            "per_type": {},
            "n_graphs": 0,
            "n_gold_tuples": 0,
            "n_pred_tuples": 0,
            "failures": 0,
            "skipped": True,
            "skip_reason": f"Build error: {exc}",
        }

    acc = _TupleAccumulator()

    for json_path in sorted(eval_paths):
        try:
            with open(json_path, encoding="utf-8") as f:
                raw = json.load(f)
            elements = raw if isinstance(raw, list) else raw.get("elements", [])
            base = annotation_to_networkx(elements)
            if base.number_of_nodes() == 0:
                continue

            gold_labels = _GraphWrapper(base).golden_labels()
            if not gold_labels:
                continue

            if json_path not in gold_cache:
                gold_cache[json_path] = _gold_tuples(base, gold_labels)
            gold = gold_cache[json_path]

            pred = _pred_tuples(base, labeler, refiner)
            acc.update(pred, gold)
        except Exception as exc:
            logger.warning("Tuple eval failed %s [%s]: %s", json_path.name, variant, exc)
            acc.failures += 1

    logger.info(
        "[%s] graphs=%d  gold=%d  pred=%d  P=%.4f  R=%.4f  F1=%.4f  failures=%d",
        variant,
        acc.n_graphs,
        acc.n_gold,
        acc.n_pred,
        acc.global_cm.precision,
        acc.global_cm.recall,
        acc.global_cm.f1_score,
        acc.failures,
    )
    return acc.to_dict()

def evaluate_vamsa_paper_tuples(
    eval_paths: list[Path],
    gold_cache: dict[Path, set[Tuple]],
    notebooks_dir: Path,
) -> dict[str, Any]:
    """Score the faithful (paper) Vamsa baseline against the gold tuples."""
    from SystemX.pipeline._shared import VAMSA_KB_PATH
    from SystemX.vamsa.experiment.paper_baseline import extract_paper_tuples

    acc = _TupleAccumulator()

    for json_path in sorted(eval_paths):
        try:
            with open(json_path, encoding="utf-8") as f:
                raw = json.load(f)
            elements = raw if isinstance(raw, list) else raw.get("elements", [])
            base = annotation_to_networkx(elements)
            if base.number_of_nodes() == 0:
                continue
            gold_labels = _GraphWrapper(base).golden_labels()
            if not gold_labels:
                continue
            if json_path not in gold_cache:
                gold_cache[json_path] = _gold_tuples(base, gold_labels)

            notebook_path = notebooks_dir / f"{json_path.stem}.ipynb"
            if not notebook_path.exists():
                logger.warning("No notebook for %s - counted as failure.", json_path.name)
                acc.failures += 1
                continue

            pred = extract_paper_tuples(elements, notebook_path, kb_csv_path=str(VAMSA_KB_PATH))
            acc.update(pred, gold_cache[json_path])
        except Exception as exc:
            logger.warning("Vamsa-paper tuple eval failed %s: %s", json_path.name, exc)
            acc.failures += 1

    logger.info(
        "[%s] graphs=%d  gold=%d  pred=%d  P=%.4f  R=%.4f  F1=%.4f  failures=%d",
        VAMSA_PAPER_VARIANT,
        acc.n_graphs,
        acc.n_gold,
        acc.n_pred,
        acc.global_cm.precision,
        acc.global_cm.recall,
        acc.global_cm.f1_score,
        acc.failures,
    )
    return acc.to_dict()

def _resolve_checkpoint(
    variant: AblationVariant, manifest: dict[str, str]
) -> tuple[str | None, str | None]:
    """Return (checkpoint_path, skip_reason)."""
    cfg = CONFIG_REGISTRY[variant]
    if cfg.labeler_type in (LabelerType.VAMSA_STATIC, LabelerType.LLM):
        return None, None
    if cfg.labeler_type == LabelerType.V1_GAT:
        v1_path = Path("checkpoints/model_epoch_pretrained_REVERSED_10.pt")
        if not v1_path.exists():
            return None, f"V1 GAT checkpoint not found: {v1_path}"
        return None, None

    ckpt_key = VARIANT_CHECKPOINT_KEY.get(variant)
    if ckpt_key is None or ckpt_key not in manifest:
        return None, f"Checkpoint key {ckpt_key!r} not in manifest"
    ckpt = manifest[ckpt_key]
    if not Path(ckpt).exists():
        return None, f"Checkpoint file missing: {ckpt}"
    return ckpt, None

def _skipped_result(reason: str) -> dict[str, Any]:
    return {
        "global": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0},
        "per_type": {},
        "n_graphs": 0,
        "n_gold_tuples": 0,
        "n_pred_tuples": 0,
        "failures": 0,
        "skipped": True,
        "skip_reason": reason,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lineage-tuple extraction for all ablation variants")
    parser.add_argument("--annotations_dir", default="data/combined_dataset/annotations")
    parser.add_argument("--manifest_path", default="output/checkpoints/manifest.json",
                        help="JSON manifest produced by train_all.py")
    parser.add_argument("--output_path", default="output/results/tuple_results_combined.json")
    parser.add_argument(
        "--eval_list",
        default=None,
        help="Optional file listing a fold's held-out test annotation filenames (one per line). "
        "Default: every annotation in --annotations_dir (the standalone / hand-labelled case).",
    )
    parser.add_argument("--variants", nargs="*", choices=[v.value for v in AblationVariant] + [VAMSA_PAPER_VARIANT], default=None,
                        help="Subset of variants to evaluate (default: all)")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if results exist")
    args = parser.parse_args()

    annotations_dir = Path(args.annotations_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.eval_list and Path(args.eval_list).exists():
        names = [ln.strip() for ln in Path(args.eval_list).read_text().splitlines() if ln.strip()]
        eval_paths = [annotations_dir / n for n in names]
        logger.info("Evaluating tuple extraction on %d held-out graphs from %s.", len(eval_paths), args.eval_list)
    else:
        eval_paths = sorted(annotations_dir.glob("*.json"))
        logger.info("Evaluating tuple extraction on all %d annotation graphs from %s.", len(eval_paths), annotations_dir)

    manifest: dict[str, str] = {}
    manifest_path = Path(args.manifest_path)
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        logger.info("Loaded manifest: %d checkpoints.", len(manifest))
    else:
        logger.warning("Manifest not found at %s - only Vamsa baseline can run.", manifest_path)

    results: dict[str, Any] = {}
    if output_path.exists() and not args.force:
        with open(output_path) as f:
            results = json.load(f)
        logger.info("Loaded %d existing results from %s.", len(results), output_path)

    variants_to_run = args.variants if args.variants else [v.value for v in AblationVariant] + [VAMSA_PAPER_VARIANT]

    gold_cache: dict[Path, set[Tuple]] = {}

    for vname in variants_to_run:
        if not args.force and vname in results and not results[vname].get("skipped"):
            logger.info("Skipping %s - results already present.", vname)
            continue

        logger.info("=" * 60)
        logger.info("Evaluating tuple extraction for variant: %s", vname)
        logger.info("=" * 60)

        if vname == VAMSA_PAPER_VARIANT:
            results[vname] = evaluate_vamsa_paper_tuples(eval_paths, gold_cache, annotations_dir.parent / "notebooks")
        else:
            variant = AblationVariant(vname)
            ckpt, skip_reason = _resolve_checkpoint(variant, manifest)
            if skip_reason is not None:
                logger.warning("Skipping %s - %s", vname, skip_reason)
                results[vname] = _skipped_result(skip_reason)
            else:
                results[vname] = evaluate_variant_tuples(variant, ckpt, eval_paths, gold_cache)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved results -> %s", output_path)

    logger.info("All done. Tuple results in %s", output_path)

if __name__ == "__main__":
    main()
