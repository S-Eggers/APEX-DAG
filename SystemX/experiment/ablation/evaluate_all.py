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
import statistics
import time
import warnings
from collections import defaultdict
from typing import Any

import nbformat
import networkx as nx
import torch

from SystemX.experiment.evaluation.ablation_variant import (
    CONFIG_REGISTRY,
    AblationVariant,
    LabelerType,
)
from SystemX.experiment.evaluation.metrics import ConfusionMatrix
from SystemX.labeler.hgt_labeler import HGTLabeler
from SystemX.nn.data.v2.fasttext_embedding import FastTextEmbeddingV2
from SystemX.nn.data.v2.pruner import NullPruner
from SystemX.nn.data.v2.tensor_encoder import EncoderV2
from SystemX.nn.models.v2.gat import SystemXHeteroGraphTransformer
from SystemX.nn.training.v2.data_utils import annotation_to_networkx
from SystemX.pipeline.hgt_lineage_pipeline_factory import HGTLineagePipelineFactory
from SystemX.pipeline.mlp_lineage_pipeline_factory import MLPLineagePipelineFactory
from SystemX.pipeline.vamsa_lineage_pipeline_factory import VamsaLineagePipelineFactory
from SystemX.pipeline.xgboost_lineage_pipeline_factory import XGBoostLineagePipelineFactory
from SystemX.sca.constants import COMPUTE_HUBS, DOMAIN_EDGE_TYPES, DOMAIN_EDGES
from SystemX.sca.refinement.factory import create_empty_refiner
from SystemX.serializer.lossless_lineage_serializer import LosslessLineageSerializer
from SystemX.util.logger import configure_systemx_logger

configure_systemx_logger()
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=SyntaxWarning)

VARIANT_CHECKPOINT_KEY: dict[AblationVariant, str | None] = {
    AblationVariant.STANDARD: "hgt_standard",
    AblationVariant.SEEDING_REFINER: "hgt_api24",
    AblationVariant.EMPTY_REFINER: "hgt_api24",
    AblationVariant.NO_PROPAGATION: "hgt_api24",
    AblationVariant.NO_RESOLUTION: "hgt_api24",
    AblationVariant.NO_ANALYSIS: "hgt_api24",
    AblationVariant.EXTRACTION_AND_PROPAGATION: "hgt_api24",
    AblationVariant.WITH_SUBGRAPH_HIGHLIGHT: "hgt_standard",
    AblationVariant.VAMSA_STATIC_BASELINE: None,
    AblationVariant.LLM_BASELINE: None,
    AblationVariant.LLM_GEMINI_A: None,
    AblationVariant.LLM_GEMINI_B: None,
    AblationVariant.LLM_GEMMA_31B: None,
    AblationVariant.LLM_GEMMA_26B: None,
    AblationVariant.LLM_GEMINI_CODE: None,
    AblationVariant.LLM_GEMINI_CODE_RICH: None,
    AblationVariant.LLM_GEMINI_STRONG: None,
    AblationVariant.LLM_GEMMA2_2B_LOCAL: None,
    AblationVariant.LLM_GEMMA2_2B_LOCAL_STRONG: None,
    AblationVariant.LLM_GEMMA2_9B_LOCAL: None,
    AblationVariant.LLM_GEMMA2_9B_LOCAL_STRONG: None,
    AblationVariant.LLM_MISTRAL_7B_LOCAL: None,
    AblationVariant.LLM_MISTRAL_7B_LOCAL_STRONG: None,
    AblationVariant.LLM_GEMMA4_4B_LOCAL: None,
    AblationVariant.LLM_GEMMA4_4B_LOCAL_STRONG: None,
    AblationVariant.LLM_GEMMA4_12B_LOCAL: None,
    AblationVariant.LLM_GEMMA4_12B_LOCAL_STRONG: None,
    AblationVariant.LLM_QWEN3_4B_LOCAL: None,
    AblationVariant.LLM_QWEN3_4B_LOCAL_STRONG: None,
    AblationVariant.LLM_QWEN25_CODER_7B_LOCAL: None,
    AblationVariant.LLM_QWEN25_CODER_7B_LOCAL_STRONG: None,
    AblationVariant.V1_GAT_BASELINE: "v1_gat",
    AblationVariant.V2_CODEBERT: None,
    AblationVariant.V2_GRAPHCODEBERT: None,
    AblationVariant.V2_MLP: "mlp_standard",
    AblationVariant.V2_XGBOOST: "xgboost_standard",
    AblationVariant.V2_HGT_FEAT_EMB_ONLY: "hgt_emb_only",
    AblationVariant.V2_HGT_FEAT_ALL: "hgt_all",
    AblationVariant.V2_HGT_FEAT_API_LIB: "hgt_api_lib",
    AblationVariant.V2_HGT_FEAT_API29: "hgt_api29",
    AblationVariant.V2_HGT_FEAT_API24: "hgt_api24",
    AblationVariant.V2_HGT_FEAT_STRUCT_ONLY: "hgt_struct_only",
    AblationVariant.V2_MLP_FEAT_EMB_ONLY: "mlp_emb_only",
    AblationVariant.V2_MLP_FEAT_ALL: "mlp_all",
    AblationVariant.V2_MLP_FEAT_API_LIB: "mlp_api_lib",
    AblationVariant.V2_MLP_FEAT_API29: "mlp_api29",
    AblationVariant.V2_MLP_FEAT_API24: "mlp_api24",
    AblationVariant.V2_MLP_FEAT_STRUCT_ONLY: "mlp_struct_only",
    AblationVariant.V2_XGBOOST_FEAT_ALL: "xgboost_all",
    AblationVariant.V2_XGBOOST_FEAT_EMB_ONLY: "xgboost_emb_only",
    AblationVariant.V2_XGBOOST_FEAT_API_LIB: "xgboost_api_lib",
    AblationVariant.V2_XGBOOST_FEAT_API29: "xgboost_api29",
    AblationVariant.V2_XGBOOST_FEAT_API24: "xgboost_api24",
    AblationVariant.V2_XGBOOST_FEAT_STRUCT_ONLY: "xgboost_struct_only",
    AblationVariant.V2_XGBOOST_FEAT_EMB_RICH: "xgboost_emb_rich",
    AblationVariant.V2_MLP_UNDERSAMPLE: "mlp_undersample",
    AblationVariant.V2_MLP_CLASSWEIGHT: "mlp_classweight",
    AblationVariant.V2_XGBOOST_UNDERSAMPLE: "xgboost_undersample",
    AblationVariant.V2_XGBOOST_CLASSWEIGHT: "xgboost_classweight",
    AblationVariant.V2_HGT_CW_NONE: "hgt_cw_none",
    AblationVariant.V2_HGT_CW_BALANCED: "hgt_cw_balanced",
}

from SystemX.nn.data.v2.tensor_encoder import HETERO_METADATA as _HETERO_METADATA

_fasttext_embedding: FastTextEmbeddingV2 | None = None

_OP_NODE_TYPES = frozenset(COMPUTE_HUBS)

_LLM_TIMING_MAX = 8

def _get_fasttext_embedding() -> FastTextEmbeddingV2:
    global _fasttext_embedding
    if _fasttext_embedding is None:
        _fasttext_embedding = FastTextEmbeddingV2()
    return _fasttext_embedding

class _GraphWrapper:
    """Thin wrapper around a NetworkX graph that satisfies the interface expected by EdgeLabeler.apply_labels() and GraphRefiner.refine()."""

    def __init__(self, nx_g: nx.MultiDiGraph) -> None:
        self._G = nx_g

    def get_graph(self) -> nx.MultiDiGraph:
        return self._G

    def get_state(self) -> _GraphWrapper:
        return self

    def optimize(self) -> None:
        pass

    def set_domain_label(self, attrs: dict, name: str) -> None:
        nx.set_edge_attributes(self._G, attrs, name=name)

    def set_domain_node_label(self, attrs: dict, name: str) -> None:
        nx.set_node_attributes(self._G, attrs, name=name)

    def filter_relevant(self, lineage_mode: bool = False) -> None:
        pass

    def golden_labels(self) -> dict[str, int]:
        """Returns {node_id: label_int} for all CALL/LOOP nodes with a domain_label."""
        out = {}
        for node_id, data in self._G.nodes(data=True):
            if data.get("node_type") not in _OP_NODE_TYPES:
                continue
            lbl = data.get("domain_label")
            if lbl is None:
                continue
            if isinstance(lbl, int):
                if lbl >= 0:
                    out[node_id] = lbl
            else:
                int_label = DOMAIN_EDGE_TYPES.get(str(lbl).upper())
                if int_label is not None and int_label >= 0:
                    out[node_id] = int_label
        return out

    def predicted_labels(self) -> dict[str, int]:
        """Returns {node_id: predicted_label_int} for all CALL/LOOP nodes after labeling."""
        out = {}
        for node_id, data in self._G.nodes(data=True):
            if data.get("node_type") not in _OP_NODE_TYPES:
                continue
            lbl = data.get("predicted_label")
            if lbl is not None:
                out[node_id] = int(lbl)
        return out

def _build_labeler_and_refiner(
    variant: AblationVariant,
    v2_checkpoint_path: str | None,
    vamsa_kb_path: Path | None = None,
    vamsa_strict_provenance: bool = False,
):
    """Return (labeler, refiner) for a given variant."""
    cfg = CONFIG_REGISTRY[variant]

    if cfg.labeler_type == LabelerType.VAMSA_STATIC:
        from SystemX.labeler.vamsa_static_labeler import VamsaStaticLabeler
        from SystemX.labeling.vamsa_loader import IOSignatureMappingPolicy, VamsaKBLoader
        from SystemX.pipeline._shared import VAMSA_KB_PATH

        kb = VamsaKBLoader(IOSignatureMappingPolicy()).load_and_map(vamsa_kb_path or VAMSA_KB_PATH)
        labeler = VamsaStaticLabeler(kb, strict_provenance=vamsa_strict_provenance)

    elif cfg.labeler_type == LabelerType.LLM:
        if cfg.llm_config_path:
            from dotenv import load_dotenv

            from SystemX.labeler.lean_llm_labeler import LeanLLMLabeler
            from SystemX.llm.config import load_config
            from SystemX.llm.llm_policy import ExecutionPolicy
            from SystemX.llm.provider_factory import ProviderFactory
            from SystemX.llm.resilient_provider import ResilientProvider

            load_dotenv()
            llm_config = load_config(cfg.llm_config_path)
            provider = ResilientProvider(
                ProviderFactory.create(llm_config), max_retries=llm_config.retry_attempts
            )
            policy = ExecutionPolicy(max_tokens=llm_config.max_tokens, max_rpm=llm_config.max_rpm)
            if cfg.llm_strong:
                from SystemX.labeler.notebook_llm_labeler import NotebookLLMLabeler

                labeler = NotebookLLMLabeler(config=llm_config, provider=provider, policy=policy)
            elif cfg.llm_rich_context:
                from SystemX.labeler.context_llm_labeler import ContextLLMLabeler

                labeler = ContextLLMLabeler(config=llm_config, provider=provider, policy=policy)
            else:
                labeler = LeanLLMLabeler(config=llm_config, provider=provider, policy=policy)
        else:
            from SystemX.labeler.random_labeler import RandomLabeler

            labeler = RandomLabeler()

    elif cfg.labeler_type == LabelerType.V2_HGT_FASTTEXT:
        embedding = _get_fasttext_embedding()
        encoder = EncoderV2(embedding_model=embedding, pruner=NullPruner(), feature_groups=cfg.feature_groups)
        state = torch.load(v2_checkpoint_path, map_location="cpu", weights_only=True)
        model = SystemXHeteroGraphTransformer(
            hidden_channels=int(state.get("hidden_channels", 128)),
            out_classes=len(DOMAIN_EDGE_TYPES),
            num_heads=int(state.get("num_heads", 4)),
            num_layers=int(state.get("num_layers", 3)),
            metadata=_HETERO_METADATA,
        )
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()
        labeler = HGTLabeler(model=model, encoder=encoder)

    elif cfg.labeler_type == LabelerType.V2_MLP:
        from SystemX.labeler.mlp_labeler import MLPLabeler

        labeler = MLPLabeler.from_checkpoint(v2_checkpoint_path, feature_groups=cfg.feature_groups)

    elif cfg.labeler_type == LabelerType.V2_XGBOOST:
        from SystemX.labeler.xgboost_labeler import XGBoostLabeler

        labeler = XGBoostLabeler.from_checkpoint(v2_checkpoint_path, feature_groups=cfg.feature_groups)

    else:
        raise ValueError(f"Unsupported labeler type: {cfg.labeler_type}")

    return labeler, cfg.refiner

class _NodeAccumulator:
    """Accumulates per-node (CALL/LOOP) classification metrics directly from annotation graphs."""

    def __init__(self) -> None:
        self.global_cm = ConfusionMatrix()
        self.class_cms: dict[int, ConfusionMatrix] = defaultdict(ConfusionMatrix)
        self.failures = 0
        self.times: list[float] = []
        self.timing_per_nb: list[dict] = []
        self.n_leakage = 0
        self.n_dead_code = 0

    def update_nodes(self, predicted: dict[str, int], golden: dict[str, int]) -> None:
        """Compare per-node predictions against gold labels."""
        all_ids = set(predicted) | set(golden)
        for node_id in all_ids:
            pred_lbl = predicted.get(node_id)
            gold_lbl = golden.get(node_id)
            if gold_lbl is None:
                continue
            if pred_lbl is None:
                self.global_cm.fn += 1
                self.class_cms[gold_lbl].fn += 1
                continue
            if pred_lbl == gold_lbl:
                self.global_cm.tp += 1
                self.class_cms[gold_lbl].tp += 1
            else:
                self.global_cm.fp += 1
                self.global_cm.fn += 1
                self.class_cms[pred_lbl].fp += 1
                self.class_cms[gold_lbl].fn += 1

    def to_dict(self) -> dict[str, Any]:
        uid_to_name = {uid: data["label"] for uid, data in DOMAIN_EDGES.items()}
        per_class = {}
        for cls_id, cm in sorted(self.class_cms.items()):
            name = uid_to_name.get(cls_id, f"UNKNOWN_{cls_id}")
            per_class[name] = {
                "precision": cm.precision,
                "recall": cm.recall,
                "f1": cm.f1_score,
                "support": cm.tp + cm.fn,
                "tp": cm.tp,
                "fp": cm.fp,
                "fn": cm.fn,
            }

        supported_f1 = [cm.f1_score for cm in self.class_cms.values() if (cm.tp + cm.fn) > 0]
        macro_f1 = sum(supported_f1) / len(supported_f1) if supported_f1 else 0.0

        n = len(self.times)
        return {
            "global": {
                "precision": self.global_cm.precision,
                "recall": self.global_cm.recall,
                "f1": self.global_cm.f1_score,
                "macro_f1": macro_f1,
                "tp": self.global_cm.tp,
                "fp": self.global_cm.fp,
                "fn": self.global_cm.fn,
            },
            "per_class": per_class,
            "timing": {
                "mean_s": statistics.mean(self.times) if n > 0 else 0.0,
                "std_s": statistics.stdev(self.times) if n > 1 else 0.0,
                "n": n,
            },
            "timing_per_notebook": self.timing_per_nb,
            "refiner_annotations": {"leakage": self.n_leakage, "dead_code": self.n_dead_code},
            "failures": self.failures,
            "skipped": False,
            "skip_reason": "",
        }

def evaluate_variant(
    variant: AblationVariant,
    raw_dir: Path,
    annotations_dir: Path,
    v2_checkpoint_path: str | None,
    eval_paths: list[Path],
    vamsa_kb_path: Path | None = None,
    apply_refiner: bool = True,
    measure_timing: bool = True,
    vamsa_strict_provenance: bool = False,
    notebook_call_provenance: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Evaluate a variant by running labeler + refiner on the held-out eval_paths annotation graphs."""
    try:
        labeler, refiner = _build_labeler_and_refiner(variant, v2_checkpoint_path, vamsa_kb_path, vamsa_strict_provenance)
    except Exception as exc:
        logger.error("Labeler/refiner build failed for %s: %s", variant, exc)
        return {
            "global": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "per_class": {},
            "timing": {"mean_s": 0.0, "std_s": 0.0, "n": 0},
            "failures": 0,
            "skipped": True,
            "skip_reason": f"Build error: {exc}",
        }

    acc = _NodeAccumulator()
    acc_raw = _NodeAccumulator()
    cfg = CONFIG_REGISTRY[variant]
    eval_paths = sorted(eval_paths)

    for json_path in eval_paths:
        try:
            with open(json_path, encoding="utf-8") as f:
                raw = json.load(f)
            elements = raw if isinstance(raw, list) else raw.get("elements", [])
            nx_G = annotation_to_networkx(elements)
            if nx_G.number_of_nodes() == 0:
                continue

            if notebook_call_provenance:
                provenance_by_code = notebook_call_provenance.get(json_path.name, {})
                for _, node_data in nx_G.nodes(data=True):
                    library = provenance_by_code.get(str(node_data.get("code", "")).strip())
                    if library:
                        node_data["base_inputs"] = library

            wrapper = _GraphWrapper(nx_G)
            golden = wrapper.golden_labels()
            if not golden:
                continue

            for nid in list(nx_G.nodes):
                nx_G.nodes[nid].pop("domain_label", None)
                nx_G.nodes[nid].pop("predicted_label", None)

            labeler.apply_labels(wrapper)
            acc_raw.update_nodes(wrapper.predicted_labels(), golden)

            if apply_refiner:
                try:
                    refiner.refine(wrapper)
                except Exception as refine_exc:
                    logger.debug("Refiner skipped for %s: %s", json_path.name, refine_exc)

            predicted = wrapper.predicted_labels()
            acc.update_nodes(predicted, golden)

            if apply_refiner:
                for _, ndata in nx_G.nodes(data=True):
                    if ndata.get("has_leakage"):
                        acc.n_leakage += 1
                    if ndata.get("is_dead_code"):
                        acc.n_dead_code += 1

        except Exception as exc:
            logger.warning("Annotation eval failed %s [%s]: %s", json_path.name, variant, exc)
            acc.failures += 1

    if not measure_timing:
        return _finalize_variant_result(variant, eval_paths, acc, acc_raw, apply_refiner)

    try:
        if cfg.labeler_type == LabelerType.VAMSA_STATIC:
            timing_pipeline = VamsaLineagePipelineFactory.create({"vamsaStaticClassification": True})
            timing_pipeline.labeler = labeler
        elif cfg.labeler_type == LabelerType.V2_HGT_FASTTEXT:
            timing_pipeline = HGTLineagePipelineFactory.create({"hgtClassification": True}, labeler)
        elif cfg.labeler_type == LabelerType.V2_MLP:
            timing_pipeline = MLPLineagePipelineFactory.create({"mlpClassification": True}, labeler)
        elif cfg.labeler_type == LabelerType.V2_XGBOOST:
            timing_pipeline = XGBoostLineagePipelineFactory.create({"xgboostClassification": True}, labeler)
        elif cfg.labeler_type == LabelerType.LLM:
            from SystemX.parser.graph_parser import GraphParser
            from SystemX.pipeline.lineage_pipeline import LineagePipeline

            timing_pipeline = LineagePipeline(
                parser=GraphParser(enrich_provenance=True),
                labeler=labeler,
                refiner=refiner,
                serializer=LosslessLineageSerializer(),
                highlight_relevant=cfg.highlight_relevant,
            )
        else:
            timing_pipeline = None

        if timing_pipeline is not None:
            timing_pipeline.serializer = LosslessLineageSerializer()
            timing_pipeline.refiner = refiner if apply_refiner else create_empty_refiner()
            timing_pipeline.highlight_relevant = cfg.highlight_relevant

            timing_cap = _LLM_TIMING_MAX if cfg.labeler_type == LabelerType.LLM else 30
            for json_path in eval_paths[:timing_cap]:
                ipynb_path = raw_dir / json_path.with_suffix(".ipynb").name
                if not ipynb_path.exists():
                    continue
                try:
                    with open(ipynb_path, encoding="utf-8") as _fh:
                        nb = nbformat.read(_fh, as_version=4)
                    cells = [dict(c) for c in nb.cells if c.cell_type == "code"]
                    loc = sum(len(c.get("source", "").splitlines()) for c in cells)

                    n_nodes: int | None = None
                    try:
                        with open(json_path, encoding="utf-8") as _jf:
                            raw = json.load(_jf)
                        elements = raw if isinstance(raw, list) else raw.get("elements", [])
                        n_nodes = sum(1 for e in elements if "source" not in e.get("data", {}))
                    except Exception:
                        pass

                    t0 = time.perf_counter()
                    timing_pipeline.execute(cells)
                    elapsed = time.perf_counter() - t0

                    acc.times.append(elapsed)
                    entry: dict = {"time_s": elapsed, "loc": loc}
                    if n_nodes is not None:
                        entry["n_nodes"] = n_nodes
                    acc.timing_per_nb.append(entry)
                except Exception:
                    pass
    except Exception as exc:
        logger.warning("Timing pipeline failed for %s: %s", variant, exc)

    return _finalize_variant_result(variant, eval_paths, acc, acc_raw, apply_refiner)

def _finalize_variant_result(
    variant: AblationVariant,
    eval_paths: list[Path],
    acc: _NodeAccumulator,
    acc_raw: _NodeAccumulator,
    apply_refiner: bool,
) -> dict[str, Any]:
    logger.info(
        "[%s] eval_graphs=%d  failures=%d  output_f1=%.4f  raw_f1=%.4f  refiner=%s  timing_n=%d",
        variant,
        len(eval_paths),
        acc.failures,
        acc.global_cm.f1_score,
        acc_raw.global_cm.f1_score,
        "on" if apply_refiner else "off",
        len(acc.times),
    )
    result = acc.to_dict()
    raw_dict = acc_raw.to_dict()
    result["raw"] = {"global": raw_dict["global"], "per_class": raw_dict["per_class"]}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all SystemX ablation variants")
    parser.add_argument("--raw_dir", default="data/jetbrains_dataset/notebooks")
    parser.add_argument("--annotations_dir", default="data/jetbrains_dataset/annotations")
    parser.add_argument(
        "--config_path",
        default="systemx-jupyter/models/config/default_reversed.yaml",
        help="YAML config (only needed for V1 GAT baseline)",
    )
    parser.add_argument(
        "--manifest_path",
        default="output/checkpoints/manifest.json",
        help="JSON manifest produced by train_all.py",
    )
    parser.add_argument(
        "--output_path",
        default="output/results/eval_results.json",
    )
    parser.add_argument(
        "--eval_list",
        default=None,
        help="Optional file listing the fold's held-out test annotation filenames (one per line). Default: all annotations (in-sample).",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        choices=[v.value for v in AblationVariant],
        default=None,
        help="Subset of variants to evaluate (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if results exist")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    annotations_dir = Path(args.annotations_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.eval_list and Path(args.eval_list).exists():
        names = [ln.strip() for ln in Path(args.eval_list).read_text().splitlines() if ln.strip()]
        eval_paths = [annotations_dir / n for n in names]
        logger.info("Evaluating on %d held-out graphs from %s.", len(eval_paths), args.eval_list)
    else:
        eval_paths = sorted(annotations_dir.glob("*.json"))
        logger.info("Evaluating on all %d graphs (in-sample - no --eval_list given).", len(eval_paths))

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

    variants_to_run = [AblationVariant(v) for v in args.variants] if args.variants else list(AblationVariant)

    for variant in variants_to_run:
        vname = variant.value

        if not args.force and vname in results and not results[vname].get("skipped"):
            logger.info("Skipping %s - results already present.", vname)
            continue

        logger.info("=" * 60)
        logger.info("Evaluating variant: %s", vname)
        logger.info("=" * 60)

        ckpt_key = VARIANT_CHECKPOINT_KEY.get(variant)
        cfg = CONFIG_REGISTRY[variant]

        if cfg.labeler_type in (LabelerType.VAMSA_STATIC, LabelerType.LLM):
            v2_ckpt = None
        elif cfg.labeler_type == LabelerType.V1_GAT:
            v1_path = Path("checkpoints/model_epoch_pretrained_REVERSED_10.pt")
            if not v1_path.exists():
                logger.warning("V1 GAT checkpoint not found at %s - skipping.", v1_path)
                results[vname] = {
                    "global": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "per_class": {},
                    "structural_recall": 0.0,
                    "timing": {"mean_s": 0.0, "std_s": 0.0, "n": 0},
                    "failures": 0,
                    "skipped": True,
                    "skip_reason": f"V1 GAT checkpoint not found: {v1_path}",
                }
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                continue
            v2_ckpt = None
        elif ckpt_key is None or ckpt_key not in manifest:
            logger.warning("Checkpoint key %r not in manifest - skipping %s.", ckpt_key, vname)
            results[vname] = {
                "global": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "per_class": {},
                "structural_recall": 0.0,
                "timing": {"mean_s": 0.0, "std_s": 0.0, "n": 0},
                "failures": 0,
                "skipped": True,
                "skip_reason": f"Checkpoint key {ckpt_key!r} not in manifest",
            }
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            continue
        else:
            v2_ckpt = manifest[ckpt_key]
            if not Path(v2_ckpt).exists():
                logger.warning("Checkpoint file not found: %s - skipping %s.", v2_ckpt, vname)
                results[vname] = {
                    "global": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "per_class": {},
                    "structural_recall": 0.0,
                    "timing": {"mean_s": 0.0, "std_s": 0.0, "n": 0},
                    "failures": 0,
                    "skipped": True,
                    "skip_reason": f"Checkpoint file missing: {v2_ckpt}",
                }
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                continue

        result = evaluate_variant(
            variant=variant,
            raw_dir=raw_dir,
            annotations_dir=annotations_dir,
            v2_checkpoint_path=v2_ckpt,
            eval_paths=eval_paths,
        )
        results[vname] = result

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved results -> %s", output_path)

    logger.info("All done. Results in %s", output_path)

if __name__ == "__main__":
    main()
