import argparse
import json
import logging
import statistics
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import nbformat
import torch
import yaml

from SystemX.nn.data.v2.fasttext_embedding import FastTextEmbeddingV2
from SystemX.nn.data.v2.feature_extractor import FeatureGroup
from SystemX.nn.data.v2.pruner import GraphPruner
from SystemX.nn.data.v2.tensor_encoder import EncoderV2
from SystemX.nn.models.v2.gat import SystemXHeteroGraphTransformer
from SystemX.pipeline.lineage_pipeline import LineagePipeline
from SystemX.pipeline.lineage_pipeline_factory import LineagePipelineFactory
from SystemX.sca.constants import DOMAIN_EDGE_TYPES, DOMAIN_EDGES
from SystemX.serializer.lossless_lineage_serializer import LosslessLineageSerializer
from SystemX.util.logger import configure_systemx_logger

from .ablation_variant import CONFIG_REGISTRY, AblationVariant, LabelerType
from .metrics import ConfusionMatrix

configure_systemx_logger()
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=SyntaxWarning, message=".*invalid.*")

from SystemX.nn.data.v2.tensor_encoder import HETERO_METADATA as _HETERO_METADATA

class SemanticEvaluator:
    """Tracks and calculates global and class-wise semantic graph metrics."""

    def __init__(self) -> None:
        self.global_metrics = ConfusionMatrix()
        self.class_metrics: dict[int, ConfusionMatrix] = defaultdict(ConfusionMatrix)
        self.structural_tp: int = 0
        self.total_golden_struct: int = 0
        self.failures: int = 0
        self.notebook_times: list[float] = []

    def record_failure(self, golden_edges: set[tuple[str, str, int]]) -> None:
        self.failures += 1
        self.global_metrics.fn += len(golden_edges)
        for _, _, label in golden_edges:
            self.class_metrics[label].fn += 1

    def update(
        self,
        pred_edges: set[tuple[str, str, int]],
        golden_edges: set[tuple[str, str, int]],
        pred_struct: set[tuple[str, str]],
        golden_struct: set[tuple[str, str]],
        elapsed: float = 0.0,
    ) -> None:
        self.global_metrics.tp += len(pred_edges.intersection(golden_edges))
        self.global_metrics.fp += len(pred_edges - golden_edges)
        self.global_metrics.fn += len(golden_edges - pred_edges)

        self.total_golden_struct += len(golden_struct)
        self.structural_tp += len(pred_struct.intersection(golden_struct))
        self.notebook_times.append(elapsed)

        all_observed = {e[2] for e in pred_edges}.union({e[2] for e in golden_edges})
        for cls in all_observed:
            pred_c = {e for e in pred_edges if e[2] == cls}
            gold_c = {e for e in golden_edges if e[2] == cls}
            self.class_metrics[cls].tp += len(pred_c.intersection(gold_c))
            self.class_metrics[cls].fp += len(pred_c - gold_c)
            self.class_metrics[cls].fn += len(gold_c - pred_c)

    def report(self, variant: AblationVariant) -> None:
        struct_recall = self.structural_tp / self.total_golden_struct if self.total_golden_struct > 0 else 0.0

        if self.notebook_times:
            t_mean = statistics.mean(self.notebook_times)
            t_std = statistics.stdev(self.notebook_times) if len(self.notebook_times) > 1 else 0.0
        else:
            t_mean = t_std = 0.0

        print(f"\n{'=' * 60}")
        print(f"SystemX SEMANTIC EVALUATION  [{variant.upper()}]")
        print(f"{'=' * 60}")
        print(f"Structural Recall (Parser Ceiling): {struct_recall:.4f}")
        print(f"Global Precision:                   {self.global_metrics.precision:.4f}")
        print(f"Global Recall:                      {self.global_metrics.recall:.4f}")
        print(f"Global F1-Score:                    {self.global_metrics.f1_score:.4f}")
        print(f"Failures (Crashes):                 {self.failures}")
        print(f"Timing  mean +/- std:               {t_mean:.3f}s +/- {t_std:.3f}s  (N={len(self.notebook_times)})")
        print(f"{'-' * 60}")
        print(f"{'Class Name':<30} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'Support'}")
        print(f"{'-' * 60}")

        uid_to_name = {uid: data["label"] for uid, data in DOMAIN_EDGES.items()}
        for cls_id, metrics in sorted(self.class_metrics.items()):
            class_name = uid_to_name.get(cls_id, f"UNKNOWN_{cls_id}")
            support = metrics.tp + metrics.fn
            print(f"{class_name:<30} | {metrics.precision:<6.4f} | {metrics.recall:<6.4f} | {metrics.f1_score:<6.4f} | {support}")
        print(f"{'=' * 60}\n")

def normalize_id(node_id: str | int) -> str:
    if not node_id:
        return ""
    return str(node_id).replace("legacy_f_", "fallback_").replace("legacy_fallback_", "fallback_")

def extract_semantic_edges(payload: dict[str, Any], normalize: bool = False) -> set[tuple[str, str, int]]:
    edges = set()
    for element in payload.get("elements", []):
        data = element.get("data", {})
        source, target = data.get("source"), data.get("target")
        label = data.get("predicted_label") if "predicted_label" in data else data.get("edge_type")
        if source is not None and target is not None and label is not None:
            s = normalize_id(source) if normalize else str(source)
            t = normalize_id(target) if normalize else str(target)
            edges.add((s, t, int(label)))
    return edges

def _build_hgt_labeler(checkpoint_path: str, labeler_type: LabelerType) -> object:
    """Build an HGTLabeler with the appropriate embedding."""
    from SystemX.labeler.hgt_labeler import HGTLabeler

    if labeler_type == LabelerType.V2_HGT_FASTTEXT:
        embedding = FastTextEmbeddingV2()
    elif labeler_type == LabelerType.V2_HGT_CODEBERT:
        from SystemX.nn.data.v2.embedding import TransformerEmbedding

        embedding = TransformerEmbedding(model_name="microsoft/codebert-base")
    elif labeler_type == LabelerType.V2_HGT_GRAPHCODEBERT:
        from SystemX.nn.data.v2.embedding import TransformerEmbedding

        embedding = TransformerEmbedding(model_name="microsoft/graphcodebert-base")
    else:
        raise ValueError(f"Not an HGT labeler type: {labeler_type}")

    encoder = EncoderV2(embedding_model=embedding, pruner=GraphPruner())
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SystemXHeteroGraphTransformer(
        hidden_channels=int(state.get("hidden_channels", 128)),
        out_classes=len(DOMAIN_EDGE_TYPES),
        num_heads=int(state.get("num_heads", 4)),
        num_layers=int(state.get("num_layers", 3)),
        metadata=_HETERO_METADATA,
    )
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    return HGTLabeler(model=model, encoder=encoder)

def build_pipeline(
    config: dict[str, Any],
    checkpoint_path: str,
    labeler_type: LabelerType = LabelerType.V2_HGT_FASTTEXT,
    v2_checkpoint_path: str | None = None,
    feature_groups: "FeatureGroup | None" = None,
) -> LineagePipeline:
    """Build the evaluation pipeline."""
    effective_v2_ckpt = v2_checkpoint_path or checkpoint_path

    if labeler_type in (LabelerType.V2_HGT_FASTTEXT, LabelerType.V2_HGT_CODEBERT, LabelerType.V2_HGT_GRAPHCODEBERT):
        labeler = _build_hgt_labeler(effective_v2_ckpt, labeler_type)
        pipeline = LineagePipelineFactory.create(
            request_payload={"llmClassification": False, "highlightRelevantSubgraphs": False},
            model=None,
        )
        pipeline.labeler = labeler

    elif labeler_type == LabelerType.V2_MLP:
        from SystemX.labeler.mlp_labeler import MLPLabeler
        from SystemX.nn.data.v2.feature_extractor import FeatureGroup

        fg = feature_groups if feature_groups is not None else FeatureGroup.STANDARD
        pipeline = LineagePipelineFactory.create(
            request_payload={"llmClassification": False, "highlightRelevantSubgraphs": False},
            model=None,
        )
        pipeline.labeler = MLPLabeler.from_checkpoint(effective_v2_ckpt, feature_groups=fg)

    elif labeler_type == LabelerType.V2_XGBOOST:
        from SystemX.labeler.xgboost_labeler import XGBoostLabeler
        from SystemX.nn.data.v2.feature_extractor import FeatureGroup

        fg = feature_groups if feature_groups is not None else FeatureGroup.STANDARD
        pipeline = LineagePipelineFactory.create(
            request_payload={"llmClassification": False, "highlightRelevantSubgraphs": False},
            model=None,
        )
        pipeline.labeler = XGBoostLabeler.from_checkpoint(effective_v2_ckpt, feature_groups=fg)

    elif labeler_type == LabelerType.V1_GAT:
        from SystemX.nn.data.v1.encoder import GraphEncoder
        from SystemX.nn.models.v1.gat import MultiTaskGATv1

        encoder = GraphEncoder(
            Path(config["encoded_checkpoint_path"]),
            min_nodes=config.get("min_nodes", 3),
            min_edges=config.get("min_edges", 2),
            load_encoded_old_if_exist=False,
            mode="REVERSED",
        )
        model = MultiTaskGATv1(
            hidden_dim=config["hidden_dim"],
            dim_embed=config["dim_embed"],
            num_heads=config["num_heads"],
            edge_classes=config["node_classes"],
            node_classes=config["edge_classes"],
            residual=config.get("residual", True),
            dropout=config.get("dropout", 0.0),
            number_gat_blocks=config.get("number_gat_blocks", 3),
            task=["node_classification"],
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        pipeline = LineagePipelineFactory.create(
            request_payload={"llmClassification": False, "highlightRelevantSubgraphs": False},
            model={"encoder": encoder, "model": model},
        )

    elif labeler_type == LabelerType.VAMSA_STATIC:
        from SystemX.labeler.vamsa_static_labeler import VamsaStaticLabeler
        from SystemX.labeling.vamsa_loader import IOSignatureMappingPolicy, VamsaKBLoader
        from SystemX.pipeline._shared import VAMSA_KB_PATH

        kb = VamsaKBLoader(IOSignatureMappingPolicy()).load_and_map(VAMSA_KB_PATH)
        pipeline = LineagePipelineFactory.create(
            request_payload={"llmClassification": False, "highlightRelevantSubgraphs": False},
            model=None,
        )
        pipeline.labeler = VamsaStaticLabeler(kb)

    else:
        raise ValueError(f"Unknown LabelerType: {labeler_type}")

    pipeline.serializer = LosslessLineageSerializer()
    return pipeline

def _apply_ablation_config(pipeline: LineagePipeline, variant: AblationVariant) -> None:
    cfg = CONFIG_REGISTRY[variant]
    pipeline.refiner = cfg.refiner
    pipeline.highlight_relevant = cfg.highlight_relevant

def main() -> None:
    parser = argparse.ArgumentParser(description="SystemX Semantic Evaluation (V2)")
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--annotations_dir", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_path", required=True, help="V1 GAT checkpoint (or V2 if --v2_checkpoint_path not given)")
    parser.add_argument("--v2_checkpoint_path", default=None, help="V2 HGT / MLP / XGBoost checkpoint")
    parser.add_argument(
        "--invariant",
        type=AblationVariant,
        choices=list(AblationVariant),
        default=AblationVariant.STANDARD,
    )
    args = parser.parse_args()

    with open(args.config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg = CONFIG_REGISTRY[args.invariant]
    pipeline = build_pipeline(
        config=config,
        checkpoint_path=args.checkpoint_path,
        labeler_type=cfg.labeler_type,
        v2_checkpoint_path=args.v2_checkpoint_path,
        feature_groups=cfg.feature_groups,
    )
    _apply_ablation_config(pipeline, args.invariant)

    evaluator = SemanticEvaluator()

    for json_path in Path(args.annotations_dir).glob("*.json"):
        ipynb_path = Path(args.raw_dir) / json_path.name.replace(".json", ".ipynb")
        if not ipynb_path.exists():
            continue

        with open(json_path, encoding="utf-8") as f:
            golden_json = json.load(f)

        golden_semantic = extract_semantic_edges(golden_json, normalize=False)
        golden_struct = {(str(e[0]), str(e[1])) for e in golden_semantic}

        with open(ipynb_path, encoding="utf-8") as f:
            notebook_data = nbformat.read(f, as_version=4)
        cells = [dict(c) for c in notebook_data.cells if c.cell_type == "code"]

        try:
            t0 = time.perf_counter()
            pred_payload = pipeline.execute(cells)
            elapsed = time.perf_counter() - t0

            pred_semantic = extract_semantic_edges(pred_payload, normalize=True)

            data_flow_graph = pipeline.parser.parse(cells)
            data_flow_graph.get_state().optimize()
            pred_struct = {(normalize_id(u), normalize_id(v)) for u, v in data_flow_graph.get_graph().edges()}

            evaluator.update(pred_semantic, golden_semantic, pred_struct, golden_struct, elapsed=elapsed)

        except Exception as e:
            logger.error("Failed %s: %s", ipynb_path.name, e)
            evaluator.record_failure(golden_semantic)

    evaluator.report(args.invariant)

if __name__ == "__main__":
    main()
