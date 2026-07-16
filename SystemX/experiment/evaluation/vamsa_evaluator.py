from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection

from SystemX.sca.constants import DOMAIN_EDGES

from .metrics import ConfusionMatrix

EdgeStruct = tuple[str, str]
EdgeLabeled = tuple[str, str, int]

class VamsaSemanticEvaluator:
    """Tracks global/class-wise semantic metrics and Vamsa node alignment yields."""

    def __init__(self) -> None:
        self.global_metrics = ConfusionMatrix()
        self.class_metrics: dict[int, ConfusionMatrix] = defaultdict(ConfusionMatrix)

        self.structural_tp: int = 0
        self.total_golden_struct: int = 0

        self.total_vamsa_nodes: int = 0
        self.total_mapped_nodes: int = 0
        self.failures: int = 0
        self.total_notebooks: int = 0

    def record_success(self, vamsa_nodes_count: int, mapped_nodes_count: int) -> None:
        self.total_notebooks += 1
        self.total_vamsa_nodes += vamsa_nodes_count
        self.total_mapped_nodes += mapped_nodes_count

    def record_failure(self, golden_edges: Collection[EdgeLabeled]) -> None:
        self.total_notebooks += 1
        self.failures += 1
        self.global_metrics.fn += len(golden_edges)

        for _, _, label in golden_edges:
            self.class_metrics[label].fn += 1

    def update(self, pred_labeled: set[EdgeLabeled], golden_labeled: set[EdgeLabeled], pred_struct: set[EdgeStruct], golden_struct: set[EdgeStruct]) -> None:

        self.global_metrics.tp += len(pred_labeled.intersection(golden_labeled))
        self.global_metrics.fp += len(pred_labeled - golden_labeled)
        self.global_metrics.fn += len(golden_labeled - pred_labeled)

        self.total_golden_struct += len(golden_struct)
        self.structural_tp += len(pred_struct.intersection(golden_struct))

        all_observed_classes = {e[2] for e in pred_labeled}.union({e[2] for e in golden_labeled})

        for cls in all_observed_classes:
            pred_c = {e for e in pred_labeled if e[2] == cls}
            gold_c = {e for e in golden_labeled if e[2] == cls}

            self.class_metrics[cls].tp += len(pred_c.intersection(gold_c))
            self.class_metrics[cls].fp += len(pred_c - gold_c)
            self.class_metrics[cls].fn += len(gold_c - pred_c)

    def report(self) -> None:
        node_yield = self.total_mapped_nodes / self.total_vamsa_nodes if self.total_vamsa_nodes > 0 else 0.0
        struct_recall = self.structural_tp / self.total_golden_struct if self.total_golden_struct > 0 else 0.0

        print(f"\n{'=' * 65}")
        print("VAMSA SEMANTIC BASELINE EXPERIMENT RESULTS")
        print(f"{'=' * 65}")
        print(f"Total Notebooks Evaluated:   {self.total_notebooks}")
        print(f"Pipeline Failures (Crashes): {self.failures}")
        print(f"Node Alignment Yield:        {node_yield:.2%} ({self.total_mapped_nodes}/{self.total_vamsa_nodes})")
        print(f"Structural Recall:           {struct_recall:.4f}")
        print(f"{'-' * 65}")
        print(f"Global Semantic Precision:   {self.global_metrics.precision:.4f}")
        print(f"Global Semantic Recall:      {self.global_metrics.recall:.4f}")
        print(f"Global Semantic F1-Score:    {self.global_metrics.f1_score:.4f}")
        print(f"{'-' * 65}")
        print(f"{'Class Name':<30} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'Support'}")
        print(f"{'-' * 65}")

        uid_to_name = {uid: data["label"] for uid, data in DOMAIN_EDGES.items()}

        for cls_id, metrics in sorted(self.class_metrics.items()):
            class_name = uid_to_name.get(cls_id, f"UNKNOWN_{cls_id}")
            support = metrics.tp + metrics.fn
            print(f"{class_name:<30} | {metrics.precision:<6.4f} | {metrics.recall:<6.4f} | {metrics.f1_score:<6.4f} | {support}")
        print(f"{'=' * 65}\n")
