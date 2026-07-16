from typing import ClassVar

import networkx as nx

from SystemX.mining.extract_tuples.domain import LineageTuple

from ..core.utils import remove_id
from .annotator import AnnotationWIR

class VamsaTupleMapper:
    """Adapter layer bridging Vamsa's Annotated Bipartite WIR to the target <M, D>, <D, D>, <D, Empty> tuple evaluation schema."""

    DATA_TAGS: ClassVar[set[str]] = {
        "data",
        "features",
        "labels",
        "validation features",
        "validation labels",
    }
    MODEL_TAGS: ClassVar[set[str]] = {"model", "trained model", "trained_model"}
    EXPORT_APIS: ClassVar[set[str]] = {"to_csv", "to_sql", "to_parquet", "dump", "savefig"}

    def __init__(self, annotated_wir: AnnotationWIR | nx.DiGraph) -> None:
        self.graph = getattr(annotated_wir, "annotated_wir", annotated_wir)

    def extract(self) -> list[LineageTuple]:
        extracted_tuples: set[LineageTuple] = set()

        data_nodes = set()
        for n, d in self.graph.nodes(data=True):
            anns = d.get("annotations", [])
            if any(tag in self.DATA_TAGS for tag in anns):
                data_nodes.add(n)

        intermediates = set()
        for dn in data_nodes:
            try:
                descendants = nx.descendants(self.graph, dn)
                intermediates.update(descendants.intersection(data_nodes))
            except nx.NetworkXError:
                pass

        true_roots = data_nodes - intermediates

        for root in true_roots:
            extracted_tuples.update(self._trace_vamsa_lineage(root))

        return list(extracted_tuples)

    def _trace_vamsa_lineage(self, root_node: str) -> set[LineageTuple]:
        local_tuples: set[LineageTuple] = set()
        has_terminal_state = False

        try:
            descendants = nx.descendants(self.graph, root_node)
        except nx.NetworkXError:
            descendants = set()

        operations = [n for n in descendants if self.graph.nodes[n].get("node_type") == 3]

        for op in operations:
            op_name = remove_id(op)

            caller = None
            for u, _, ed in self.graph.in_edges(op, data=True):
                if ed.get("edge_type") == 0:
                    caller = u
                    break

            is_model_op = False
            if caller:
                caller_anns = self.graph.nodes[caller].get("annotations", [])
                if any(tag in self.MODEL_TAGS for tag in caller_anns):
                    is_model_op = True
                    local_tuples.add(LineageTuple(tuple_type="<M, D>", subject_id=caller, object_id=root_node))
                    has_terminal_state = True

            if not is_model_op and op_name in ("fit", "predict", "predict_proba", "score"):
                subject = caller if caller else op
                local_tuples.add(LineageTuple(tuple_type="<M, D>", subject_id=subject, object_id=root_node))
                has_terminal_state = True

            if op_name in self.EXPORT_APIS:
                local_tuples.add(LineageTuple(tuple_type="<D, D>", subject_id=root_node, object_id=op))
                has_terminal_state = True

        if not has_terminal_state:
            local_tuples.add(LineageTuple(tuple_type="<D, Empty>", subject_id=root_node, object_id="Empty"))

        return local_tuples
