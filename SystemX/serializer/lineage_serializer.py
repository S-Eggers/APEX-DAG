import networkx as nx
from SystemX.sca.lineage_projector import BipartiteLineageProjector, SerializedElement, SerializedElementData, SerializedLineage
from SystemX.sca.cdf_ir import CDFIntermediateRepresentation
from SystemX.sca.refinement.constants import callee_name
from SystemX.serializer.base import CytoscapeSerializerMixin
from SystemX.serializer.tuple_utils import extract_tuples_from_projected

class LineageSerializer(CytoscapeSerializerMixin):
    def __init__(self, projector: BipartiteLineageProjector | None = None) -> None:
        self._projector = projector or BipartiteLineageProjector()

    def to_dict(self, graph: CDFIntermediateRepresentation) -> SerializedLineage:
        lineage_graph = self._projector.project(graph.get_graph())
        result = self._format_json(lineage_graph)
        result["tuples"] = extract_tuples_from_projected(lineage_graph)
        return result

    def _format_json(self, macro_graph: nx.DiGraph) -> SerializedLineage:
        elements: list[SerializedElement] = []

        for n, data in macro_graph.nodes(data=True):
            raw_inputs = data.get("base_inputs", [])
            sorted_inputs = [item[1] for item in sorted(raw_inputs, key=lambda x: x[0])]

            node_defaults: SerializedElementData = {
                "label": data.get("label", str(n)),
                "node_type": 0,
                "transform_history": data.get("transform_history", []),
                "base_inputs": ", ".join(sorted_inputs),
            }
            elements.append(
                self._node_element(n, {k: v for k, v in data.items() if k not in ("id", "base_inputs", "transform_history")}, node_defaults)  # type: ignore[arg-type]
            )

        for u, v, data in macro_graph.edges(data=True):
            ops = data.get("operations", [])
            clean_ops = [str(o) for o in ops if o]

            full_chain = " -> ".join(clean_ops) if clean_ops else data.get("label", "Transform")
            display_ops = clean_ops[:-1] if len(clean_ops) > 1 else clean_ops
            op_methods = [m for m in (callee_name(o) for o in display_ops) if m]
            edge_lbl = " → ".join(op_methods) if op_methods else (display_ops[-1] if display_ops else full_chain)

            edge_defaults: SerializedElementData = {  # type: ignore[assignment]
                "id": f"edge_{u}_{v}",
                "edge_type": 2,
                "label": edge_lbl,
                "raw_code": full_chain,
                "operations": display_ops,
                "predicted_label": data.get("predicted_label", 2),
            }
            elements.append(
                self._edge_element(
                    u,
                    v,
                    {k: v for k, v in data.items() if k not in ("id", "operations", "predicted_label")},
                    edge_defaults,  # type: ignore[arg-type]
                    exclude=frozenset({"id"}),
                )
            )

        return {"elements": elements}
