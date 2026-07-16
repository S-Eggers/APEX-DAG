from SystemX.sca.cdf_ir import CDFIntermediateRepresentation


class LosslessLineageSerializer:
    """Serializer: Bypasses path collapsing for raw semantic evaluation."""

    def to_dict(self, graph: CDFIntermediateRepresentation) -> dict:
        G = graph.get_graph()
        elements = []
        for n, data in G.nodes(data=True):
            elements.append({"data": {"id": str(n), "label": data.get("label", str(n)), "node_type": data.get("node_type", 0)}})
        for u, v, key, data in G.edges(keys=True, data=True):
            elements.append({"data": {"id": f"edge_{u}_{v}_{key}", "source": str(u), "target": str(v), "predicted_label": data.get("predicted_label", 0)}})
        return {"elements": elements}
