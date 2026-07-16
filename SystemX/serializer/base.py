class CytoscapeSerializerMixin:
    @staticmethod
    def _node_element(node_id: object, data: dict, defaults: dict) -> dict:
        payload = {"id": str(node_id), **defaults}
        payload.update(data)
        return {"data": payload}

    @staticmethod
    def _edge_element(
        src: object,
        tgt: object,
        data: dict,
        defaults: dict,
        exclude: frozenset[str] = frozenset({"id"}),
    ) -> dict:
        payload = {"source": str(src), "target": str(tgt), **defaults}
        safe_data = {k: v for k, v in data.items() if k not in exclude}
        payload.update(safe_data)
        return {"data": payload}
