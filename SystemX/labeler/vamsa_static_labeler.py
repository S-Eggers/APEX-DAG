import logging

from SystemX.labeler.edge_labeler import EdgeLabeler
from SystemX.labeler.llm_labeler import _collect_node_labels_as_edge_attrs
from SystemX.labeling.vamsa_kb_index import VamsaKBIndex
from SystemX.labeling.vamsa_loader import DomainEdgeId, VamsaEntry
from SystemX.sca.constants import REVERSE_DOMAIN_EDGE_TYPES
from SystemX.sca.cdf_ir import CDFIntermediateRepresentation

logger = logging.getLogger(__name__)

_CALL_NODE_TYPES = frozenset({9, 6})

class VamsaStaticLabeler(EdgeLabeler):
    """Labels CALL/LOOP nodes using the static Vamsa KB - no LLM fallback."""

    def __init__(self, vamsa_mapping: dict[VamsaEntry, DomainEdgeId], *, strict_provenance: bool = False) -> None:
        self._kb = VamsaKBIndex(vamsa_mapping, allow_unambiguous_fallback=not strict_provenance)

    def apply_labels(self, graph: CDFIntermediateRepresentation) -> None:
        G = graph.get_graph()

        for node_id, data in G.nodes(data=True):
            if data.get("node_type") not in _CALL_NODE_TYPES:
                continue
            if "domain_label" in data:
                continue
            match = self._kb.match(node_id, data)
            if match:
                label_int = match.predicted_label
                G.nodes[node_id].update(
                    {
                        "domain_label": REVERSE_DOMAIN_EDGE_TYPES.get(label_int, match.domain_label),
                        "predicted_label": label_int,
                    }
                )

        attrs_to_set = _collect_node_labels_as_edge_attrs(G)
        graph.set_domain_label(attrs_to_set, name="predicted_label")
        logger.info("VamsaStaticLabeler finished. %d edges labelled.", len(attrs_to_set))
