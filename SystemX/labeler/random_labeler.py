import logging

import networkx as nx
import numpy as np

from SystemX.labeler._scoring import top2_and_confidence
from SystemX.labeler.edge_labeler import EdgeLabeler
from SystemX.sca.constants import REVERSE_DOMAIN_EDGE_TYPES
from SystemX.sca.cdf_ir import CDFIntermediateRepresentation

logger = logging.getLogger(__name__)

_CALL_NODE_TYPES = frozenset({9, 6})

_N_DOMAIN_CLASSES = 6

class RandomLabeler(EdgeLabeler):
    """Placeholder LLM labeler - assigns a random domain class to each CALL/LOOP node."""

    def __init__(self, seed: int = 42, n_classes: int = _N_DOMAIN_CLASSES) -> None:
        self._rng = np.random.default_rng(seed)
        self._n = n_classes

    def apply_labels(self, graph: CDFIntermediateRepresentation) -> None:
        nx_G = graph.get_graph()

        node_updates: dict = {}
        for node_id, data in nx_G.nodes(data=True):
            if data.get("node_type") not in _CALL_NODE_TYPES:
                continue
            probs = self._rng.dirichlet(np.ones(self._n))
            label_int, runner_up, conf, margin = top2_and_confidence(probs)
            node_updates[node_id] = {
                "predicted_label": label_int,
                "domain_label": REVERSE_DOMAIN_EDGE_TYPES.get(label_int, "NOT_RELEVANT"),
                "predicted_runner_up": runner_up,
                "predicted_confidence": conf,
                "predicted_margin": margin,
            }

        nx.set_node_attributes(nx_G, node_updates)
        logger.info("RandomLabeler (LLM placeholder) labelled %d COMPUTE_HUB nodes.", len(node_updates))
