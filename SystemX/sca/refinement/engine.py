from collections.abc import Sequence

from SystemX.sca.constants import DOMAIN_EDGE_TYPES
from SystemX.sca.cdf_ir import CDFIntermediateRepresentation
from SystemX.sca.refinement.interfaces import GraphRefinementRule
from SystemX.sca.refinement.state import RefinementState

from .constants import (
    ATTR_DOMAIN_NODE,
    ATTR_HAS_LEAKAGE,
    ATTR_IS_DEAD_CODE,
    ATTR_LEAKAGE_CLASS,
    ATTR_LEAKAGE_FINDINGS,
    ATTR_NODE_TYPE,
    ATTR_PREDICTED_LABEL,
)

class GraphRefiner:
    """Executes a sequence of refinement strategies against a Bipartite Dataflow Graph."""

    def __init__(self, rules: Sequence[GraphRefinementRule]) -> None:
        if not rules:
            raise ValueError("GraphRefiner requires at least one rule to execute.")
        self._rules = rules

    def refine(self, graph: CDFIntermediateRepresentation) -> None:
        g = graph.get_graph()
        state = RefinementState()

        for rule in self._rules:
            rule.apply(g, state)

        self._apply_state_updates(graph, state)

    def _apply_state_updates(self, graph: CDFIntermediateRepresentation, state: RefinementState) -> None:
        """Applies collected semantic and structural updates back to the graph."""

        for node, label in state.node_domain_updates.items():
            if node in state.node_predicted_updates:
                continue
            idx = DOMAIN_EDGE_TYPES.get(label)
            if idx is not None and idx >= 0:
                state.node_predicted_updates[node] = idx

        if state.node_type_updates:
            graph.set_domain_node_label(state.node_type_updates, name=ATTR_NODE_TYPE)
            graph.set_domain_node_label({n: True for n in state.node_type_updates}, name=ATTR_DOMAIN_NODE)

        if state.node_domain_updates:
            graph.set_domain_node_label(state.node_domain_updates, name="domain_label")

        if state.node_predicted_updates:
            graph.set_domain_node_label(state.node_predicted_updates, name=ATTR_PREDICTED_LABEL)

        if state.edge_domain_updates:
            graph.set_domain_label(state.edge_domain_updates, name="domain_label")

        if state.edge_numeric_updates:
            graph.set_domain_label(state.edge_numeric_updates, name=ATTR_PREDICTED_LABEL)
            graph.set_domain_label(state.edge_numeric_updates, name="edge_type")

        if state.leakage_nodes:
            graph.set_domain_node_label({n: True for n in state.leakage_nodes}, name=ATTR_HAS_LEAKAGE)

        if state.dead_code_nodes:
            graph.set_domain_node_label({n: True for n in state.dead_code_nodes}, name=ATTR_IS_DEAD_CODE)

        if state.leakage_findings:
            graph.set_domain_node_label(
                {f.node: f.leakage_class for f in state.leakage_findings}, name=ATTR_LEAKAGE_CLASS
            )
            g = graph.get_graph()
            g.graph[ATTR_LEAKAGE_FINDINGS] = [f.to_dict() for f in state.leakage_findings]
