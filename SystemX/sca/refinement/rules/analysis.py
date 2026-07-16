from __future__ import annotations

import re

import networkx as nx

from SystemX.sca.constants import HUB_NODE_TYPES

from ..constants import ATTR_HAS_LEAKAGE, ATTR_NODE_TYPE
from ..interfaces import GraphRefinementRule
from ..state import RefinementState

_TEST_TOKENS = frozenset({"test", "val", "valid", "validation", "holdout"})

def _tokens(name: object) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t}

class LeakageDetectionRule(GraphRefinementRule):
    """Flags data leakage: a test/val dataset node with a path to a training hub."""

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        test_vars = {n for n in state.known_datasets if _tokens(n) & _TEST_TOKENS}
        model_hubs = {n for n, label in state.node_domain_updates.items() if label == "MODEL_OPERATION"}

        for t_var in test_vars:
            for m_hub in model_hubs:
                if nx.has_path(g, t_var, m_hub):
                    g.graph[ATTR_HAS_LEAKAGE] = True
                    state.leakage_nodes.add(m_hub)

class DeadCodeEliminationRule(GraphRefinementRule):
    """Marks a terminal hub (FUNCTION, LOOP, CALL) as Dead Code only when it has no meaningful domain label."""

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        dead_eligible = {None, "NOT_RELEVANT"}

        for n, data in g.nodes(data=True):
            if data.get(ATTR_NODE_TYPE) in HUB_NODE_TYPES and g.out_degree(n) == 0:
                label = state.node_domain_updates.get(n)
                if label in dead_eligible:
                    state.dead_code_nodes.add(n)
