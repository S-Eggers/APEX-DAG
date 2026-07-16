from __future__ import annotations

from collections import deque

import networkx as nx

from SystemX.sca import DOMAIN_NODE_TYPES, NODE_TYPES
from SystemX.sca.constants import COMPUTE_HUBS, DOMAIN_EDGE_TYPES, HUB_NODE_TYPES, REVERSE_DOMAIN_EDGE_TYPES

from ..constants import (
    ATTR_NODE_TYPE,
    ATTR_PREDICTED_LABEL,
    CONFIDENCE_OVERRIDE_THRESHOLD,
    MARGIN_OVERRIDE_THRESHOLD,
    is_overridable,
)
from ..interfaces import GraphRefinementRule
from ..state import RefinementState

_VARIABLE_TYPES = frozenset({NODE_TYPES["VARIABLE"], NODE_TYPES.get("INTERMEDIATE", -1)})

class DynamicTaintPropagationRule(GraphRefinementRule):
    """Forward BFS: Variable -> Hub -> Variable."""

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        queue = deque([n for n in state.known_datasets if g.nodes[n].get(ATTR_NODE_TYPE) not in HUB_NODE_TYPES])
        visited = set(queue)

        while queue:
            u = queue.popleft()
            for _, op_hub in g.out_edges(u):
                if g.nodes[op_hub].get(ATTR_NODE_TYPE) in HUB_NODE_TYPES:
                    if op_hub not in state.node_domain_updates:
                        state.node_domain_updates[op_hub] = "DATA_TRANSFORM"

                    for _, v in g.out_edges(op_hub):
                        if v not in visited:
                            visited.add(v)
                            if g.nodes[v].get(ATTR_NODE_TYPE) not in HUB_NODE_TYPES:
                                state.known_datasets.add(v)
                                state.node_type_updates[v] = DOMAIN_NODE_TYPES["DATASET"]
                            queue.append(v)

class BackwardFeatureRule(GraphRefinementRule):
    """Backward jump: Hub(MODEL_OPERATION) -> predecessor Variables become Features."""

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        feature_type = DOMAIN_NODE_TYPES.get("FEATURE_ENGINEERING", DOMAIN_NODE_TYPES["DATASET"])

        for node, label in state.node_domain_updates.items():
            if label == "MODEL_OPERATION":
                for pred in g.predecessors(node):
                    if g.nodes[pred].get(ATTR_NODE_TYPE) in (NODE_TYPES["VARIABLE"], NODE_TYPES["INTERMEDIATE"]):
                        state.known_datasets.add(pred)
                        state.node_type_updates[pred] = feature_type

class ForwardDatasetPropagationRule(GraphRefinementRule):
    """Forward propagation: DATA_IMPORT_EXTRACTION hub -> successor Variables become Datasets."""

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        for node, label in state.node_domain_updates.items():
            if label != "DATA_IMPORT_EXTRACTION":
                continue
            for _, successor in g.out_edges(node):
                if g.nodes[successor].get(ATTR_NODE_TYPE) in _VARIABLE_TYPES:
                    state.known_datasets.add(successor)

class ForwardModelPropagationRule(GraphRefinementRule):
    """Forward propagation: MODEL_OPERATION hub -> successor Variables become Model artifacts."""

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        for node, label in state.node_domain_updates.items():
            if label != "MODEL_OPERATION":
                continue
            for _, successor in g.out_edges(node):
                if g.nodes[successor].get(ATTR_NODE_TYPE) in _VARIABLE_TYPES:
                    state.known_models.add(successor)

class RelevancePropagationRule(GraphRefinementRule):
    """Rescue low-confidence NOT_RELEVANT calls that sit on a data-carrying path."""

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD,
        margin_threshold: float = MARGIN_OVERRIDE_THRESHOLD,
    ) -> None:
        self._thr = confidence_threshold
        self._margin = margin_threshold

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        not_rel = DOMAIN_EDGE_TYPES["NOT_RELEVANT"]
        data_import = DOMAIN_EDGE_TYPES["DATA_IMPORT_EXTRACTION"]

        anchors = [
            n for n, d in g.nodes(data=True)
            if int(d.get(ATTR_NODE_TYPE, -1)) in COMPUTE_HUBS and d.get(ATTR_PREDICTED_LABEL) == data_import
        ]
        if not anchors:
            return

        reachable: set = set()
        for a in anchors:
            reachable |= nx.descendants(g, a)

        for n in reachable:
            d = g.nodes[n]
            if int(d.get(ATTR_NODE_TYPE, -1)) not in COMPUTE_HUBS:
                continue
            if d.get(ATTR_PREDICTED_LABEL) != not_rel:
                continue
            if not is_overridable(d, self._thr, self._margin):
                continue
            runner = d.get("predicted_runner_up")
            if runner is None or int(runner) == not_rel:
                continue
            runner = int(runner)
            state.node_predicted_updates[n] = runner
            state.node_domain_updates[n] = REVERSE_DOMAIN_EDGE_TYPES.get(runner, "DATA_TRANSFORM")

class BackwardSourcePropagationRule(GraphRefinementRule):
    """Infer a data source at the *root* of a confident transform chain."""

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD,
        margin_threshold: float = MARGIN_OVERRIDE_THRESHOLD,
    ) -> None:
        self._thr = confidence_threshold
        self._margin = margin_threshold

    def _is_source_call(self, g: nx.MultiDiGraph, n: object) -> bool:
        """True if no input of n is produced by another operation hub."""
        for var in g.predecessors(n):
            for producer in g.predecessors(var):
                if int(g.nodes[producer].get(ATTR_NODE_TYPE, -1)) in COMPUTE_HUBS:
                    return False
        return True

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        transform = DOMAIN_EDGE_TYPES["DATA_TRANSFORM"]
        not_rel = DOMAIN_EDGE_TYPES["NOT_RELEVANT"]
        data_import = DOMAIN_EDGE_TYPES["DATA_IMPORT_EXTRACTION"]

        transforms = [
            n for n, d in g.nodes(data=True)
            if int(d.get(ATTR_NODE_TYPE, -1)) in COMPUTE_HUBS
            and d.get(ATTR_PREDICTED_LABEL) == transform
            and float(d.get("predicted_confidence", 0.0)) >= self._thr
        ]
        if not transforms:
            return

        upstream: set = set()
        for t in transforms:
            upstream |= nx.ancestors(g, t)

        for n in upstream:
            d = g.nodes[n]
            if int(d.get(ATTR_NODE_TYPE, -1)) not in COMPUTE_HUBS:
                continue
            if d.get(ATTR_PREDICTED_LABEL) != not_rel:
                continue
            if not is_overridable(d, self._thr, self._margin):
                continue
            if not self._is_source_call(g, n):
                continue
            runner = d.get("predicted_runner_up")
            target = data_import if int(runner) == data_import else (int(runner) if runner is not None and int(runner) != not_rel else None)
            if target is None:
                continue
            state.node_predicted_updates[n] = target
            state.node_domain_updates[n] = REVERSE_DOMAIN_EDGE_TYPES.get(target, "DATA_IMPORT_EXTRACTION")

class NoneRule(GraphRefinementRule):
    """Null Object: structural placeholder that performs no graph mutations."""

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        pass
