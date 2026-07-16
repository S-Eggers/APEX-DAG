from __future__ import annotations

import networkx as nx

from SystemX.sca import DOMAIN_NODE_TYPES, NODE_TYPES
from SystemX.sca.constants import DOMAIN_EDGE_TYPES, HUB_NODE_TYPES

from ..constants import ATTR_NODE_TYPE, CONFIDENCE_OVERRIDE_THRESHOLD, callee_name, is_confident
from ..interfaces import GraphRefinementRule
from ..state import RefinementState

_WRITE_CALLEES = frozenset({
    "to_csv", "to_excel", "to_parquet", "to_pickle", "to_json", "to_sql", "to_hdf",
    "to_feather", "to_stata", "to_html", "savefig", "savez", "savetxt", "save",
    "save_weights", "save_model", "dump", "write", "writerow", "writerows", "writelines",
})

class EvaluationSinkRule(GraphRefinementRule):
    """Marks a hub that consumes both a Model and a Dataset as an evaluation (EDA) sink."""

    def __init__(self, confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD) -> None:
        self._thr = confidence_threshold

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        for n in g.nodes():
            if g.nodes[n].get(ATTR_NODE_TYPE) in HUB_NODE_TYPES:
                if state.node_domain_updates.get(n) == "MODEL_OPERATION":
                    continue
                if is_confident(g.nodes[n], self._thr):
                    continue
                preds = list(g.predecessors(n))
                has_model = any(p in state.known_models for p in preds)
                has_data = any(p in state.known_datasets for p in preds)

                if has_model and has_data:
                    state.node_domain_updates[n] = "EDA"
                    state.node_predicted_updates[n] = DOMAIN_EDGE_TYPES.get("EDA", 3)

class ArtifactSerializationRule(GraphRefinementRule):
    """Labels hubs containing serialization calls and their outputs as Artifacts."""

    def __init__(self, confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD) -> None:
        self._thr = confidence_threshold

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        artifact_type = DOMAIN_NODE_TYPES.get("ARTIFACT")

        for n, data in g.nodes(data=True):
            if data.get(ATTR_NODE_TYPE) in HUB_NODE_TYPES:
                if is_confident(data, self._thr):
                    continue
                if callee_name(data.get("code", "")) in _WRITE_CALLEES:
                    state.node_domain_updates[n] = "DATA_EXPORT"
                    for _, v in g.out_edges(n):
                        if g.nodes[v].get(ATTR_NODE_TYPE) not in HUB_NODE_TYPES:
                            state.node_type_updates[v] = artifact_type

class MutualExclusionRule(GraphRefinementRule):
    """Final pass: synchronize physical node types with resolved domain identities."""

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        dataset_type = DOMAIN_NODE_TYPES["DATASET"]
        model_type = DOMAIN_NODE_TYPES["MODEL"]

        for n in state.known_datasets:
            if g.nodes[n].get(ATTR_NODE_TYPE) in (NODE_TYPES["VARIABLE"], NODE_TYPES["INTERMEDIATE"]):
                state.node_type_updates[n] = dataset_type

        for n in state.known_models:
            if g.nodes[n].get(ATTR_NODE_TYPE) in (NODE_TYPES["VARIABLE"], NODE_TYPES["INTERMEDIATE"]) and n not in state.node_type_updates:
                state.node_type_updates[n] = model_type

class LiteralResolutionRule(GraphRefinementRule):
    """Resolves Literal nodes into Hyperparameters or Parameters based on their downstream hub."""

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        literal_type = NODE_TYPES["LITERAL"]
        hyperparam_type = DOMAIN_NODE_TYPES.get("HYPERPARAMETER")

        for node, data in g.nodes(data=True):
            if data.get(ATTR_NODE_TYPE) == literal_type:
                for _, op_hub in g.out_edges(node):
                    hub_label = state.node_domain_updates.get(op_hub)

                    if hub_label == "MODEL_OPERATION":
                        state.node_type_updates[node] = hyperparam_type
                        state.known_hyperparams.add(node)
                    elif hub_label == "DATA_IMPORT_EXTRACTION":
                        state.node_domain_updates[node] = "PARAMETER"
