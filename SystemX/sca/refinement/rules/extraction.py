from __future__ import annotations

import networkx as nx

from SystemX.sca import DOMAIN_NODE_TYPES, NODE_TYPES, REVERSE_DOMAIN_EDGE_TYPES
from SystemX.sca.constants import COMPUTE_HUBS

from ..constants import (
    ATTR_DOMAIN_NODE,
    ATTR_NODE_TYPE,
    ATTR_PREDICTED_LABEL,
    CONFIDENCE_OVERRIDE_THRESHOLD,
    callee_name,
    is_confident,
)
from ..interfaces import GraphRefinementRule
from ..state import RefinementState

_LOAD_CALLEES = frozenset({
    "read_csv", "read_excel", "read_sql", "read_sql_query", "read_json", "read_table",
    "read_parquet", "read_html", "read_pickle", "read_fwf", "read_stata", "read_feather",
    "read_hdf", "load", "loadtxt", "load_data", "load_iris", "load_boston", "load_digits",
    "load_wine", "load_breast_cancer", "load_diabetes", "load_dataset", "load_files",
    "fetch_openml", "fetch_california_housing", "fetch_20newsgroups", "imread", "open",
})
_MODEL_CALLEES = frozenset({
    "fit", "predict", "predict_proba", "predict_log_proba", "fit_predict",
    "partial_fit", "score", "decision_function",
})
_MODEL_CTOR_SUFFIXES = ("Classifier", "Regressor", "Regression")

class BaseTruthExtractionRule(GraphRefinementRule):
    """Extracts initial entity identities from node types and GNN predictions."""

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        domain_dataset_type = DOMAIN_NODE_TYPES["DATASET"]
        domain_model_type = DOMAIN_NODE_TYPES["MODEL"]
        base_dataset_type = NODE_TYPES["DATASET"]

        for node, data in g.nodes(data=True):
            n_type = data.get(ATTR_NODE_TYPE)
            is_domain = bool(data.get(ATTR_DOMAIN_NODE))

            if n_type == base_dataset_type or (is_domain and n_type == domain_dataset_type):
                state.known_datasets.add(node)
            elif is_domain and n_type == domain_model_type:
                state.known_models.add(node)

            if n_type in COMPUTE_HUBS:
                pred_idx = data.get(ATTR_PREDICTED_LABEL)
                if pred_idx is not None:
                    label = REVERSE_DOMAIN_EDGE_TYPES.get(pred_idx)
                    if label:
                        state.node_domain_updates[node] = label
                        if label == "DATA_IMPORT_EXTRACTION":
                            state.known_datasets.add(node)
                        elif label == "MODEL_OPERATION":
                            state.known_models.add(node)

class StaticSeedingRule(GraphRefinementRule):
    """Deterministic seeding of COMPUTE_HUB nodes based on source code patterns."""

    def __init__(self, confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD) -> None:
        self._thr = confidence_threshold

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        for node, data in g.nodes(data=True):
            if data.get(ATTR_NODE_TYPE) not in COMPUTE_HUBS:
                continue

            code = str(data.get("code", ""))
            callee = callee_name(code)
            confident = is_confident(data, self._thr)

            is_load = callee in _LOAD_CALLEES
            is_model = callee in _MODEL_CALLEES or callee.endswith(_MODEL_CTOR_SUFFIXES)

            if is_load:
                if not confident:
                    state.node_domain_updates[node] = "DATA_IMPORT_EXTRACTION"
                state.known_datasets.add(node)
            elif is_model:
                if not confident:
                    state.node_domain_updates[node] = "MODEL_OPERATION"
                state.known_models.add(node)
