from __future__ import annotations

NodeId = str

class RefinementState:
    """Carries mutation state across the bipartite rule pipeline."""

    def __init__(self) -> None:
        self.known_datasets: set[NodeId] = set()
        self.known_models: set[NodeId] = set()
        self.known_hyperparams: set[NodeId] = set()

        self.node_type_updates: dict[NodeId, int] = {}
        self.node_domain_updates: dict[NodeId, str] = {}
        self.node_predicted_updates: dict[NodeId, int] = {}

        self.leakage_nodes: set[NodeId] = set()
        self.dead_code_nodes: set[NodeId] = set()

        self.leakage_findings: list = []

        self.edge_domain_updates: dict[tuple[NodeId, NodeId, int], str] = {}
        self.edge_numeric_updates: dict[tuple[NodeId, NodeId, int], int] = {}
