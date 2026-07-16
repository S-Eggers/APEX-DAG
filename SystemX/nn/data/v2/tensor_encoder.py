import logging

import networkx as nx
import torch
from torch_geometric.data import HeteroData

from SystemX.nn.data.v2.embedding import EmbeddingProtocol
from SystemX.nn.data.v2.feature_extractor import ComputeHubFeatureExtractor, FeatureGroup
from SystemX.nn.data.v2.pruner import GraphPruner
from SystemX.sca.constants import COMPUTE_HUBS

logger = logging.getLogger(__name__)

HETERO_METADATA = (
    ["variable", "operation"],
    [
        ("variable", "flows_to", "operation"),
        ("operation", "produces", "variable"),
        ("operation", "rev_flows_to", "variable"),
        ("variable", "rev_produces", "operation"),
    ],
)

class EncoderV2:
    """Encodes a raw DFG into a bipartite HeteroData for the HGT model."""

    def __init__(
        self,
        embedding_model: EmbeddingProtocol,
        pruner: GraphPruner,
        feature_groups: FeatureGroup = FeatureGroup.STANDARD,
        rich_variable_features: bool = False,
    ) -> None:
        self._embedding = embedding_model
        self._pruner = pruner
        self._feature_groups = feature_groups
        self._rich_variable_features = rich_variable_features
        self._op_extractor = ComputeHubFeatureExtractor(embedding=embedding_model, groups=feature_groups)

    @property
    def op_extractor(self) -> ComputeHubFeatureExtractor:
        """The COMPUTE_HUB feature extractor used for operation nodes."""
        return self._op_extractor

    @property
    def variable_feature_dim(self) -> int:
        d = self._embedding.dimension
        return (2 * d + 2) if self._rich_variable_features else d

    def encode(self, raw_graph: nx.MultiDiGraph) -> HeteroData:
        """Encode without the index mapping (used during training)."""
        data, _ = self.encode_with_mapping(raw_graph)
        return data

    def encode_with_mapping(self, raw_graph: nx.MultiDiGraph) -> tuple[HeteroData, dict[int, object]]:
        """Encode the graph and return a mapping from operation-tensor-index back to the original graph node id."""
        graph = self._pruner.prune(raw_graph)
        data = HeteroData()

        op_feature_rows: dict[object, torch.Tensor] = {}
        op_feature_matrix, op_row_ids, _ = self._op_extractor.extract(graph)
        for nid, row in zip(op_row_ids, op_feature_matrix, strict=False):
            op_feature_rows[nid] = torch.as_tensor(row, dtype=torch.float32)

        in_deg = dict(graph.in_degree()) if self._rich_variable_features else {}
        out_deg = dict(graph.out_degree()) if self._rich_variable_features else {}

        var_features: list[torch.Tensor] = []
        op_features: list[torch.Tensor] = []
        op_labels: list[int] = []
        op_mask: list[bool] = []

        var_map: dict[object, int] = {}
        op_map: dict[object, int] = {}
        op_index_to_node_id: dict[int, object] = {}

        for node_id, attrs in graph.nodes(data=True):
            node_type = int(attrs.get("node_type", -1))

            if node_type in COMPUTE_HUBS:
                idx = len(op_features)
                op_map[node_id] = idx
                op_index_to_node_id[idx] = node_id

                row = op_feature_rows.get(node_id)
                if row is None:
                    row = torch.zeros(self._op_extractor.feature_dim, dtype=torch.float32)
                op_features.append(row)

                domain_label = int(attrs.get("domain_label", -1))
                op_labels.append(domain_label if domain_label != -1 else 0)
                op_mask.append(domain_label != -1)
            else:
                raw_code = attrs.get("code", "")
                text = str(raw_code) if raw_code and raw_code != "None" else str(attrs.get("label", ""))
                var_map[node_id] = len(var_features)
                var_features.append(self._encode_variable(graph, node_id, text, in_deg, out_deg))

        var_dim = self.variable_feature_dim
        op_dim = self._op_extractor.feature_dim
        data["variable"].x = torch.stack(var_features) if var_features else torch.empty((0, var_dim))
        data["operation"].x = torch.stack(op_features) if op_features else torch.empty((0, op_dim))
        data["operation"].y = torch.tensor(op_labels, dtype=torch.long)
        data["operation"].train_mask = torch.tensor(op_mask, dtype=torch.bool)

        var_to_op_src, var_to_op_tgt = [], []
        op_to_var_src, op_to_var_tgt = [], []

        for u, v, _ in graph.edges(data=True):
            if u in var_map and v in op_map:
                var_to_op_src.append(var_map[u])
                var_to_op_tgt.append(op_map[v])
            elif u in op_map and v in var_map:
                op_to_var_src.append(op_map[u])
                op_to_var_tgt.append(var_map[v])

        data["variable", "flows_to", "operation"].edge_index = torch.tensor([var_to_op_src, var_to_op_tgt], dtype=torch.long)
        data["operation", "produces", "variable"].edge_index = torch.tensor([op_to_var_src, op_to_var_tgt], dtype=torch.long)
        data["operation", "rev_flows_to", "variable"].edge_index = torch.tensor([var_to_op_tgt, var_to_op_src], dtype=torch.long)
        data["variable", "rev_produces", "operation"].edge_index = torch.tensor([op_to_var_tgt, op_to_var_src], dtype=torch.long)

        return data, op_index_to_node_id

    def _encode_variable(
        self,
        graph: nx.MultiDiGraph,
        node_id: object,
        text: str,
        in_deg: dict,
        out_deg: dict,
    ) -> torch.Tensor:
        """Variable-node feature vector."""
        name_emb = self._embedding.embed(text)
        if not self._rich_variable_features:
            return name_emb

        prod_emb = torch.zeros(self._embedding.dimension, dtype=torch.float32)
        for u in graph.predecessors(node_id):
            if int(graph.nodes[u].get("node_type", -1)) in COMPUTE_HUBS:
                u_code = graph.nodes[u].get("code", "") or graph.nodes[u].get("label", "")
                prod_emb = self._embedding.embed(str(u_code))
                break
        struct = torch.tensor([float(in_deg.get(node_id, 0)), float(out_deg.get(node_id, 0))], dtype=torch.float32)
        return torch.cat([name_emb, prod_emb, struct])
