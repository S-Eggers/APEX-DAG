import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class MultiTaskGATv2(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 128,
        out_node_classes: int = 9,
        out_edge_classes: int = 6,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.dropout = dropout

        self.node_proj = nn.Linear(in_channels, hidden_channels)
        self.edge_proj = nn.Linear(in_channels, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_channels,
                    concat=True,
                    dropout=dropout
                )
            )

        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_node_classes)
        )

        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_edge_classes)
        )

    def forward(self, x, edge_index, edge_attr):
        x = F.gelu(self.node_proj(x))
        edge_attr_proj = F.gelu(self.edge_proj(edge_attr))

        for conv in self.convs:
            x_residual = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.gelu(conv(x, edge_index, edge_attr=edge_attr_proj))
            x = x + x_residual 

        node_logits = self.node_classifier(x)
        row, col = edge_index
        source_nodes = x[row]
        target_nodes = x[col]

        edge_representations = torch.cat([source_nodes, target_nodes, edge_attr_proj], dim=-1)
        edge_logits = self.edge_classifier(edge_representations)

        return node_logits, edge_logits