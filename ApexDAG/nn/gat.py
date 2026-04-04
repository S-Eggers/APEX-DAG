import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm


class GATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout, edge_dim, residual=False):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=edge_dim,
            residual=residual,
        )
        self.norm = LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x = self.gat(x, edge_index, edge_attr=edge_attr)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class MultiTaskGAT(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dim_embed,
        num_heads=8,
        node_classes=8,
        edge_classes=6,
        dropout=0.2,
        number_gat_blocks=2,
        residual=False,
        task=None,
    ):
        super().__init__()

        self.task = task
        self.up_projection = nn.Linear(dim_embed, hidden_dim)

        # GAT Blocks
        self.gat_blocks = nn.ModuleList(
            [
                GATBlock(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    edge_dim=hidden_dim,
                    residual=residual,
                )
                for _ in range(number_gat_blocks)
            ]
        )

        # Task-specific heads
        self.node_type_head = nn.Linear(hidden_dim, node_classes)
        self.edge_type_head = nn.Linear(hidden_dim, edge_classes)
        self.reconstruction_head = nn.Linear(hidden_dim, dim_embed)

        # Edge existence prediction
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_embeds, edge_index = data.x, data.edge_features, data.edge_index

        # up-project node and edge embeddings
        x = self.up_projection(x)
        edge_embeds = self.up_projection(edge_embeds)

        for gat_block in self.gat_blocks:
            x = gat_block(x, edge_index, edge_embeds)

        outputs = {}

        if "node_classification" in self.task or self.task is None:
            outputs["node_type_preds"] = F.softmax(self.node_type_head(x), dim=-1)

        if "edge_classification" in self.task or self.task is None:
            edge_features = x[edge_index[0]]
            outputs["edge_type_preds"] = F.softmax(
                self.edge_type_head(edge_features), dim=-1
            )

        if "edge_existence" in self.task or self.task is None:
            source_embeddings = x[edge_index[0]]
            target_embeddings = x[edge_index[1]]
            combined_edge_embeddings = torch.cat(
                [source_embeddings, target_embeddings], dim=-1
            )
            outputs["edge_existence_preds"] = torch.sigmoid(
                self.edge_mlp(combined_edge_embeddings)
            )

            # Edge reconstruction
        if "reconstruction" in self.task or self.task is None:
            outputs["reconstruction"] = self.reconstruction_head(
                x
            )  # Predict original edge features

        return outputs
