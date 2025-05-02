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
            residual=residual
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
    def __init__(self, hidden_dim, dim_embed, num_heads=8, node_classes=8, edge_classes=6, dropout=0.2, number_gat_blocks = 2, residual = False):
        super().__init__()
        
        self.up_projection = nn.Linear(dim_embed, hidden_dim)
        
        # GAT Blocks
        self.gat_blocks = nn.ModuleList([
            GATBlock(hidden_dim, hidden_dim, num_heads, dropout, edge_dim=hidden_dim, residual=residual)
            for _ in range(number_gat_blocks)
        ])

        # Task-specific heads
        self.node_type_head = nn.Linear(hidden_dim, node_classes)
        self.edge_type_head = nn.Linear(hidden_dim, edge_classes)
        self.reconstruction_head = nn.Linear(hidden_dim, dim_embed)

        # Edge existence prediction
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data, task=None, mask=None):
        """
        Forward pass for the MultiTaskGAT model.

        Args:
            data: Input graph data containing x (node features), edge_features, and edge_index.
            task (str): Task to perform ("node_classification", "edge_classification", "edge_reconstruction").
            mask (torch.Tensor): Binary mask indicating which edges are masked (for reconstruction).

        Returns:
            outputs (dict): Dictionary containing task-specific outputs.
        """
        x, edge_embeds, edge_index = data.x, data.edge_features, data.edge_index

        # Apply masking if provided
        if mask is not None:
            edge_embeds = edge_embeds.clone()
            edge_embeds[mask] = 0  # Mask the edge features

        # Up-project node and edge embeddings
        x = self.up_projection(x)
        edge_embeds = self.up_projection(edge_embeds)

        # Pass through GAT blocks
        for gat_block in self.gat_blocks:
            x = gat_block(x, edge_index, edge_embeds)

        outputs = {}

        # Node classification
        if task == "node_classification" or task is None:
            outputs["node_type_preds"] = F.softmax(self.node_type_head(x), dim=-1)

        # Edge classification
        if task == "edge_classification" or task is None:
            edge_features = x[edge_index[0]]
            outputs["edge_type_preds"] = F.softmax(self.edge_type_head(edge_features), dim=-1)

        # Edge reconstruction
        if task == "edge_reconstruction" or task is None:
            edge_features = x[edge_index[0]]
            outputs["edge_reconstruction"] = self.reconstruction_head(edge_features)  # Predict original edge features

        return outputs