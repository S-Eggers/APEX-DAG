import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, LayerNorm

class MultiTaskGAT(nn.Module):
    def __init__(self, hidden_dim, 
                 num_heads=8, 
                 node_classes=8, 
                 edge_classes=6, 
                 dropout=0.2,
                 num_types_pretrain = 8,
                 edge_num_types_pretrain = 6,
                 hidden_dim_pretrain_node_embed = 50,
                 hidden_dim_pretrain_edge_embed = 50):
        super(MultiTaskGAT, self).__init__()
        
        # First GAT layer
        self.gat1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=hidden_dim
        )
        self.ln1 = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Second GAT layer (deeper model)
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=hidden_dim
        )
        self.ln2 = LayerNorm(hidden_dim)

        # Task-specific heads
        self.node_type_head = nn.Linear(hidden_dim, node_classes)
        self.edge_type_head = nn.Linear(hidden_dim, edge_classes)
        
        # Edge existence prediction with additional transformation
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data, task=None):
        x, edge_embeds, edge_index = data.x, data.edge_features, data.edge_index

        # First GAT layer with normalization and dropout
        x_res = x  # Save residual connection
        x = self.gat1(x, edge_index, edge_attr=edge_embeds)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Second GAT layer (Deeper Model)
        x = self.gat2(x, edge_index, edge_attr=edge_embeds)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Residual connection
        x = x + x_res

        outputs = {}

        if task == "node_classification" or task is None:
            outputs["node_type_preds"] = F.softmax(self.node_type_head(x), dim=-1)

        if task == "edge_classification" or task is None:
            edge_features = x[edge_index[0]]
            outputs["edge_type_preds"] = F.softmax(self.edge_type_head(edge_features), dim=-1)

        if task == "edge_existence" or task is None:
            source_embeddings = x[edge_index[0]]
            target_embeddings = x[edge_index[1]]
            combined_edge_embeddings = torch.cat([source_embeddings, target_embeddings], dim=-1)
            outputs["edge_existence_preds"] = torch.sigmoid(self.edge_mlp(combined_edge_embeddings))

        return outputs


class ExtendedMultiTaskGAT(MultiTaskGAT):
    def __init__(self, hidden_dim, num_heads, node_classes, edge_classes, downstream_classes):
        super(ExtendedMultiTaskGAT, self).__init__(hidden_dim, num_heads, node_classes, edge_classes)
        # Add downstream task head
        self.downstream_head = nn.Linear(hidden_dim * 2, downstream_classes)

    def forward(self, data, task=None):
        """
        Perform forward pass for all tasks, emphasizing edge embeddings in the downstream task.

        Args:
            data (Data): PyTorch Geometric Data object.
            task (str): Task to run. If None, all tasks are computed.

        Returns:
            dict: Predictions for the requested tasks.
        """
        outputs = super(ExtendedMultiTaskGAT, self).forward(data, task)

        if task == "downstream_task" or task is None:
            # Use both node and edge embeddings for downstream task
            node_embeddings = data.x  # Node embeddings from the GAT layer
            edge_index = data.edge_index
            edge_embeddings = node_embeddings[edge_index[0]] + node_embeddings[edge_index[1]]  # Summing source and target

            # Aggregate edge embeddings
            edge_embeddings_agg = torch.mean(edge_embeddings, dim=0)  # Mean pooling for edges

            # Aggregate node embeddings
            node_embeddings_agg = torch.mean(node_embeddings, dim=0)  # Mean pooling for nodes

            # Combine node and edge embeddings
            combined_features = torch.cat([node_embeddings_agg, edge_embeddings_agg], dim=-1)

            # Downstream prediction
            outputs["downstream_preds"] = F.softmax(self.downstream_head(combined_features), dim=-1)

        return outputs