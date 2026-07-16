import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

class SystemXHeteroGraphTransformer(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        out_classes: int = 9,
        num_heads: int = 4,
        num_layers: int = 3,
        metadata: tuple[list[str], list[tuple[str, str, str]]] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.dropout = dropout
        self.node_types = list(metadata[0]) if metadata else ["variable", "operation"]

        self.lin_dict = torch.nn.ModuleDict(
            {nt: Linear(-1, hidden_channels) for nt in self.node_types}
        )

        self.input_norm = torch.nn.ModuleDict(
            {nt: nn.LayerNorm(hidden_channels) for nt in self.node_types}
        )

        self.convs = torch.nn.ModuleList()
        self.conv_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=self.metadata,
                    heads=num_heads,
                )
            )
            self.conv_norms.append(
                torch.nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in self.node_types})
            )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_classes),
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        h_dict = {
            node_type: F.gelu(self.input_norm[node_type](self.lin_dict[node_type](x)))
            for node_type, x in x_dict.items()
        }

        for conv, norm in zip(self.convs, self.conv_norms):
            m_dict = conv(h_dict, edge_index_dict)
            h_dict = {
                nt: F.dropout(F.gelu(norm[nt](m + h_dict[nt])), p=self.dropout, training=self.training)
                for nt, m in m_dict.items()
            }

        return self.classifier(h_dict["operation"])
