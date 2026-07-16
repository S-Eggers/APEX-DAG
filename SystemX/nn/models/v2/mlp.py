import torch
import torch.nn as nn

from SystemX.sca.constants import DOMAIN_EDGE_TYPES

class ComputeHubMLP(nn.Module):
    """Lightweight MLP baseline for COMPUTE_HUB (CALL node) domain classification."""

    def __init__(
        self,
        in_features: int = 302,
        hidden: int = 256,
        num_classes: int | None = None,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes is None:
            num_classes = len(DOMAIN_EDGE_TYPES)
        self.use_batchnorm = use_batchnorm

        def _norm(width: int) -> nn.Module:
            return nn.BatchNorm1d(width) if use_batchnorm else nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            _norm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            _norm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
