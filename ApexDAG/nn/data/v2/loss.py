import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for severe class imbalances in AST extraction.
    """

    def __init__(
        self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # inputs: [N, C] (logits)
        # targets: [N] (ground truth indices)

        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MultiTaskUncertaintyLoss(nn.Module):
    """
    Combines Node and Edge Focal Losses using learnable Homoscedastic Task Uncertainty.
    """

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.node_criterion = FocalLoss(gamma=gamma)
        self.edge_criterion = FocalLoss(gamma=gamma)

        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        node_logits: torch.Tensor,
        node_targets: torch.Tensor,
        edge_logits: torch.Tensor,
        edge_targets: torch.Tensor,
    ):
        loss_node = self.node_criterion(node_logits, node_targets)
        loss_edge = self.edge_criterion(edge_logits, edge_targets)

        precision_node = torch.exp(-self.log_vars[0])
        precision_edge = torch.exp(-self.log_vars[1])

        weighted_loss_node = 0.5 * precision_node * loss_node + 0.5 * self.log_vars[0]
        weighted_loss_edge = 0.5 * precision_edge * loss_edge + 0.5 * self.log_vars[1]

        total_loss = weighted_loss_node + weighted_loss_edge

        return total_loss, loss_node.item(), loss_edge.item()
