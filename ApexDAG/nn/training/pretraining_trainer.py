import os
import torch
import wandb

from ApexDAG.nn.training.base_trainer import BaseTrainer


class PretrainingTrainer(BaseTrainer):
    def __init__(self, model, train_dataset, val_dataset, **kwargs):
        super().__init__(model, train_dataset, val_dataset, **kwargs)
        self.conf_matrices_types = [
            "node_type_preds",
            "edge_type_preds",
            "edge_existence_preds",
        ]

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)
        outputs = self.model(data)

        losses = {}
        if "node_type_preds" in outputs:
            losses["node_type_loss"] = self.criterion_node(
                outputs["node_type_preds"], data.node_types
            )
        if "edge_type_preds" in outputs:
            valid_edge_mask = data.edge_types != -1  # Mask for valid edges
            if valid_edge_mask.any():  # Ensure there are valid edges
                edge_type_preds = outputs["edge_type_preds"][valid_edge_mask]
                edge_type_targets = data.edge_types[valid_edge_mask]
                losses["edge_type_loss"] = self.criterion_edge_type(
                    edge_type_preds, edge_type_targets
                )

        if "edge_existence_preds" in outputs:
            edge_existence_preds = outputs["edge_existence_preds"].squeeze(dim=-1)
            edge_existence_targets = data.edge_existence.float()
            losses["edge_existence_loss"] = self.criterion_edge_existence(
                edge_existence_preds, edge_existence_targets
            )

        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def validate_step(self, data):
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            outputs = self.model(data)

        losses = {}
        if "node_type_preds" in outputs:
            losses["node_type_loss"] = self.criterion_node(
                outputs["node_type_preds"], data.node_types
            )
        if "edge_type_preds" in outputs:
            valid_edge_mask = data.edge_types != -1  # Mask for valid edges
            if valid_edge_mask.any():  # Ensure there are valid edges
                edge_type_preds = outputs["edge_type_preds"][valid_edge_mask]
                edge_type_targets = data.edge_types[valid_edge_mask]
                losses["edge_type_loss"] = self.criterion_edge_type(
                    edge_type_preds, edge_type_targets
                )

        if "edge_existence_preds" in outputs:
            edge_existence_preds = outputs["edge_existence_preds"].squeeze(dim=-1)
            edge_existence_targets = data.edge_existence.float()
            losses["edge_existence_loss"] = self.criterion_edge_existence(
                edge_existence_preds, edge_existence_targets
            )

        return {k: v.item() for k, v in losses.items()}

    def save_checkpoint(self, epoch, val_loss, suffix_name="", filename=None):
        if filename is None:
            filename = f"model_epoch_pretrained_{suffix_name}_{epoch}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            checkpoint_path,
        )

        artifact = wandb.Artifact("model-checkpoints", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)


class PretrainingTrainerMasked(PretrainingTrainer):
    def __init__(self, model, train_dataset, val_dataset, **kwargs):
        super().__init__(model, train_dataset, val_dataset, **kwargs)
        self.conf_matrices_types = ["node_type_preds", "edge_type_preds"]

    def mask_nodes(self, node_features, mask_prob=0.15):
        """
        Masks a percentage of edge features.

        Args:
            edge_features (torch.Tensor): Edge feature tensor of shape (num_edges, feature_dim).
            mask_prob (float): Probability of masking an edge.

        Returns:
            masked_edge_features (torch.Tensor): Edge features with some values masked.
            mask (torch.Tensor): Binary mask indicating which edges were masked.
        """
        num_nodes = node_features.size(0)
        mask = torch.rand(num_nodes) < mask_prob  # Randomly select edges to mask
        return mask

    def compute_masked_node_loss(self, predictions, targets, mask):
        """
        Computes the reconstruction loss for masked edges.

        Args:
            predictions (torch.Tensor): Predicted edge features of shape (num_edges, feature_dim).
            targets (torch.Tensor): Original edge features of shape (num_edges, feature_dim).
            mask (torch.Tensor): Binary mask indicating which edges were masked.

        Returns:
            loss (torch.Tensor): Reconstruction loss for the masked edges.
        """
        masked_predictions = predictions[mask]
        masked_targets = targets[mask]
        return self.criterion_reconstruction(masked_predictions, masked_targets)

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)

        # mask edge features
        mask = self.mask_nodes(data.x)
        data.mask = mask

        # forward pass
        outputs = self.model(data, task=None, mask=mask)

        # compute losses
        losses = {}
        if "node_type_preds" in outputs:
            losses["node_type_loss"] = self.criterion_node(
                outputs["node_type_preds"], data.node_types
            )
        if "edge_type_preds" in outputs:
            losses["edge_type_loss"] = self.criterion_edge_type(
                outputs["edge_type_preds"], data.edge_types
            )

        # compute masked edge reconstruction loss
        if "reconstruction" in outputs:
            predictions = outputs["reconstruction"]
            targets = data.x  # Use the original edge features for reconstruction
            losses["reconstruction_loss"] = self.compute_masked_node_loss(
                predictions, targets, mask
            )

        # backward pass and optimization
        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def validate_step(self, data):
        self.model.eval()
        data = data.to(self.device)

        # mask edge features
        mask = self.mask_nodes(data.x)
        data.mask = mask

        with torch.no_grad():
            outputs = self.model(data, task=None, mask=mask)

        # compute losses
        losses = {}
        if "node_type_preds" in outputs:
            losses["node_type_loss"] = self.criterion_node(
                outputs["node_type_preds"], data.node_types
            )
        if "edge_type_preds" in outputs:
            losses["edge_type_loss"] = self.criterion_edge_type(
                outputs["edge_type_preds"], data.edge_types
            )

        # compute masked edge reconstruction loss
        if "reconstruction" in outputs:
            predictions = outputs["reconstruction"]
            targets = data.x  # Use the original edge features for reconstruction
            losses["reconstruction_loss"] = self.compute_masked_node_loss(
                predictions, targets, mask
            )

        return {k: v.item() for k, v in losses.items()}

    def save_checkpoint(
        self, epoch, val_loss, suffix_name="masked_model", filename=None
    ):
        super().save_checkpoint(epoch, val_loss, suffix_name, filename)
