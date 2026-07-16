import os

import torch
import wandb

from SystemX.nn.training.base_trainer import BaseTrainer

class PretrainingTrainer(BaseTrainer):
    def __init__(self, model: object, train_dataset: object, val_dataset: object, **kwargs: object) -> None:
        super().__init__(model, train_dataset, val_dataset, **kwargs)
        self.conf_matrices_types = [
            "node_type_preds",
            "edge_type_preds",
            "edge_existence_preds",
        ]

    def train_step(self, data: object) -> dict:
        self.model.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)
        outputs = self.model(data)

        losses = {}
        if "node_type_preds" in outputs:
            losses["node_type_loss"] = self.criterion_node(outputs["node_type_preds"], data.node_types)
        if "edge_type_preds" in outputs:
            valid_edge_mask = data.edge_types != -1
            if valid_edge_mask.any():
                edge_type_preds = outputs["edge_type_preds"][valid_edge_mask]
                edge_type_targets = data.edge_types[valid_edge_mask]
                losses["edge_type_loss"] = self.criterion_edge_type(edge_type_preds, edge_type_targets)

        if "edge_existence_preds" in outputs:
            edge_existence_preds = outputs["edge_existence_preds"].squeeze(dim=-1)
            edge_existence_targets = data.edge_existence.float()
            losses["edge_existence_loss"] = self.criterion_edge_existence(edge_existence_preds, edge_existence_targets)

        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def validate_step(self, data: object) -> dict:
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            outputs = self.model(data)

        losses = {}
        if "node_type_preds" in outputs:
            losses["node_type_loss"] = self.criterion_node(outputs["node_type_preds"], data.node_types)
        if "edge_type_preds" in outputs:
            valid_edge_mask = data.edge_types != -1
            if valid_edge_mask.any():
                edge_type_preds = outputs["edge_type_preds"][valid_edge_mask]
                edge_type_targets = data.edge_types[valid_edge_mask]
                losses["edge_type_loss"] = self.criterion_edge_type(edge_type_preds, edge_type_targets)

        if "edge_existence_preds" in outputs:
            edge_existence_preds = outputs["edge_existence_preds"].squeeze(dim=-1)
            edge_existence_targets = data.edge_existence.float()
            losses["edge_existence_loss"] = self.criterion_edge_existence(edge_existence_preds, edge_existence_targets)

        return {k: v.item() for k, v in losses.items()}

    def save_checkpoint(self, epoch: int, val_loss: float, suffix_name: str = "", filename: str | None = None) -> None:
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
    def __init__(self, model: object, train_dataset: object, val_dataset: object, **kwargs: object) -> None:
        super().__init__(model, train_dataset, val_dataset, **kwargs)
        self.conf_matrices_types = ["node_type_preds", "edge_type_preds"]

    def mask_nodes(self, node_features: object, mask_prob: float = 0.15) -> object:
        """Masks a percentage of edge features."""
        num_nodes = node_features.size(0)
        mask = torch.rand(num_nodes) < mask_prob
        return mask

    def compute_masked_node_loss(self, predictions: object, targets: object, mask: object) -> object:
        """Computes the reconstruction loss for masked edges."""
        masked_predictions = predictions[mask]
        masked_targets = targets[mask]
        return self.criterion_reconstruction(masked_predictions, masked_targets)

    def train_step(self, data: object) -> dict:
        self.model.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)

        mask = self.mask_nodes(data.x)
        data.mask = mask

        outputs = self.model(data, task=None, mask=mask)

        losses = {}
        if "node_type_preds" in outputs:
            losses["node_type_loss"] = self.criterion_node(outputs["node_type_preds"], data.node_types)
        if "edge_type_preds" in outputs:
            losses["edge_type_loss"] = self.criterion_edge_type(outputs["edge_type_preds"], data.edge_types)

        if "reconstruction" in outputs:
            predictions = outputs["reconstruction"]
            targets = data.x
            losses["reconstruction_loss"] = self.compute_masked_node_loss(predictions, targets, mask)

        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def validate_step(self, data: object) -> dict:
        self.model.eval()
        data = data.to(self.device)

        mask = self.mask_nodes(data.x)
        data.mask = mask

        with torch.no_grad():
            outputs = self.model(data, task=None, mask=mask)

        losses = {}
        if "node_type_preds" in outputs:
            losses["node_type_loss"] = self.criterion_node(outputs["node_type_preds"], data.node_types)
        if "edge_type_preds" in outputs:
            losses["edge_type_loss"] = self.criterion_edge_type(outputs["edge_type_preds"], data.edge_types)

        if "reconstruction" in outputs:
            predictions = outputs["reconstruction"]
            targets = data.x
            losses["reconstruction_loss"] = self.compute_masked_node_loss(predictions, targets, mask)

        return {k: v.item() for k, v in losses.items()}

    def save_checkpoint(self, epoch: int, val_loss: float, suffix_name: str = "masked_model", filename: str | None = None) -> None:
        super().save_checkpoint(epoch, val_loss, suffix_name, filename)
