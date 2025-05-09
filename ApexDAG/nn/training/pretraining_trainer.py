import os
import torch
import wandb
from ApexDAG.nn.training.base_trainer import BaseTrainer


import torch.nn as nn



class PretrainingTrainer(BaseTrainer):
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
        return  mask
    
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
        return self.criterion_node_reconstruction(masked_predictions, masked_targets)

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)

        # Mask edge features
        mask = self.mask_nodes(data.x)
        data.mask = mask

        # Forward pass
        outputs = self.model(data, task=None, mask=mask)

        # Compute losses
        losses = {}
        if "node_type_preds" in outputs:
            losses["node_type_loss"] = self.criterion_node(outputs["node_type_preds"], data.node_types)
        if "edge_type_preds" in outputs:
            losses["edge_type_loss"] = self.criterion_edge_type(outputs["edge_type_preds"], data.edge_types)

        # Compute masked edge reconstruction loss
        if "node_reconstruction" in outputs:
            predictions = outputs["node_reconstruction"]
            targets = data.x  # Use the original edge features for reconstruction
            losses["node_reconstruction_loss"] = self.compute_masked_node_loss(predictions, targets, mask)

        # Backward pass and optimization
        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def validate_step(self, data):
        self.model.eval()
        data = data.to(self.device)

        # Mask edge features
        mask = self.mask_nodes(data.x)
        data.mask = mask

        with torch.no_grad():
            outputs = self.model(data, task=None, mask=mask)

        # Compute losses
        losses = {}
        if "node_type_preds" in outputs:
            losses["node_type_loss"] = self.criterion_node(outputs["node_type_preds"], data.node_types)
        if "edge_type_preds" in outputs:
            losses["edge_type_loss"] = self.criterion_edge_type(outputs["edge_type_preds"], data.edge_types)

        # Compute masked edge reconstruction loss
        if "node_reconstruction" in outputs:
            predictions = outputs["node_reconstruction"]
            targets = data.x  # Use the original edge features for reconstruction
            losses["node_reconstruction_loss"] = self.compute_masked_node_loss(predictions, targets, mask)

        return {k: v.item() for k, v in losses.items()}
    
    def save_checkpoint(self, epoch, val_loss, filename=None):
        if filename is None:
            filename = f"model_epoch_pretrained_masked_{epoch}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        
        artifact = wandb.Artifact('model-checkpoints', type='model')
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

