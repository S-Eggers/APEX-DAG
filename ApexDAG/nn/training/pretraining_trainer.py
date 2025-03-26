import os
import torch

from ApexDAG.nn.training.base_trainer import BaseTrainer

class PretrainingTrainer(BaseTrainer):
    def __init__(self, model, train_dataset, val_dataset, **kwargs):
        super().__init__(model, train_dataset, val_dataset, **kwargs)
        self.conf_matrices_types = ["node_type_preds", "edge_type_preds"]

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)
        outputs = self.model(data)

        losses = {}
        if "node_type_preds" in outputs:
            valid_edge_mask = data.node_types != -1  
            node_type_preds = outputs["node_type_preds"][valid_edge_mask]
            node_type_targets = data.node_types[valid_edge_mask]
            losses["node_type_loss"] = self.criterion_node(node_type_preds, node_type_targets)
        if "edge_type_preds" in outputs:
            edge_type_preds = outputs["edge_type_preds"]
            edge_type_targets = data.edge_types
            losses["edge_type_loss"] = self.criterion_edge_type(edge_type_preds, edge_type_targets)

    
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
            valid_edge_mask = data.node_types != -1  
            node_type_preds = outputs["node_type_preds"][valid_edge_mask]
            node_type_targets = data.node_types[valid_edge_mask]
            losses["node_type_loss"] = self.criterion_node(node_type_preds, node_type_targets)
        if "edge_type_preds" in outputs:
            edge_type_preds = outputs["edge_type_preds"]
            edge_type_targets = data.edge_types
            losses["edge_type_loss"] = self.criterion_edge_type(edge_type_preds, edge_type_targets)

        return {k: v.item() for k, v in losses.items()}
    
    def save_checkpoint(self, epoch, val_loss, filename=None):
        if filename is None:
            filename = f"model_epoch_pretrained.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)

