import os
import torch

from ApexDAG.nn.training.base_trainer import BaseTrainer

class PretrainingTrainer(BaseTrainer):

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)
        outputs = self.model(data)

        losses = {}
        if "node_type_preds" in outputs:
            losses["node_type_loss"] = self.criterion_node(outputs["node_type_preds"], data.node_types)
        if "edge_type_preds" in outputs:
            losses["edge_type_loss"] = self.criterion_edge_type(outputs["edge_type_preds"], data.edge_types)
        if "edge_existence_preds" in outputs:
            edge_existence_preds = outputs["edge_existence_preds"].squeeze(dim=-1)
            edge_existence_targets = data.edge_existence.float()
            losses["edge_existence_loss"] = self.criterion_edge_existence(edge_existence_preds, edge_existence_targets)

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
            losses["node_type_loss"] = self.criterion_node(outputs["node_type_preds"], data.node_types)
        if "edge_type_preds" in outputs:
            losses["edge_type_loss"] = self.criterion_edge_type(outputs["edge_type_preds"], data.edge_types)
        if "edge_existence_preds" in outputs:
            edge_existence_preds = outputs["edge_existence_preds"].squeeze(dim=-1)
            edge_existence_targets = data.edge_existence.float()
            losses["edge_existence_loss"] = self.criterion_edge_existence(edge_existence_preds, edge_existence_targets)

        return {k: v.item() for k, v in losses.items()}
    
    def save_checkpoint(self, epoch, val_loss, filename=None):
        if filename is None:
            filename = f"model_epoch_pretrained_{epoch}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)

