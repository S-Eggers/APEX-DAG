import os
import torch

from ApexDAG.nn.training.base_trainer import BaseTrainer
from torch_geometric.loader import DataLoader


class FinetuningTrainer(BaseTrainer):
    def __init__(self, model, train_dataset, val_dataset, test_dataset, **kwargs):
        super().__init__(model, train_dataset, val_dataset, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    def save_checkpoint(self, epoch, val_loss, filename=None):
        if filename is None:
            filename = f"model_epoch_finetuned_{epoch}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)
        outputs = self.model(data)

        losses = {}
        
        losses["edge_type_loss"] = self.criterion_edge_type(outputs["edge_type_preds"], data.edge_types)
       
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

        if "edge_type_preds" in outputs:
            losses["edge_type_loss"] = self.criterion_edge_type(outputs["edge_type_preds"], data.edge_types)
    
        return {k: v.item() for k, v in losses.items()}
