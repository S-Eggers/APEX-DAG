import os
import torch
import wandb
from ApexDAG.nn.training.base_trainer import BaseTrainer
from torch_geometric.loader import DataLoader


class FinetuningTrainer(BaseTrainer):
    def __init__(self, model, train_dataset, val_dataset, test_dataset, batch_size = 32, **kwargs):
        super().__init__(model, train_dataset, val_dataset, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.conf_matrices_types = ["node_type_preds"]
        self.best_losses = {
            "node_type_loss": float("inf")
        }

    def save_checkpoint(self, epoch, val_loss, filename=None):
        if filename is None:
            filename = f"model_epoch_finetuned_masked_{epoch}.pt"
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
        

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)
        outputs = self.model(data)

        losses = {}
        
        if "node_type_preds" in outputs:
            valid_mask = data.node_types != -1
            node_types = data.node_types[valid_mask]
            node_type_preds = outputs["node_type_preds"][valid_mask]
            losses["node_type_loss"] = self.criterion_node(node_type_preds, node_types)

       
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
            valid_mask = data.node_types != -1
            node_types = data.node_types[valid_mask]
            node_type_preds = outputs["node_type_preds"][valid_mask]
            
            losses["node_type_loss"] = self.criterion_node(node_type_preds, node_types)

        return {k: v.item() for k, v in losses.items()}
