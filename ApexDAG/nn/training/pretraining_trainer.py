import os
import tqdm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

global INTERRUPTED
INTERRUPTED = False

class PretrainingTrainer:
    def __init__(self, model, train_dataset, val_dataset, device="cpu", log_dir="runs/", checkpoint_dir="checkpoints/", patience=10):
        self.model = model.to(device)
        self.device = device
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion_node = nn.CrossEntropyLoss()
        self.criterion_edge_type = nn.CrossEntropyLoss()
        self.criterion_edge_existence = nn.BCELoss()
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Early stopping variables
        self.patience = patience
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0

    def save_checkpoint(self, epoch, val_loss, filename=None):
        if filename is None:
            filename = f"model_pretrained_epoch_{epoch}.pt"
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

    def train(self, num_epochs):
        training_bar = tqdm.tqdm(range(num_epochs))
        training_bar.set_description("Training")

        for epoch in training_bar:
            is_interrupted = INTERRUPTED
            if is_interrupted:
                self.save_checkpoint(epoch, self.best_val_loss, "model_interrupted.pt")
                training_bar.write("Training interrupted, last model saved.")
                break

            train_losses = []
            val_losses = []

            for batch in self.train_loader:
                train_losses.append(self.train_step(batch))

            for batch in self.val_loader:
                val_losses.append(self.validate_step(batch))

            avg_train_losses = {k: sum(d[k] for d in train_losses) / len(train_losses) for k in train_losses[0]}
            avg_val_losses = {k: sum(d[k] for d in val_losses) / len(val_losses) for k in val_losses[0]}
            avg_val_loss = sum(avg_val_losses.values())

            for k, v in avg_train_losses.items():
                self.writer.add_scalar(f"Train/{k}", v, epoch)
                wandb.log({f"Train/{k}": v, "epoch": epoch})
            for k, v in avg_val_losses.items():
                self.writer.add_scalar(f"Validation/{k}", v, epoch)
                wandb.log({f"Validation/{k}": v, "epoch": epoch})

            self.log_histograms(epoch)
            if epoch % 10 == 0: 
                example_data = next(iter(self.train_loader)).to(self.device)
                self.log_embeddings(epoch, example_data)

            training_bar.write(f"Train Losses: {avg_train_losses}")
            training_bar.write(f"Val Losses: {avg_val_losses}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch, avg_val_loss)
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.patience:
                training_bar.write("Early stopping triggered!")
                break
        
        self.writer.close()

    def log_histograms(self, epoch):
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, epoch)

    def log_embeddings(self, epoch, data):
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model(data)["node_type_preds"]
            metadata = [f"Node {i}" for i in range(node_embeddings.size(0))]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            global_step = f"{epoch}_{timestamp}"
            self.writer.add_embedding(node_embeddings, metadata=metadata, global_step=global_step, tag="embeddings")
            wandb.log({"embeddings": node_embeddings.cpu().data.numpy(), "epoch": epoch})