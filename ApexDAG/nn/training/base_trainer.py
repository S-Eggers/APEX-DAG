import os
import tqdm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from ApexDAG.util.training_utils import GraphTransformsMode

class BaseTrainer:
    def __init__(self, model, train_dataset, val_dataset, device="cuda", log_dir="runs/", checkpoint_dir="checkpoints/", patience=10, batch_size=32, lr=0.001, weight_decay=0.00001, graph_transform_mode = GraphTransformsMode.ORIGINAL):
        self.model = model.to(device)
        self.device = device
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.criterion_node = nn.CrossEntropyLoss()
        self.criterion_edge_type = nn.CrossEntropyLoss()
        self.criterion_edge_existence = nn.BCELoss()
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.patience = patience
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0
        
        self.conf_matrices_types = ["edge_type_preds"] # defined in subclasses
        self.graph_transform_mode = graph_transform_mode

    def save_checkpoint(self, epoch, val_loss, suffix_name = "", filename=None):
        if filename is None:
            filename = f"model_epoch_{suffix_name}_{epoch}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        

    def log_confusion_matrix(self, loader, phase, pred_type = "edge_type_preds"):
        self.model.eval().to(self.device)
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in  tqdm.tqdm(loader, desc="Processing Data for conf matrix", leave=False):
                data = data.to(self.device)
                outputs = self.model(data)
                
                if pred_type not in outputs:
                    self.logger.warning(f"Prediction type '{pred_type}' not found in model outputs. Skipping.")
                    return  # Skip logging for this pred_type
                
                if pred_type == "edge_type_preds":
                    labels = data.edge_types
                    preds = torch.argmax(outputs[pred_type], dim=1)
                    # if the label is -1 then omit  it with mask
                    valid_edge_mask = labels != -1
                    preds = preds[valid_edge_mask]
                    labels = labels[valid_edge_mask]
                elif pred_type == "edge_existence_preds":
                    labels = data.edge_existence
                    preds = (outputs[pred_type] > 0.5).astype(int)
                elif pred_type == "node_type_preds":
                    preds = torch.argmax(outputs[pred_type], dim=1)
                    labels = data.node_types
                
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"{phase} Confusion Matrix ({pred_type})")

        cm_path = f"{self.checkpoint_dir}/{phase}_conf_matrix_epoch.png"
        plt.savefig(cm_path)
        plt.close()

        wandb.log({f"{phase}/Confusion_Matrix_{pred_type}": wandb.Image(cm_path)})

    def train_step(self, data):
        raise NotImplementedError("train_step should be implemented in subclasses")

    def validate_step(self, data):
        raise NotImplementedError("validate_step should be implemented in subclasses")

    def train(self, num_epochs):
        training_bar = tqdm.tqdm(range(num_epochs))
        training_bar.set_description("Training")
        
        best_losses = {
            "node_type_loss": float("inf"),
            "edge_type_loss": float("inf"),
            "reconstruction_loss": float("inf"),
            "edge_existence_loss": float("inf")
        }
        best_losses_table = wandb.Table(columns=["Epoch", "Best_Node_Loss", "Best_Edge_Type_Loss", "Best_Reconstruction_Loss", "Best_Edge_Existence_Loss"])


        for epoch in training_bar:
            train_losses = []
            val_losses = []

            for batch in tqdm.tqdm(self.train_loader, desc="Training Batches"):
                train_losses.append(self.train_step(batch))

            for batch in self.val_loader:
                val_losses.append(self.validate_step(batch))

            avg_train_losses = {k: sum(d[k] for d in train_losses) / len(train_losses) for k in train_losses[0]}
            avg_val_losses = {k: sum(d[k] for d in val_losses) / len(val_losses) for k in val_losses[0]}
            avg_val_loss = sum(avg_val_losses.values())

            for k, v in avg_train_losses.items():
                self.writer.add_scalar(f"Train/{k}", v, epoch)
                wandb.log({f"Train/{k}": v, "epoch": epoch}, step = epoch)
            for k, v in avg_val_losses.items():
                self.writer.add_scalar(f"Validation/{k}", v, epoch)
                wandb.log({f"Validation/{k}": v, "epoch": epoch}, step = epoch)

            self.log_histograms(epoch)

            if epoch % 10 == 0:
                example_data = next(iter(self.train_loader)).to(self.device)
                self.log_embeddings(epoch, example_data)

            training_bar.write(f"Train Losses: {avg_train_losses}")
            training_bar.write(f"Val Losses: {avg_val_losses}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                for loss_type in best_losses:
                    best_losses[loss_type] = avg_val_losses[loss_type]
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch, avg_val_loss, suffix_name=self.graph_transform_mode)
    
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.patience:
                training_bar.write("Early stopping triggered!")
                best_losses_table.add_data(
                    epoch,
                    best_losses["node_type_loss"],
                    best_losses["edge_type_loss"],
                    best_losses["reconstruction_loss"],
                    best_losses["edge_existence_loss"]
                )
                wandb.log({"Best_Losses": best_losses_table})
                break
        
        self.writer.close()
        best_losses_table.add_data(
                    epoch,
                    best_losses["node_type_loss"],
                    best_losses["edge_type_loss"],
                    best_losses["reconstruction_loss"],
                    best_losses["edge_existence_loss"]
                )
        wandb.log({"Best_Losses": best_losses_table})
        return self.best_val_loss

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


