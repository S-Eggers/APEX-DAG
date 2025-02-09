import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter


class PretrainingTrainer:
    def __init__(self, model, train_dataset, val_dataset, device="cpu", log_dir="runs/"):
        self.model = model.to(device)
        self.device = device
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion_node = nn.CrossEntropyLoss()
        self.criterion_edge_type = nn.CrossEntropyLoss()
        self.criterion_edge_existence = nn.BCELoss()

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)

        # Forward pass
        outputs = self.model(data)

        # Compute losses
        losses = {}
        if "node_type_preds" in outputs:
            losses["node_type_loss"] = self.criterion_node(outputs["node_type_preds"], data.node_types)
        if "edge_type_preds" in outputs:
            losses["edge_type_loss"] = self.criterion_edge_type(outputs["edge_type_preds"], data.edge_types)
        if "edge_existence_preds" in outputs:
            edge_existence_preds = outputs["edge_existence_preds"].squeeze(dim=-1)  # Ensure correct shape
            edge_existence_targets = data.edge_existence.float()

            losses["edge_existence_loss"] = self.criterion_edge_existence(edge_existence_preds, edge_existence_targets)


        # Aggregate losses
        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def validate_step(self, data):
        self.model.eval()
        data = data.to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(data)

        # Compute losses
        losses = {}
        if "node_type_preds" in outputs:
            losses["node_type_loss"] = self.criterion_node(outputs["node_type_preds"], data.node_types)
        if "edge_type_preds" in outputs:
            losses["edge_type_loss"] = self.criterion_edge_type(outputs["edge_type_preds"], data.edge_types)
        if "edge_existence_preds" in outputs:
            edge_existence_preds = outputs["edge_existence_preds"].squeeze(dim=-1)  # Ensure correct shape
            edge_existence_targets = data.edge_existence.float()  # Ensure matching type

            losses["edge_existence_loss"] = self.criterion_edge_existence(edge_existence_preds, edge_existence_targets)


        return {k: v.item() for k, v in losses.items()}

    def train(self, num_epochs):
        training_bar = tqdm.tqdm(range(num_epochs))
        training_bar.set_description("Training")
        
        for epoch in training_bar:
            train_losses = []
            val_losses = []

            # Training loop
            for batch in self.train_loader:
                train_losses.append(self.train_step(batch))

            # Validation loop
            for batch in self.val_loader:
                val_losses.append(self.validate_step(batch))

            # Compute average losses
            avg_train_losses = {k: sum(d[k] for d in train_losses) / len(train_losses) for k in train_losses[0]}
            avg_val_losses = {k: sum(d[k] for d in val_losses) / len(val_losses) for k in val_losses[0]}

            # Log losses to TensorBoard
            for k, v in avg_train_losses.items():
                self.writer.add_scalar(f"Train/{k}", v, epoch)
            for k, v in avg_val_losses.items():
                self.writer.add_scalar(f"Validation/{k}", v, epoch)

            # Advanced logging
            self.log_histograms(epoch)
            if epoch % 10 == 0:  # Log embeddings periodically
                example_data = next(iter(self.train_loader)).to(self.device)
                self.log_embeddings(epoch, example_data)

            training_bar.write(f"Train Losses: {avg_train_losses}")
            training_bar.write(f"Val Losses: {avg_val_losses}")

        # Close TensorBoard writer
        self.writer.close()

    def log_histograms(self, epoch):
        """
        Logs histograms of model parameters to TensorBoard.
        """
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, epoch)

    def log_graph(self, example_data):
        """
        Logs the model graph to TensorBoard.
        """
        print(f"Logging graph with data type: {type(example_data)}")

        # Extract only tensor attributes to pass to add_graph
        example_input = (example_data.x, example_data.edge_index)

        self.writer.add_graph(self.model, example_input)

    def log_embeddings(self, epoch, data):
        """
        Logs embeddings (e.g., node embeddings) to TensorBoard.

        Args:
            epoch (int): Current epoch number.
            data (Data): Example graph data.
        """
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model(data)["node_type_preds"]
            metadata = [f"Node {i}" for i in range(node_embeddings.size(0))]
            
            self.writer.add_embedding(
                node_embeddings, metadata=metadata, global_step=epoch, tag=f"embedding_{epoch}"
            )
