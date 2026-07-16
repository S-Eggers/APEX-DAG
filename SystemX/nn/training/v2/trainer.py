import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

class SystemXOnlineTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        learning_rate: float = 1e-4,
        log_dir: str = "runs/v2_experiment",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": learning_rate},
                {"params": self.criterion.parameters(), "lr": learning_rate * 10},
            ],
            weight_decay=1e-4,
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, timestamp))
        self.global_step = 0

        logger.info(f"Trainer initialized on {self.device}. Logs at {log_dir}/{timestamp}")

    def log_model_graph(self, sample_data: HeteroData) -> None:
        sample_data = sample_data.to(self.device)
        try:
            self.writer.add_graph(
                self.model,
                (sample_data.x_dict, sample_data.edge_index_dict),
            )
            logger.info("Successfully traced GNN model graph to TensorBoard.")
        except Exception as e:
            logger.warning(f"TensorBoard graph tracing failed: {e}")

    def train_step(self, batch_data: HeteroData) -> float:
        self.model.train()
        batch_data = batch_data.to(self.device)

        self.optimizer.zero_grad()

        op_logits = self.model(batch_data.x_dict, batch_data.edge_index_dict)

        mask = batch_data["operation"].train_mask
        masked_logits = op_logits[mask]
        masked_targets = batch_data["operation"].y[mask]

        if masked_targets.numel() == 0:
            return 0.0

        loss = self.criterion(masked_logits, masked_targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()

        self._log_metrics(loss.item(), masked_logits, masked_targets)

        self.global_step += 1
        return loss.item()

    def _log_metrics(
        self,
        total_loss: float,
        op_logits: torch.Tensor,
        op_targets: torch.Tensor,
    ) -> None:
        self.writer.add_scalar("Loss/Total", total_loss, self.global_step)

        op_preds = torch.argmax(op_logits, dim=1).float()
        self.writer.add_histogram("Predictions/Op_Classes", op_preds, self.global_step)
        self.writer.add_histogram("Targets/Op_Classes", op_targets.float(), self.global_step)

    def save_cpu_checkpoint(self, path: str, meta: dict | None = None) -> None:
        self.model.eval()
        cpu_model = self.model.to("cpu")
        payload: dict = {"model_state_dict": cpu_model.state_dict()}
        if meta:
            payload.update(meta)
        torch.save(payload, path)
        self.model.to(self.device)
        logger.info("GNN checkpoint saved to %s", path)

class MLPTrainer:
    """Trainer for ComputeHubMLP with TensorBoard integration."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        learning_rate: float = 1e-3,
        log_dir: str = "runs/v2_mlp",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, timestamp))
        self.global_step = 0

        logger.info("MLPTrainer initialized on %s. Logs at %s/%s", self.device, log_dir, timestamp)

    def log_model_graph(self, sample_x: torch.Tensor) -> None:
        """Trace the MLP computational graph to TensorBoard."""
        sample_x = sample_x[:1].to(self.device)
        try:
            self.writer.add_graph(self.model, sample_x)
            logger.info("Successfully traced MLP model graph to TensorBoard.")
        except Exception as e:
            logger.warning("TensorBoard MLP graph tracing failed: %s", e)

    def train_epoch(self, x: torch.Tensor, y: torch.Tensor, epoch: int) -> tuple[float, float]:
        """Run one full epoch on the provided tensors."""
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()

        self._log_metrics(loss.item(), acc, epoch)
        self.global_step += 1
        return loss.item(), acc

    def _log_metrics(self, loss: float, acc: float, epoch: int) -> None:
        self.writer.add_scalar("Loss/Train", loss, epoch)
        self.writer.add_scalar("Accuracy/Train", acc, epoch)

        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().norm().item() ** 2
        self.writer.add_scalar("GradNorm", total_norm**0.5, epoch)

    def save_cpu_checkpoint(self, path: str, meta: dict | None = None) -> None:
        self.model.eval()
        cpu_model = self.model.to("cpu")
        payload = {"model_state_dict": cpu_model.state_dict()}
        if meta:
            payload.update(meta)
        torch.save(payload, path)
        self.model.to(self.device)
        logger.info("MLP checkpoint saved to %s", path)
