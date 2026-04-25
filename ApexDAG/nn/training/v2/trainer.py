import os
import torch
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)

class ApexOnlineTrainer:
    """
    Stateful V2 Trainer. 
    Maintains optimizer momentum and loss uncertainty parameters in memory 
    so it can be incrementally updated via an online labeling pipeline.
    """
    def __init__(self, model, criterion, learning_rate=1e-4, log_dir="runs/v2_experiment"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.parameters(), 'lr': learning_rate},
            {'params': self.criterion.parameters(), 'lr': learning_rate * 10} 
        ], weight_decay=1e-4)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, timestamp))
        self.global_step = 0
        
        logger.info(f"Trainer initialized on {self.device}. Logs at {log_dir}/{timestamp}")

    def log_model_graph(self, sample_data):
        sample_data = sample_data.to(self.device)
        try:
            self.writer.add_graph(
                self.model, 
                (sample_data.x, sample_data.edge_index, sample_data.edge_attr)
            )
            logger.info("Successfully traced model graph to TensorBoard.")
        except Exception as e:
            logger.warning(f"TensorBoard graph tracing failed (common with PyG dynamic graphs): {e}")

    def train_step(self, batch_data):
        self.model.train()
        batch_data = batch_data.to(self.device)
        
        self.optimizer.zero_grad()
        
        node_logits, edge_logits = self.model(
            batch_data.x, 
            batch_data.edge_index, 
            batch_data.edge_attr
        )
        
        loss, loss_node, loss_edge = self.criterion(
            node_logits, batch_data.y_node,
            edge_logits, batch_data.y_edge
        )
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()
        
        self._log_metrics(loss.item(), loss_node, loss_edge, node_logits, edge_logits)
        
        self.global_step += 1
        return loss.item()

    def _log_metrics(self, total_loss, node_loss, edge_loss, node_logits, edge_logits):
        """Logs scalars, network uncertainty, and gradient histograms."""
        
        self.writer.add_scalar('Loss/Total', total_loss, self.global_step)
        self.writer.add_scalar('Loss/Node_Classification', node_loss, self.global_step)
        self.writer.add_scalar('Loss/Edge_Classification', edge_loss, self.global_step)
        
        log_vars = self.criterion.log_vars.detach().cpu()
        self.writer.add_scalar('Uncertainty/Node_LogVar', log_vars[0], self.global_step)
        self.writer.add_scalar('Uncertainty/Edge_LogVar', log_vars[1], self.global_step)
        
        node_preds = torch.argmax(node_logits, dim=1).float()
        edge_preds = torch.argmax(edge_logits, dim=1).float()
        self.writer.add_histogram('Predictions/Node_Classes', node_preds, self.global_step)
        self.writer.add_histogram('Predictions/Edge_Classes', edge_preds, self.global_step)

    def save_cpu_checkpoint(self, path: str):
        """
        Strips optimizer state and moves model to CPU for deployment inside the Jupyter Extension.
        """
        self.model.eval()
        cpu_model = self.model.to('cpu')
        torch.save(cpu_model.state_dict(), path)
        self.model.to(self.device)
        logger.info(f"Deployable CPU checkpoint saved to {path}")