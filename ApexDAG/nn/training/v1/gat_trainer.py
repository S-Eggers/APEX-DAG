import logging
import os
from datetime import datetime
from enum import Enum

from torch.utils.data import Subset, random_split
from torch.utils.tensorboard import SummaryWriter

from ApexDAG.nn.dataset import GraphDataset
from ApexDAG.nn.training.finetuning_trainer import FinetuningTrainer
from ApexDAG.nn.training.pretraining_trainer import (
    PretrainingTrainer,
    PretrainingTrainerMasked,
)
from ApexDAG.sca.constants import DOMAIN_EDGE_TYPES
from ApexDAG.util.logger import configure_apexdag_logger
from ApexDAG.util.training_utils import DOMAIN_LABEL_TO_SUBSAMPLE, GraphTransformsMode

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class Modes(Enum):
    PRETRAINING = "pretraining"
    FINETUNING = "finetune"


class GATTrainer:
    """Handles model training lifecycle, dataset splitting, and local metric tracking."""

    def __init__(self, config: dict, mode: Modes) -> None:
        self.config = config
        self.mode = mode
        self.subsample_train = config.get("subsample", False)
        self.graph_transform_mode = config.get("mode", "ORIGINAL")
        self.subsample_threshold = self.config.get("subsample_thereshold", 0.999999)
        self.trainer = None

        run_name = f"{self.mode.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(self.config.get("log_dir", "runs"), run_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        logger.info(
            f"TensorBoard initialized. Run 'tensorboard --logdir={self.config.get('log_dir', 'runs')}' to view metrics."
        )

    def split_and_subsample(
        self, dataset: GraphDataset, split_ratio: list, subsample: bool = False
    ):
        train_dataset, val_dataset = random_split(dataset, split_ratio)

        if subsample and self.mode == Modes.FINETUNING:
            index_to_subsample = DOMAIN_EDGE_TYPES[DOMAIN_LABEL_TO_SUBSAMPLE]

            if self.graph_transform_mode in [
                GraphTransformsMode.REVERSED,
                GraphTransformsMode.REVERSED_MASKED,
            ]:
                filtered_indices = [
                    idx
                    for idx in train_dataset.indices
                    if sum(dataset[idx].node_types == index_to_subsample)
                    / len(dataset[idx].node_types)
                    < self.subsample_threshold
                ]
            elif self.graph_transform_mode in [
                GraphTransformsMode.ORIGINAL,
                GraphTransformsMode.ORIGINAL_MASKED,
            ]:
                filtered_indices = [
                    idx
                    for idx in train_dataset.indices
                    if sum(dataset[idx].edge_types == index_to_subsample)
                    / len(dataset[idx].edge_types)
                    < self.subsample_threshold
                ]
            else:
                filtered_indices = train_dataset.indices

            train_dataset = Subset(dataset, filtered_indices)

        return train_dataset, val_dataset

    def train(
        self,
        encoded_graphs,
        model,
        device: str = "cuda",
        graph_transform_mode: str = GraphTransformsMode.ORIGINAL,
    ):
        try:
            sample_data = encoded_graphs[0].to(device)
            self.writer.add_graph(model, sample_data)
        except Exception as e:
            logger.warning(f"Could not log model graph to TensorBoard: {e}")

        dataset = GraphDataset(encoded_graphs)

        if self.mode == Modes.PRETRAINING:
            logger.info("Training in pretraining mode")
            train_size = int(self.config["train_split"] * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            TrainerClass = (
                PretrainingTrainerMasked
                if graph_transform_mode
                in [
                    GraphTransformsMode.REVERSED_MASKED,
                    GraphTransformsMode.ORIGINAL_MASKED,
                ]
                else PretrainingTrainer
            )

            self.trainer = TrainerClass(
                model,
                train_dataset,
                val_dataset,
                device=device,
                patience=self.config["patience"],
                batch_size=self.config["batch_size"],
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
                graph_transform_mode=graph_transform_mode,
                logger=logger,
                writer=self.writer,  # INJECT TENSORBOARD WRITER
            )

        elif self.mode == Modes.FINETUNING:
            logger.info("Training in finetuning mode")
            train_size = int(self.config["train_split"] * len(dataset))
            test_size = int(self.config["test_split"] * len(dataset))
            val_size = len(dataset) - train_size - test_size

            train_dataset, val_dataset = self.split_and_subsample(
                dataset,
                [train_size, val_size + test_size],
                subsample=self.subsample_train,
            )
            val_dataset, test_dataset = random_split(val_dataset, [val_size, test_size])

            self.trainer = FinetuningTrainer(
                model,
                train_dataset,
                val_dataset,
                test_dataset,
                device=device,
                patience=self.config["patience"],
                batch_size=self.config["batch_size"],
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
                graph_transform_mode=graph_transform_mode,
                logger=logger,
                writer=self.writer,  # INJECT TENSORBOARD WRITER
            )

        best_loss = self.trainer.train(num_epochs=self.config["num_epochs"])

        for type_conf_matrix in self.trainer.conf_matrices_types:
            self.trainer.log_confusion_matrix(train_dataset, "Train", type_conf_matrix)
            self.trainer.log_confusion_matrix(val_dataset, "Val", type_conf_matrix)
            if self.mode == Modes.FINETUNING:
                self.trainer.log_confusion_matrix(
                    test_dataset, "Test", type_conf_matrix
                )

        self.writer.close()

        return best_loss
