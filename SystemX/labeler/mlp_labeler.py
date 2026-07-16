import logging
from pathlib import Path

import networkx as nx
import torch
from SystemX.labeler._scoring import top2_and_confidence
from SystemX.labeler.edge_labeler import EdgeLabeler
from SystemX.nn.data.v2.feature_extractor import ComputeHubFeatureExtractor, FeatureGroup, infer_feature_groups
from SystemX.nn.data.v2.feature_scaler import FeatureScaler
from SystemX.nn.models.v2.mlp import ComputeHubMLP
from SystemX.sca.constants import REVERSE_DOMAIN_EDGE_TYPES
from SystemX.sca.cdf_ir import CDFIntermediateRepresentation

logger = logging.getLogger(__name__)

class MLPLabeler(EdgeLabeler):
    """MLP baseline labeler for COMPUTE_HUB (CALL) node classification."""

    def __init__(
        self,
        model: ComputeHubMLP,
        extractor: ComputeHubFeatureExtractor | None = None,
        feature_groups: FeatureGroup = FeatureGroup.STANDARD,
        scaler: FeatureScaler | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.extractor = extractor or ComputeHubFeatureExtractor(groups=feature_groups)
        self.scaler = scaler
        logger.info("MLPLabeler: %s (scaler=%s)", self.extractor.description, "on" if scaler else "off")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        feature_groups: FeatureGroup | None = None,
        embedding_model: object | None = None,
    ) -> "MLPLabeler":
        """Build an MLPLabeler from a checkpoint produced by train_mlp."""
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        in_features = state.get("in_features", 302)
        sd = state["model_state_dict"]
        use_bn = state.get("use_batchnorm")
        if use_bn is None:
            use_bn = any(k.endswith("running_mean") for k in sd)
        model = ComputeHubMLP(
            in_features=in_features,
            hidden=state.get("hidden", 256),
            num_classes=state.get("num_classes", None),
            use_batchnorm=bool(use_bn),
        )
        model.load_state_dict(sd)

        scaler = FeatureScaler.from_meta(state)

        if "feature_groups" in state:
            resolved = FeatureGroup(state["feature_groups"])
        elif feature_groups is not None:
            resolved = feature_groups
        else:
            resolved = infer_feature_groups(in_features)
            logger.info("MLPLabeler: inferred feature_groups=%s from in_features=%d", resolved, in_features)

        extractor = (
            ComputeHubFeatureExtractor(embedding=embedding_model, groups=resolved) if embedding_model is not None else None
        )
        return cls(model=model, extractor=extractor, feature_groups=resolved, scaler=scaler)

    def apply_labels(self, graph: CDFIntermediateRepresentation) -> None:
        nx_G = graph.get_graph()
        x, node_ids, _ = self.extractor.extract(nx_G)

        if len(node_ids) == 0:
            logger.info("[MLPLabeler] No COMPUTE_HUB nodes found; skipping.")
            return

        logger.info("[MLPLabeler] Classifying %d COMPUTE_HUB nodes.", len(node_ids))

        if self.scaler is not None:
            x = self.scaler.transform(x)

        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(x, dtype=torch.float32)), dim=1).numpy()

        node_updates: dict = {}
        for node_id, prob_row in zip(node_ids, probs, strict=False):
            label_int, runner_up, conf, margin = top2_and_confidence(prob_row)
            label_str = REVERSE_DOMAIN_EDGE_TYPES.get(label_int, "NOT_RELEVANT")
            node_updates[node_id] = {
                "predicted_label": label_int,
                "domain_label": label_str,
                "predicted_runner_up": runner_up,
                "predicted_confidence": conf,
                "predicted_margin": margin,
            }

        nx.set_node_attributes(nx_G, node_updates)
        logger.info("MLPLabeler labelled %d COMPUTE_HUB nodes.", len(node_updates))
