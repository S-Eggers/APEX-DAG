import logging
from pathlib import Path

import networkx as nx
import torch
from SystemX.labeler._scoring import top2_and_confidence
from SystemX.labeler.edge_labeler import EdgeLabeler
from SystemX.nn.data.v2.feature_extractor import FeatureGroup, infer_feature_groups
from SystemX.nn.data.v2.pruner import NullPruner
from SystemX.nn.data.v2.tensor_encoder import HETERO_METADATA as _HETERO_METADATA
from SystemX.nn.data.v2.tensor_encoder import EncoderV2
from SystemX.nn.models.v2.gat import SystemXHeteroGraphTransformer
from SystemX.sca.constants import DOMAIN_EDGE_TYPES, REVERSE_DOMAIN_EDGE_TYPES
from SystemX.sca.cdf_ir import CDFIntermediateRepresentation

logger = logging.getLogger(__name__)

class HGTLabeler(EdgeLabeler):
    """Heterogeneous Graph Transformer labeler for COMPUTE_HUB nodes."""

    def __init__(
        self,
        model: SystemXHeteroGraphTransformer,
        encoder: EncoderV2,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        feature_groups: FeatureGroup | None = None,
        embedding_model: object | None = None,
    ) -> "HGTLabeler":
        """Build an HGTLabeler from a checkpoint produced by train_hgt."""
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        model_state = state.get("model_state_dict", state)

        if "feature_groups" in state:
            resolved = FeatureGroup(state["feature_groups"])
        elif feature_groups is not None:
            resolved = feature_groups
        else:
            op_dim = int(model_state["lin_dict.operation.weight"].shape[1]) if "lin_dict.operation.weight" in model_state else 305
            resolved = infer_feature_groups(op_dim)
            logger.info("HGTLabeler: inferred feature_groups=%s from operation dim=%d", resolved, op_dim)

        if embedding_model is None:
            from SystemX.nn.data.v2.fasttext_embedding import FastTextEmbeddingV2

            embedding_model = FastTextEmbeddingV2()

        encoder = EncoderV2(
            embedding_model=embedding_model,
            pruner=NullPruner(),
            feature_groups=resolved,
            rich_variable_features=bool(state.get("rich_var_features", False)),
        )
        model = SystemXHeteroGraphTransformer(
            hidden_channels=int(state.get("hidden_channels", 128)),
            out_classes=len(DOMAIN_EDGE_TYPES),
            num_heads=int(state.get("num_heads", 4)),
            num_layers=int(state.get("num_layers", 3)),
            metadata=_HETERO_METADATA,
        )
        model.load_state_dict(model_state)
        model.eval()
        return cls(model=model, encoder=encoder)

    def apply_labels(self, graph: CDFIntermediateRepresentation, *, explain: bool = False) -> None:
        nx_G = graph.get_graph()

        data, op_index_to_node_id = self.encoder.encode_with_mapping(nx_G)

        n_ops = data["operation"].x.shape[0]
        if n_ops == 0:
            logger.info("[HGTLabeler] No COMPUTE_HUB nodes found; skipping.")
            return

        logger.info("[HGTLabeler] Classifying %d COMPUTE_HUB nodes.", n_ops)

        data = data.to(self.device)

        with torch.no_grad():
            logits = self.model(data.x_dict, data.edge_index_dict)

        probs = torch.softmax(logits, dim=1).cpu().numpy()

        node_updates: dict[object, dict] = {}
        for op_idx, node_id in op_index_to_node_id.items():
            label_int, runner_up, conf, margin = top2_and_confidence(probs[op_idx])
            label_str = REVERSE_DOMAIN_EDGE_TYPES.get(label_int, "NOT_RELEVANT")
            node_updates[node_id] = {
                "predicted_label": label_int,
                "domain_label": label_str,
                "predicted_runner_up": runner_up,
                "predicted_confidence": conf,
                "predicted_margin": margin,
            }

        if explain:
            self._attach_feature_importance(data, op_index_to_node_id, node_updates)

        nx.set_node_attributes(nx_G, node_updates)
        logger.info("HGTLabeler labelled %d COMPUTE_HUB nodes.", len(node_updates))

    def _attach_feature_importance(self, data: object, op_index_to_node_id: dict, node_updates: dict) -> None:
        """Compute per-node input-times-gradient saliency and merge it into node_updates."""
        try:
            from SystemX.labeler.explain import hgt_operation_importance

            importance = hgt_operation_importance(
                self.model, data, op_index_to_node_id, self.encoder.op_extractor
            )
            for node_id, fi in importance.items():
                if node_id in node_updates:
                    node_updates[node_id]["feature_importance"] = fi
        except Exception:
            logger.warning("[HGTLabeler] feature-importance computation failed; skipping.", exc_info=True)
