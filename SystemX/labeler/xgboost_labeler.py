import logging
from pathlib import Path

import networkx as nx
from SystemX.labeler._scoring import top2_and_confidence
from SystemX.labeler.edge_labeler import EdgeLabeler
from SystemX.nn.data.v2.feature_extractor import ComputeHubFeatureExtractor, FeatureGroup, infer_feature_groups
from SystemX.sca.constants import REVERSE_DOMAIN_EDGE_TYPES
from SystemX.sca.cdf_ir import CDFIntermediateRepresentation

logger = logging.getLogger(__name__)

class XGBoostLabeler(EdgeLabeler):
    """XGBoost baseline labeler for COMPUTE_HUB (CALL) node classification."""

    def __init__(
        self,
        model: object,
        extractor: ComputeHubFeatureExtractor | None = None,
        feature_groups: FeatureGroup = FeatureGroup.STANDARD,
    ) -> None:
        self.model = model
        self.extractor = extractor or ComputeHubFeatureExtractor(groups=feature_groups)
        logger.info("XGBoostLabeler: %s", self.extractor.description)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        feature_groups: FeatureGroup | None = None,
        embedding_model: object | None = None,
    ) -> "XGBoostLabeler":
        """Build an XGBoostLabeler from a checkpoint produced by train_xgboost."""
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost is required: pip install xgboost") from e

        model = xgb.XGBClassifier()
        model.load_model(str(checkpoint_path))

        if feature_groups is not None:
            resolved = feature_groups
        else:
            n_feat = int(model.n_features_in_)
            resolved = infer_feature_groups(n_feat)
            logger.info(
                "XGBoostLabeler: inferred feature_groups=%s from n_features_in_=%d",
                resolved,
                n_feat,
            )

        extractor = (
            ComputeHubFeatureExtractor(embedding=embedding_model, groups=resolved) if embedding_model is not None else None
        )
        return cls(model=model, extractor=extractor, feature_groups=resolved)

    def apply_labels(self, graph: CDFIntermediateRepresentation, *, explain: bool = False) -> None:
        nx_G = graph.get_graph()
        x, node_ids, _ = self.extractor.extract(nx_G)

        if len(node_ids) == 0:
            logger.info("[XGBoostLabeler] No COMPUTE_HUB nodes found; skipping.")
            return

        logger.info("[XGBoostLabeler] Classifying %d COMPUTE_HUB nodes.", len(node_ids))

        probs = self.model.predict_proba(x)

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

        if explain:
            self._attach_feature_importance(x, node_ids, node_updates)

        nx.set_node_attributes(nx_G, node_updates)
        logger.info("XGBoostLabeler labelled %d COMPUTE_HUB nodes.", len(node_updates))

    def _attach_feature_importance(self, x: object, node_ids: list, node_updates: dict) -> None:
        """Compute native per-instance (tree-SHAP) importance and merge it in."""
        try:
            from SystemX.labeler.explain import xgboost_row_importance

            importance = xgboost_row_importance(self.model, x, node_ids, self.extractor)
            for node_id, fi in importance.items():
                if node_id in node_updates:
                    node_updates[node_id]["feature_importance"] = fi
        except Exception:
            logger.warning("[XGBoostLabeler] feature-importance computation failed; skipping.", exc_info=True)
