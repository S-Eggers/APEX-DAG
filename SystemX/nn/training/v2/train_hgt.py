import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from SystemX.nn.data.v2.feature_extractor import FEATURE_PRESETS
from SystemX.nn.data.v2.pruner import NullPruner
from SystemX.nn.data.v2.tensor_encoder import EncoderV2
from SystemX.nn.models.v2.gat import SystemXHeteroGraphTransformer
from SystemX.nn.training.v2.data_utils import annotation_to_networkx as _annotation_to_networkx
from SystemX.nn.training.v2.trainer import SystemXOnlineTrainer
from SystemX.sca.constants import COMPUTE_HUBS, DOMAIN_EDGE_TYPES
from SystemX.util.logger import configure_systemx_logger

configure_systemx_logger()
logger = logging.getLogger("SystemX.nn.training.v2.train_hgt")

from SystemX.nn.data.v2.tensor_encoder import HETERO_METADATA as _HETERO_METADATA

def _load_annotation_paths(annotations_dir: Path, train_list: Path | None) -> list[Path]:
    """Resolve the training annotation files, optionally restricted to a fold's train split listed (one filename per line) in train_list."""
    if train_list is not None and train_list.exists():
        names = [ln.strip() for ln in train_list.read_text().splitlines() if ln.strip()]
        return [annotations_dir / n for n in names]
    return sorted(annotations_dir.glob("*.json"))

def _compute_class_weights(ann_paths: list[Path], num_classes: int, mode: str, max_weight: float = 5.0) -> torch.Tensor | None:
    """Class weights from the training split, to counter majority-class collapse on the imbalanced 7-class problem."""
    if mode == "none":
        return None

    counts = [0] * num_classes
    for ann_path in ann_paths:
        try:
            with open(ann_path, encoding="utf-8") as f:
                raw = json.load(f)
            elements = raw if isinstance(raw, list) else raw.get("elements", [])
            nx_G = _annotation_to_networkx(elements)
        except Exception:
            continue
        for _, attrs in nx_G.nodes(data=True):
            if int(attrs.get("node_type", -1)) not in COMPUTE_HUBS:
                continue
            lbl = int(attrs.get("domain_label", -1))
            if 0 <= lbl < num_classes:
                counts[lbl] += 1

    total = sum(counts)
    if total == 0:
        logger.warning("No labelled CALL nodes found; class weighting disabled.")
        return None

    raw_w = [(total / (num_classes * c)) if c > 0 else 1.0 for c in counts]
    if mode == "sqrt_inverse":
        raw_w = [w**0.5 for w in raw_w]

    mean_w = sum(raw_w) / len(raw_w)
    weights = [min(w / mean_w, max_weight) for w in raw_w]
    logger.info("Class counts: %s", counts)
    logger.info("Class weights (%s, normalized, clip<=%.1f): %s", mode, max_weight, [round(w, 3) for w in weights])
    return torch.tensor(weights, dtype=torch.float32)

def _build_embedding(name: str) -> object:
    if name == "fasttext":
        from SystemX.nn.data.v2.fasttext_embedding import FastTextEmbeddingV2

        return FastTextEmbeddingV2()
    if name in ("codebert", "graphcodebert"):
        from SystemX.nn.data.v2.embedding import TransformerEmbedding

        model_name = "microsoft/codebert-base" if name == "codebert" else "microsoft/graphcodebert-base"
        return TransformerEmbedding(model_name=model_name)
    raise ValueError(f"Unknown embedding: {name!r}. Choose fasttext, codebert, or graphcodebert.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Train HGT labeler (V2)")
    parser.add_argument("--annotations_dir", required=True)
    parser.add_argument("--output_dir", default="./checkpoints/v2")
    parser.add_argument("--embedding", default="fasttext", choices=["fasttext", "codebert", "graphcodebert"])
    parser.add_argument("--features", default="standard", choices=list(FEATURE_PRESETS), help="Operation-node feature preset (same presets as the MLP/XGBoost baselines)")
    parser.add_argument("--class_weighting", default="sqrt_inverse", choices=["none", "inverse", "balanced", "sqrt_inverse"], help="Class-weighted loss vs majority-class collapse")
    parser.add_argument("--max_class_weight", type=float, default=5.0, help="Cap on per-class weight (prevents a rare class from dominating the loss)")
    parser.add_argument("--rich_var_features", action="store_true", help="Variable nodes carry their producing-op embedding + degree (lineage signal for message passing)")
    parser.add_argument("--train_list", default=None, help="Optional file listing the fold's training annotation filenames (one per line)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16, help="Graphs per optimizer step (mini-batching; main CPU speedup)")
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (1e-3 best with mini-batching; batched gradients tolerate a larger step)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_groups = FEATURE_PRESETS[args.features]
    embedding = _build_embedding(args.embedding)
    encoder = EncoderV2(embedding_model=embedding, pruner=NullPruner(), feature_groups=feature_groups, rich_variable_features=args.rich_var_features)
    num_classes = len(DOMAIN_EDGE_TYPES)

    model = SystemXHeteroGraphTransformer(
        hidden_channels=args.hidden_channels,
        out_classes=num_classes,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        metadata=_HETERO_METADATA,
    )

    train_list = Path(args.train_list) if args.train_list else None
    ann_paths = _load_annotation_paths(Path(args.annotations_dir), train_list)
    logger.info("Found %d annotation files (features=%s, class_weighting=%s).", len(ann_paths), args.features, args.class_weighting)

    class_weights = _compute_class_weights(ann_paths, num_classes, args.class_weighting, args.max_class_weight)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    trainer = SystemXOnlineTrainer(model=model, criterion=criterion, learning_rate=args.lr)

    from torch_geometric.loader import DataLoader

    encoded: list = []
    for ann_path in ann_paths:
        try:
            with open(ann_path, encoding="utf-8") as f:
                raw = json.load(f)
            elements = raw if isinstance(raw, list) else raw.get("elements", [])
            nx_G = _annotation_to_networkx(elements)
            if nx_G.number_of_nodes() == 0:
                continue
            data = encoder.encode(nx_G)
            if data["operation"].train_mask.sum() == 0:
                continue
            encoded.append(data)
        except Exception as e:
            logger.warning("Skipping %s: %s", ann_path.name, e)

    if not encoded:
        logger.error("No trainable graphs after encoding. Aborting.")
        return
    logger.info("Pre-encoded %d trainable graphs. Training with batch_size=%d.", len(encoded), args.batch_size)
    trainer.log_model_graph(encoded[0])

    for epoch in range(1, args.epochs + 1):
        loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True)
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            epoch_loss += trainer.train_step(batch)
            n_batches += 1
        logger.info("Epoch %d/%d  loss=%.4f  batches=%d", epoch, args.epochs, epoch_loss / max(n_batches, 1), n_batches)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = output_dir / f"hgt_{args.embedding}_{args.features}_{timestamp}.pt"
    trainer.save_cpu_checkpoint(
        str(checkpoint_path),
        meta={
            "hidden_channels": args.hidden_channels,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "embedding": args.embedding,
            "features": args.features,
            "feature_groups": feature_groups.value,
            "class_weighting": args.class_weighting,
            "rich_var_features": args.rich_var_features,
        },
    )
    logger.info("Checkpoint saved to %s", checkpoint_path)

if __name__ == "__main__":
    main()
