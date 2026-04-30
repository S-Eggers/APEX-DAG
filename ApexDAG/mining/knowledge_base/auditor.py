import logging

from tqdm import tqdm

from ApexDAG.util.logger import configure_apexdag_logger
from ApexDAG.vamsa.lineage import KB, AnnotationWIR

from .models import CachedNotebook

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class KBAuditor:
    """Evaluates a KB against the memory cache in seconds."""

    def __init__(self, cache: list[CachedNotebook]) -> None:
        self.cache = cache

    def evaluate(self, kb: KB) -> float:
        total_ops = 0
        annotated_ops = 0

        logger.info(f"Auditing KB against {len(self.cache)} cached notebooks...")

        # Wrapped the cache loop in tqdm for observability
        for nb in tqdm(self.cache, desc="Evaluating KB Coverage"):
            G_eval = nb.base_graph.copy()
            annotator = AnnotationWIR(G_eval, nb.prs, kb)
            annotated_g = annotator.annotate()

            for pr in nb.prs:
                inputs, _, operation_node, outputs = pr
                if not operation_node:
                    continue

                op_name = str(operation_node).split(":")[0]
                if not op_name or op_name.startswith(("Assign", "Call", "Attribute")):
                    continue

                total_ops += 1
                is_annotated = False

                out_nodes = outputs if isinstance(outputs, list) else [outputs]
                for out in out_nodes:
                    if out and annotated_g.nodes[out].get("annotations"):
                        is_annotated = True
                        break

                if not is_annotated:
                    in_nodes = inputs if isinstance(inputs, list) else [inputs]
                    for inp in in_nodes:
                        if inp and annotated_g.nodes[inp].get("annotations"):
                            is_annotated = True
                            break

                if is_annotated:
                    annotated_ops += 1

        return (annotated_ops / total_ops) if total_ops > 0 else 0.0
