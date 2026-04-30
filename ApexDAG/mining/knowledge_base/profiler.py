import ast
import json
import logging
import os
from collections import Counter

from tqdm import tqdm

from ApexDAG.util.logger import configure_apexdag_logger
from ApexDAG.vamsa.extraction import GenWIR
from ApexDAG.vamsa.lineage import KB, AnnotationWIR

from .models import CachedNotebook

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class CorpusProfiler:
    def __init__(self, corpus_path: str) -> None:
        self.corpus_path = corpus_path
        self._cache: list[CachedNotebook] = []
        self._missing_operations_counter = Counter()

    def get_cache(self) -> list[CachedNotebook]:
        return self._cache

    def _read_code(self, path: str) -> str:
        if path.endswith(".ipynb"):
            with open(path, encoding="utf-8") as f:
                nb = json.load(f)
                return "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []) if cell.get("cell_type") == "code")
        with open(path, encoding="utf-8") as f:
            return f.read()

    def build_cache_and_profile(self, baseline_kb: KB) -> None:
        files = [os.path.join(root, f) for root, _, filenames in os.walk(self.corpus_path) for f in filenames if f.endswith((".py", ".ipynb"))]

        logger.info(f"Phase 1: Parsing {len(files)} files and building memory cache...")

        for filepath in tqdm(files, desc="Caching WIRs"):
            try:
                code = self._read_code(filepath)
                if not code.strip():
                    continue

                ast_tree = ast.parse(code)
                G, prs, _ = GenWIR(ast_tree)

                self._cache.append(CachedNotebook(filepath, G, prs))

                annotator = AnnotationWIR(G.copy(), prs, baseline_kb)
                annotated_g = annotator.annotate()

                for pr in prs:
                    inputs, _, operation_node, outputs = pr
                    if not operation_node:
                        continue

                    op_name = str(operation_node).split(":")[0]
                    if not op_name or op_name.startswith(("Assign", "Call", "Attribute")):
                        continue

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

                    if not is_annotated:
                        self._missing_operations_counter[op_name] += 1

            except Exception as e:
                logger.debug(f"Failed to parse {filepath}: {e}")

        logger.info(f"Identified {len(self._missing_operations_counter)} unique unannotated operations.")

    def get_top_missing(self, limit: int = 50) -> list[tuple[str, int]]:
        return self._missing_operations_counter.most_common(limit)
