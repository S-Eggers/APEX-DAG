import ast
import logging
import re

from SystemX.labeling.models import MultiLabelledNode
from SystemX.labeling.vamsa_loader import DomainEdgeId, VamsaEntry
from SystemX.sca.constants import REVERSE_DOMAIN_EDGE_TYPES, canonical_domain_label

logger = logging.getLogger(__name__)

_CALL_SHAPED_TOKEN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

def _attribute_root(node: ast.expr) -> str | None:
    """Returns the leftmost Name of a (possibly chained) attribute/call expression."""
    while True:
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return None

class VamsaKBIndex:
    """O(1) lookup index built from the Vamsa knowledge base."""

    def __init__(self, vamsa_mapping: dict[VamsaEntry, DomainEdgeId], *, allow_unambiguous_fallback: bool = True) -> None:
        self._allow_unambiguous_fallback = allow_unambiguous_fallback
        self._index = self._build(vamsa_mapping)

    def _build(self, vamsa_mapping: dict[VamsaEntry, DomainEdgeId]) -> dict[str, dict[str, DomainEdgeId]]:
        per_library: dict[str, dict[str, set[DomainEdgeId]]] = {}
        all_ids: dict[str, set[DomainEdgeId]] = {}

        for entry, edge_id in vamsa_mapping.items():
            if not entry.api_name:
                continue
            api_name = entry.api_name.rsplit(".", 1)[-1]
            all_ids.setdefault(api_name, set()).add(edge_id)
            if entry.library:
                per_library.setdefault(api_name, {}).setdefault(entry.library, set()).add(edge_id)

        index: dict[str, dict[str, DomainEdgeId]] = {}
        ambiguous_count = 0
        for api_name, edge_ids in all_ids.items():
            index[api_name] = {}
            for library, lib_ids in per_library.get(api_name, {}).items():
                if len(lib_ids) == 1:
                    index[api_name][library] = next(iter(lib_ids))
            if self._allow_unambiguous_fallback and len(edge_ids) == 1:
                index[api_name]["*"] = next(iter(edge_ids))
            else:
                ambiguous_count += 1

        logger.info(
            "Built context-aware KB index. %d APIs require strict provenance resolution.",
            ambiguous_count,
        )
        return index

    def _extract_call(self, code_snippet: str) -> tuple[str | None, str | None]:
        """Extracts (api_name, qualifier_root) of the outermost call in the snippet, e.g."""
        try:
            tree = ast.parse(code_snippet.strip(), mode="exec")
        except (SyntaxError, ValueError):
            return None, None

        for stmt in reversed(tree.body):
            value = getattr(stmt, "value", stmt)
            call = next((n for n in ast.walk(value) if isinstance(n, ast.Call)), None)
            if call is None:
                continue
            if isinstance(call.func, ast.Name):
                return call.func.id, None
            if isinstance(call.func, ast.Attribute):
                return call.func.attr, _attribute_root(call.func.value)
        return None, None

    def match(self, node_id: str, data: dict) -> MultiLabelledNode | None:
        """Returns a MultiLabelledNode if the node's code matches a KB entry, otherwise returns None."""
        code = data.get("code", "")
        if not code:
            return None

        api_name, qualifier = self._extract_call(code)

        if not api_name:
            call_tokens = _CALL_SHAPED_TOKEN.findall(code)
            api_name = next((t for t in reversed(call_tokens) if t in self._index), None)

        if not (api_name and api_name in self._index):
            return None

        mapping = self._index[api_name]
        provenance_tokens = {t.casefold() for t in re.split(r"\W+", str(data.get("base_inputs", ""))) if t}
        if qualifier:
            provenance_tokens.add(qualifier.casefold())

        matched_id = None
        for lib, edge_id in mapping.items():
            if lib != "*" and lib.casefold() in provenance_tokens:
                matched_id = edge_id
                logger.debug("Static KB match for %s: %s resolved via provenance %r", node_id, api_name, lib)
                break

        if matched_id is None and "*" in mapping:
            matched_id = mapping["*"]
            logger.debug("Static KB match for %s: unambiguous signature %s", node_id, api_name)

        if matched_id is None:
            return None

        matched_id = canonical_domain_label(matched_id)
        return MultiLabelledNode(domain_label=REVERSE_DOMAIN_EDGE_TYPES[matched_id])
