import importlib.metadata as importlib_metadata
import sys
from typing import Any

def _resolve_versions(declared: list[tuple[str, str | None]]) -> dict[str, str | None]:
    """Best-effort installed version for each imported top-level package."""
    try:
        dist_map = importlib_metadata.packages_distributions()
    except Exception:
        dist_map = {}
    stdlib = getattr(sys, "stdlib_module_names", frozenset())

    versions: dict[str, str | None] = {}
    for full_path, _alias in declared:
        top = str(full_path).split(".")[0]
        if not top or top in versions:
            continue
        if top in stdlib:
            versions[top] = "stdlib"
            continue
        resolved: str | None = None
        for dist in dist_map.get(top, [top]):
            try:
                resolved = importlib_metadata.version(dist)
                break
            except Exception:
                resolved = None
        versions[top] = resolved
    return versions

class EnvironmentSerializer:
    def to_dict(
        self,
        import_visitor: object,
        complexity_visitor: object,
        runtime: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Transforms visitor state into the final API contract."""
        sanitized_usage = {module: dict(usages) for module, usages in import_visitor.import_usage.items()}

        return {
            "imports": {
                "declared": import_visitor.imports,
                "usage": sanitized_usage,
                "counts": dict(import_visitor.import_counts),
                "classes_defined": import_visitor.classes,
                "functions_defined": import_visitor.functions,
                "versions": _resolve_versions(import_visitor.imports),
            },
            "complexity": complexity_visitor.metrics,
            "runtime": runtime or self._empty_runtime(),
        }

    def _empty_runtime(self) -> dict[str, Any]:
        return {"python_version": "", "lines_of_code": 0, "code_cells": 0}

    def empty_payload(self) -> dict[str, Any]:
        """Provides the default schema for empty code blocks."""
        return {
            "imports": {
                "declared": [],
                "usage": {},
                "counts": {},
                "classes_defined": [],
                "functions_defined": [],
                "versions": {},
            },
            "complexity": {
                "loops": 0,
                "for_else": 0,
                "while_else": 0,
                "branches": 0,
                "match_cases": 0,
                "list_comp": 0,
                "dict_comp": 0,
                "set_comp": 0,
                "gen_expr": 0,
                "try_except": 0,
                "with_blocks": 0,
                "async_functions": 0,
                "awaits": 0,
                "max_nesting_depth": 0,
            },
            "runtime": self._empty_runtime(),
        }
