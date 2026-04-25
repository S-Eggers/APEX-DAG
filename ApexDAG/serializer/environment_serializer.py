from typing import Any


class EnvironmentSerializer:
    def to_dict(self, import_visitor, complexity_visitor) -> dict[str, Any]:
        """Transforms visitor state into the final API contract."""
        sanitized_usage = {
            module: dict(usages) for module, usages in import_visitor.import_usage.items()
        }

        return {
            "imports": {
                "declared": import_visitor.imports,
                "usage": sanitized_usage,
                "counts": dict(import_visitor.import_counts),
                "classes_defined": import_visitor.classes,
                "functions_defined": import_visitor.functions
            },
            "complexity": complexity_visitor.metrics
        }

    def empty_payload(self) -> dict[str, Any]:
        """Provides the default schema for empty code blocks."""
        return {
            "imports": {
                "declared": [],
                "usage": {},
                "counts": {},
                "classes_defined": [],
                "functions_defined": []
            },
            "complexity": {
                "loops": 0, "for_else": 0, "while_else": 0, "branches": 0,
                "match_cases": 0, "list_comp": 0, "dict_comp": 0, "set_comp": 0,
                "gen_expr": 0, "try_except": 0, "with_blocks": 0, "max_nesting_depth": 0
            }
        }
