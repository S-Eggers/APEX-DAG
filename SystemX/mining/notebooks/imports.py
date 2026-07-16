from __future__ import annotations

import ast
from typing import Any

from SystemX.parser.sanitizer_mixin import IPythonSanitizerMixin

_SANITIZER = IPythonSanitizerMixin()

def _iter_code_cells(notebook: dict[str, Any]) -> list[dict[str, Any]]:
    """Return code cells with their source normalized to a single string."""
    cells = notebook.get("cells", []) if isinstance(notebook, dict) else []
    code_cells = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        normalized = dict(cell)
        source = normalized.get("source", "")
        if isinstance(source, list):
            normalized["source"] = "".join(str(line) for line in source)
        code_cells.append(normalized)
    return code_cells

def parse_code_cells(notebook: dict[str, Any]) -> list[ast.AST]:
    """Sanitize IPython syntax and parse each code cell, skipping unparseable ones."""
    trees: list[ast.AST] = []
    for cell in _SANITIZER.sanitize_ipython_cells(_iter_code_cells(notebook)):
        source = str(cell.get("source", ""))
        try:
            trees.append(ast.parse(source))
        except SyntaxError:
            continue
    return trees

def import_aliases(trees: list[ast.AST]) -> dict[str, str]:
    """Map each imported alias/name to its case-folded root library."""
    aliases: dict[str, str] = {}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    library = alias.name.split(".", 1)[0].strip().casefold()
                    if library:
                        aliases[alias.asname or alias.name.split(".", 1)[0]] = library
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                library = node.module.split(".", 1)[0].strip().casefold()
                for alias in node.names:
                    aliases[alias.asname or alias.name] = library
    return aliases

def imported_roots(notebook: dict[str, Any]) -> set[str]:
    """Case-folded set of root libraries imported anywhere in the notebook."""
    return set(import_aliases(parse_code_cells(notebook)).values())
