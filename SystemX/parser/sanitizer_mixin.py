import re

class IPythonSanitizerMixin:
    """Mixin to intercept and sanitize Jupyter/IPython specific syntax before AST parsing."""

    def sanitize_ipython_cells(self, cells: list) -> list:
        sanitized_cells = []
        for cell in cells:
            safe_cell = dict(cell)
            source = str(cell.get("source", ""))

            source = re.sub(r"^[ \t]*[%!].*", "", source, flags=re.MULTILINE)
            source = re.sub(r"^[ \t]*\?.*|.*\?[ \t]*$", "", source, flags=re.MULTILINE)

            safe_cell["source"] = source
            sanitized_cells.append(safe_cell)

        return sanitized_cells
