import re


class IPythonSanitizerMixin:
    """
    Mixin to intercept and sanitize Jupyter/IPython specific syntax before AST parsing.
    """

    def sanitize_ipython_cells(self, cells: list) -> list:
        """
        Removes IPython magic (%) and shell (!) commands from a list
        of cell dictionaries.
        Replaces them with blank space to preserve strict AST global
        line numbering.
        """
        sanitized_cells = []
        for cell in cells:
            safe_cell = dict(cell)
            source = str(cell.get("source", ""))

            safe_cell["source"] = re.sub(
                r"^[ \t]*[%!].*", "", source, flags=re.MULTILINE
            )

            sanitized_cells.append(safe_cell)

        return sanitized_cells
