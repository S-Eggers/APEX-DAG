import ast

from SystemX.sca.dsl.base import ImportContext

class ImportScanner(ast.NodeVisitor):
    """Collects imports across cells, mirroring the asname handling of CDFIntermediateRepresentation.visit_Import / visit_ImportFrom."""

    def __init__(self) -> None:
        self._context = ImportContext()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._context.aliases[alias.asname or alias.name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            self._context.from_imports[alias.asname or alias.name] = module

    def scan_sources(self, sources: list[str]) -> ImportContext:
        for source in sources:
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            self.visit(tree)
        return self._context
