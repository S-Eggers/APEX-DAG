import ast
from collections.abc import Sequence

from SystemX.sca.dsl.base import DslPlugin
from SystemX.sca.dsl.scanner import ImportScanner

def default_plugins() -> list[DslPlugin]:
    from SystemX.sca.dsl.plugins import DEFAULT_PLUGINS

    return [plugin_cls() for plugin_cls in DEFAULT_PLUGINS]

class DslRewriter:
    """Rewrites DSL constructs per parse session (one instance per notebook)."""

    def __init__(self, plugins: Sequence[DslPlugin] | None = None) -> None:
        self._plugins = list(plugins) if plugins is not None else default_plugins()
        self._transformers: list[ast.NodeTransformer] = []

    def pre_scan(self, sources: list[str]) -> None:
        imports = ImportScanner().scan_sources(sources)
        self._transformers = [plugin.make_transformer(imports) for plugin in self._plugins if plugin.is_active(imports)]

    def rewrite(self, tree: ast.Module) -> ast.Module:
        if not self._transformers:
            return tree
        for transformer in self._transformers:
            tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        return tree
