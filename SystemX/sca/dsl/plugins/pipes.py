import ast

from SystemX.sca.dsl.base import DslPlugin, ImportContext

_MODULES = ("sspipe", "pipe")

class PipesPlugin(DslPlugin):
    """Rewrites pipe-library | chains (sspipe, pipe) into method-chain calls."""

    name = "pipes"
    trigger_modules = frozenset(_MODULES)

    def make_transformer(self, imports: ImportContext) -> ast.NodeTransformer:
        return _PipesTransformer(imports)

class _PipesTransformer(ast.NodeTransformer):
    def __init__(self, imports: ImportContext) -> None:
        self._from_modules: dict[str, str] = {}
        self._alias_modules: dict[str, str] = {}
        for module in _MODULES:
            self._from_modules.update(dict.fromkeys(imports.from_import_names(module), module))
            self._alias_modules.update(dict.fromkeys(imports.aliases_for(module), module))

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        if not isinstance(node.op, ast.BitOr):
            return node

        new = self._rewrite_stage(node.left, node.right)
        if new is None:
            return node

        ast.copy_location(new, node.right)
        ast.copy_location(new.func, node.right)
        new._dsl_rewritten = True
        return new

    def _rewrite_stage(self, left: ast.expr, stage: ast.expr) -> ast.Call | None:
        if isinstance(stage, ast.Name) and stage.id in self._from_modules:
            return ast.Call(func=ast.Attribute(value=left, attr=stage.id, ctx=ast.Load()), args=[], keywords=[])

        if not isinstance(stage, ast.Call):
            return None

        module = self._stage_module(stage.func)
        if module is None:
            return None

        if module == "sspipe":
            first = stage.args[0] if stage.args else None
            if isinstance(first, (ast.Name, ast.Attribute)):
                return ast.Call(
                    func=ast.Attribute(value=left, attr=self._last_segment(first), ctx=ast.Load()),
                    args=stage.args[1:],
                    keywords=stage.keywords,
                )
            return ast.Call(func=ast.Attribute(value=left, attr="pipe", ctx=ast.Load()), args=stage.args, keywords=stage.keywords)

        return ast.Call(
            func=ast.Attribute(value=left, attr=self._last_segment(stage.func), ctx=ast.Load()),
            args=stage.args,
            keywords=stage.keywords,
        )

    def _stage_module(self, func: ast.expr) -> str | None:
        if isinstance(func, ast.Name):
            return self._from_modules.get(func.id)
        if isinstance(func, ast.Attribute):
            root = func
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                return self._alias_modules.get(root.id)
        return None

    @staticmethod
    def _last_segment(expr: ast.expr) -> str:
        return expr.attr if isinstance(expr, ast.Attribute) else expr.id
