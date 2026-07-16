import ast

from SystemX.sca.dsl.base import DslPlugin, ImportContext

_MODULE = "apache_beam"

class BeamPlugin(DslPlugin):
    """Rewrites Apache Beam's | / 'Label' >> transform pipeline DSL into method-chain calls on the upstream PCollection."""

    name = "beam"
    trigger_modules = frozenset({_MODULE})

    def make_transformer(self, imports: ImportContext) -> ast.NodeTransformer:
        return _BeamTransformer(imports)

class _BeamTransformer(ast.NodeTransformer):
    def __init__(self, imports: ImportContext) -> None:
        self._aliases = imports.aliases_for(_MODULE)
        self._from_imports = imports.from_import_names(_MODULE)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        if not isinstance(node.op, ast.BitOr):
            return node

        transform, had_label = self._unwrap_label(node.right)

        if self._is_beam_call(transform):
            func = ast.Attribute(value=node.left, attr=self._attr_name(transform.func), ctx=ast.Load())
            new = ast.Call(func=func, args=transform.args, keywords=transform.keywords)
        elif (had_label or getattr(node.left, "_dsl_rewritten", False)) and isinstance(transform, (ast.Name, ast.Attribute, ast.Call)):
            func = ast.Attribute(value=node.left, attr="apply", ctx=ast.Load())
            new = ast.Call(func=func, args=[transform], keywords=[])
        else:
            return node

        ast.copy_location(new, node.right)
        ast.copy_location(new.func, node.right)
        new._dsl_rewritten = True
        return new

    def visit_With(self, node: ast.With) -> ast.AST | list[ast.stmt]:
        node = self.generic_visit(node)
        if len(node.items) != 1:
            return node

        item = node.items[0]
        if not self._is_beam_call(item.context_expr) or not isinstance(item.optional_vars, ast.Name):
            return node

        assign = ast.Assign(targets=[item.optional_vars], value=item.context_expr)
        ast.copy_location(assign, item.context_expr)
        return [assign, *node.body]

    def _is_beam_call(self, expr: ast.AST) -> bool:
        return isinstance(expr, ast.Call) and self._is_beam_rooted(expr.func)

    def _is_beam_rooted(self, expr: ast.AST) -> bool:
        if isinstance(expr, ast.Attribute):
            root = expr
            while isinstance(root, ast.Attribute):
                root = root.value
            return isinstance(root, ast.Name) and root.id in self._aliases
        return isinstance(expr, ast.Name) and expr.id in self._from_imports

    @staticmethod
    def _unwrap_label(expr: ast.AST) -> tuple[ast.AST, bool]:
        """'Label' >> transform -> transform."""
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.RShift) and isinstance(expr.left, ast.Constant) and isinstance(expr.left.value, str):
            return expr.right, True
        return expr, False

    @staticmethod
    def _attr_name(func: ast.expr) -> str:
        return func.attr if isinstance(func, ast.Attribute) else func.id
