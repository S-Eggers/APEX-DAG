import ast

from SystemX.sca.dsl.base import DslPlugin, ImportContext

class AirflowPlugin(DslPlugin):
    """Rewrites Airflow's t1 >> t2 / t1 << t2 task-dependency DSL into set_downstream method calls."""

    name = "airflow"
    trigger_modules = frozenset({"airflow"})

    def make_transformer(self, imports: ImportContext) -> ast.NodeTransformer:
        return _AirflowTransformer()

class _AirflowTransformer(ast.NodeTransformer):
    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.stmt]:
        value = node.value
        if isinstance(value, ast.BinOp) and self._is_dep_tree(value):
            statements, _ = self._expand(value)
            if statements:
                return statements
        return self.generic_visit(node)

    def _is_dep_tree(self, expr: ast.AST) -> bool:
        return (
            isinstance(expr, ast.BinOp)
            and isinstance(expr.op, (ast.RShift, ast.LShift))
            and self._is_operand(expr.left)
            and self._is_operand(expr.right)
        )

    def _is_operand(self, expr: ast.AST) -> bool:
        if isinstance(expr, ast.Name):
            return True
        if isinstance(expr, ast.List):
            return bool(expr.elts) and all(isinstance(element, ast.Name) for element in expr.elts)
        return self._is_dep_tree(expr)

    def _expand(self, expr: ast.AST) -> tuple[list[ast.stmt], list[ast.Name]]:
        """Both a >> b and a << b evaluate to the right operand in Airflow, so chains fold left-to-right over the returned names."""
        if not isinstance(expr, ast.BinOp):
            return [], self._names(expr)

        left_stmts, left_names = self._expand(expr.left)
        right_stmts, right_names = self._expand(expr.right)
        statements = [*left_stmts, *right_stmts]

        if isinstance(expr.op, ast.RShift):
            upstreams, downstreams = left_names, right_names
        else:
            upstreams, downstreams = right_names, left_names

        locate_on_upstream = len(downstreams) == 1 and len(upstreams) > 1
        for upstream in upstreams:
            for downstream in downstreams:
                location = upstream if locate_on_upstream else downstream
                statements.append(self._dep_assign(upstream, downstream, location))

        return statements, right_names

    @staticmethod
    def _names(expr: ast.AST) -> list[ast.Name]:
        return [expr] if isinstance(expr, ast.Name) else list(expr.elts)

    @staticmethod
    def _dep_assign(upstream: ast.Name, downstream: ast.Name, location: ast.AST) -> ast.Assign:
        call = ast.Call(
            func=ast.Attribute(value=ast.Name(id=upstream.id, ctx=ast.Load()), attr="set_downstream", ctx=ast.Load()),
            args=[ast.Name(id=downstream.id, ctx=ast.Load())],
            keywords=[],
        )
        assign = ast.Assign(targets=[ast.Name(id=downstream.id, ctx=ast.Store())], value=call)
        for synthetic in (assign, assign.targets[0], call, call.func, call.func.value, call.args[0]):
            ast.copy_location(synthetic, location)
        return assign
