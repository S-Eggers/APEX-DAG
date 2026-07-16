import ast
import itertools
import logging

from ..core.types import PRType, SpatialMetadata, WIRNode
from ..core.utils import add_id, flatten, is_empty_or_none_list

logger = logging.getLogger(__name__)

def extract_from_node(node: WIRNode, field: str) -> WIRNode | None:
    """Extracts specific structural elements (operation, input, output, caller) from an AST node."""
    ast_node = node.node
    if ast_node is None:
        return WIRNode(None)

    match ast_node.__class__.__name__:
        case "Assign":
            if field == "operation":
                return WIRNode(ast_node.__class__.__name__ + add_id(), parent_ast=ast_node)
            elif field == "input":
                return WIRNode(ast_node.value, parent_ast=ast_node)
            elif field == "output":
                return WIRNode(ast_node.targets, parent_ast=ast_node)
        case "Call":
            if field == "operation":
                return WIRNode(ast_node.func, True, parent_ast=ast_node)
            elif field == "input":
                return WIRNode(ast_node.args + ast_node.keywords, parent_ast=ast_node)
            elif ast_node.func and hasattr(ast_node.func, "id") and isinstance(ast_node.func.id, str) and field == "output":
                return WIRNode(ast_node.func.id + add_id(), parent_ast=ast_node)
        case "Attribute":
            if field == "caller":
                return WIRNode(ast_node.value, parent_ast=ast_node)
            elif field == "operation":
                return WIRNode(ast_node.attr, True, parent_ast=ast_node)
            elif field == "output":
                return WIRNode(ast_node.attr + add_id(), parent_ast=ast_node)
        case "Name":
            if field == "output":
                return WIRNode(ast_node.id, parent_ast=ast_node)
        case "Constant":
            if field == "output":
                return WIRNode(f"{ast_node.value}{add_id()}", parent_ast=ast_node)
        case "Import":
            if field == "output":
                return WIRNode(ast_node.names, parent_ast=ast_node)
        case "Module":
            return WIRNode(None, parent_ast=ast_node)
        case "alias":
            if field == "output":
                return WIRNode(ast_node.asname if ast_node.asname is not None else ast_node.name, parent_ast=ast_node)
            elif field == "input":
                return WIRNode(ast_node.name if ast_node.asname is not None else ast_node.name + add_id())
            elif field == "operation":
                return WIRNode("ImportAs" + add_id(), parent_ast=ast_node)
        case "ImportFrom":
            if field == "operation":
                return WIRNode(ast_node.__class__.__name__ + add_id(), parent_ast=ast_node)
            elif field == "input":
                return WIRNode(ast_node.module, parent_ast=ast_node)
            elif field == "output":
                return WIRNode([child.name if child.asname is None else child for child in ast_node.names], parent_ast=ast_node)
        case "Subscript":
            if field == "operation":
                return WIRNode(ast_node.__class__.__name__ + add_id(), parent_ast=ast_node)
            elif field == "input":
                return WIRNode(ast_node.slice, parent_ast=ast_node)
            elif field == "caller":
                return WIRNode(ast_node.value, parent_ast=ast_node)
        case "Tuple":
            if field == "output":
                return WIRNode(ast_node.elts, parent_ast=ast_node)
        case "Slice":
            if field == "operation":
                return WIRNode(ast_node.__class__.__name__ + add_id(), parent_ast=ast_node)
            elif field == "input":
                return WIRNode([val if val is not None else "" + add_id() for val in (ast_node.lower, ast_node.upper, ast_node.step)], parent_ast=ast_node)
        case "List":
            if field == "input":
                return WIRNode(ast_node.elts, parent_ast=ast_node)
            elif field == "operation":
                return WIRNode(ast_node.__class__.__name__ + add_id(), parent_ast=ast_node)
        case "Expr":
            if field == "output":
                return WIRNode(ast_node.value, parent_ast=ast_node)
        case "For":
            if field == "input":
                return WIRNode(ast_node.iter, parent_ast=ast_node)
            elif field == "operation":
                return WIRNode(ast_node.__class__.__name__ + add_id(), parent_ast=ast_node)
            elif field == "output":
                return WIRNode(ast_node.target, parent_ast=ast_node)
        case "Compare":
            if field == "input":
                return WIRNode([ast_node.left, *ast_node.comparators], parent_ast=ast_node)
            elif field == "operation":
                return WIRNode(ast_node.ops, parent_ast=ast_node)
        case "BinOp":
            if field == "input":
                return WIRNode([ast_node.left, ast_node.right], parent_ast=ast_node)
            elif field == "operation":
                return WIRNode(ast_node.op, parent_ast=ast_node)
        case "Lambda":
            if field == "operation":
                return WIRNode(ast_node.__class__.__name__ + add_id(), parent_ast=ast_node)
        case "keyword":
            if field == "input":
                return WIRNode(ast_node.value, parent_ast=ast_node)
            elif field == "output":
                if ast_node.arg is not None:
                    return WIRNode(ast_node.arg + add_id(), parent_ast=ast_node)
            elif field == "operation":
                return WIRNode(ast_node.__class__.__name__ + add_id(), parent_ast=ast_node)
        case "ListComp":
            if field == "operation":
                return WIRNode(ast_node.__class__.__name__ + add_id(), parent_ast=ast_node)
            elif field == "input":
                return WIRNode([*ast_node.generators, ast_node.elt], parent_ast=ast_node)
        case "UnaryOp":
            if field == "operation":
                return WIRNode(ast_node.__class__.__name__ + add_id(), parent_ast=ast_node)
            elif field == "input":
                return WIRNode(ast_node.operand, parent_ast=ast_node)
        case "comprehension":
            if field == "input":
                return WIRNode(ast_node.iter, parent_ast=ast_node)
            elif field == "output":
                return WIRNode(ast_node.target, parent_ast=ast_node)
            elif field == "operation":
                return WIRNode(ast_node.__class__.__name__ + add_id(), parent_ast=ast_node)
        case "IfExp":
            if field == "operation":
                return WIRNode(ast_node.__class__.__name__ + add_id(), parent_ast=ast_node)
            elif field == "input":
                return WIRNode([ast_node.test, ast_node.test, ast_node.orelse], parent_ast=ast_node)
        case "Dict":
            if field == "input":
                return WIRNode([ast.Tuple([key, value]) for key, value in zip(ast_node.keys, ast_node.values, strict=False)], parent_ast=ast_node)
            elif field == "operation":
                return WIRNode(ast_node.__class__.__name__ + add_id(), parent_ast=ast_node)
        case "Add" | "Sub" | "Mult" | "Div" | "Eq" | "Lt" | "Gt" | "GtE" | "LtE" | "BitAnd" | "Mod":
            if field == "operation":
                return WIRNode(ast_node.__class__.__name__ + add_id(), parent_ast=ast_node)
        case _:
            pass

    return WIRNode(None, parent_ast=ast_node)

def gen_pr(v: WIRNode, prs: list[PRType], registry: dict[str, SpatialMetadata]) -> tuple[WIRNode, list[PRType]]:
    is_attribute = v.is_attribute
    if v.node is None:
        return WIRNode(None), prs

    if isinstance(v.node, list):
        os_list = []
        for node in v.node:
            o, prs = gen_pr(WIRNode(node), prs, registry)
            os_list.append(o.node)
        return WIRNode(list(flatten(os_list))), prs

    if isinstance(v.node, (str, int, float, bool, type(None))):
        if isinstance(v.node, str):
            return_name = v.node.replace("\n", "")
            if is_attribute:
                return_name = return_name + ":meth"
            v_new = WIRNode(return_name)
            v_new.spatial = v.spatial
            v = v_new
        return v, prs

    p, prs = gen_pr(extract_from_node(v, "operation"), prs, registry)
    input_i, prs = gen_pr(extract_from_node(v, "input"), prs, registry)
    c, prs = gen_pr(extract_from_node(v, "caller"), prs, registry)
    output_o, prs = gen_pr(extract_from_node(v, "output"), prs, registry)

    if output_o.node is None:
        if is_empty_or_none_list(input_i.node) and c.node is None:
            output_o = p
            return output_o, prs
        elif isinstance(p.node, list):
            output_o = WIRNode([f"{op}{add_id()}" for op in p.node])
        else:
            output_o = WIRNode(f"{p.node}{add_id()}")

    input_nodes = input_i.node if isinstance(input_i.node, list) else [input_i.node]
    output_nodes = output_o.node if isinstance(output_o.node, list) else [output_o.node]
    caller_nodes = c.node if isinstance(c.node, list) else [c.node]
    operation_nodes = p.node if isinstance(p.node, list) else [p.node]

    def _register(nodes: list, source_wir: WIRNode) -> None:
        if source_wir.spatial.lineno is not None:
            for n in nodes:
                if n is not None:
                    registry[str(n)] = source_wir.spatial

    _register(input_nodes, input_i)
    _register(output_nodes, output_o)
    _register(caller_nodes, c)
    _register(operation_nodes, p)

    for _i, _c, _p, _o in itertools.product(input_nodes, caller_nodes, operation_nodes, output_nodes):
        prs.append((_i, _c, _p, _o))

    return output_o, prs
