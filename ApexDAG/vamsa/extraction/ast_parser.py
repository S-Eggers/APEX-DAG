import ast
import itertools
import logging
from typing import Tuple, Optional

from ..core.types import WIRNode, WIRNodeType, PRType
from ..core.utils import add_id, flatten, is_empty_or_none_list
from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


def extract_from_node(node: WIRNode, field: str) -> Optional[WIRNode]:
    """
    Extracts specific structural elements (operation, input, output, caller) from an AST node.
    WARNING: Highly complex match/case structure inherited from original Vamsa implementation.
    """
    ast_node = node.node
    if ast_node is None:
        return WIRNode(None)

    match ast_node.__class__.__name__:
        case "Assign":
            if field == "operation": return WIRNode(ast_node.__class__.__name__ + add_id())
            elif field == "input": return WIRNode(ast_node.value)
            elif field == "output": return WIRNode(ast_node.targets)
        case "Call":
            if field == "operation": return WIRNode(ast_node.func, True)
            elif field == "input": return WIRNode(ast_node.args + ast_node.keywords)
            elif ast_node.func and hasattr(ast_node.func, "id") and isinstance(ast_node.func.id, str) and field == "output":
                return WIRNode(ast_node.func.id + add_id())
        case "Attribute":
            if field == "caller": return WIRNode(ast_node.value)
            elif field == "operation": return WIRNode(ast_node.attr, True)
            elif field == "output": return WIRNode(ast_node.attr + add_id())
        case "Name":
            if field == "output": return WIRNode(ast_node.id)
        case "Constant":
            if field == "output": return WIRNode(f"{ast_node.value}{add_id()}")
        case "Import":
            if field == "output": return WIRNode(ast_node.names)
        case "Module":
            return WIRNode(None)
        case "alias":
            if field == "output": return WIRNode(ast_node.asname if ast_node.asname is not None else ast_node.name)
            elif field == "input": return WIRNode(ast_node.name if ast_node.asname is not None else ast_node.name + add_id())
            elif field == "operation": return WIRNode("ImportAs" + add_id())
        case "ImportFrom":
            if field == "operation": return WIRNode(ast_node.__class__.__name__ + add_id())
            elif field == "input": return WIRNode(ast_node.module)
            elif field == "output": return WIRNode([child.name if child.asname is None else child for child in ast_node.names])
        case "Subscript":
            if field == "operation": return WIRNode(ast_node.__class__.__name__ + add_id())
            elif field == "input": return WIRNode(ast_node.slice)
            elif field == "caller": return WIRNode(ast_node.value)
        case "Tuple":
            if field == "output": return WIRNode(ast_node.elts)
        case "Slice":
            if field == "operation": return WIRNode(ast_node.__class__.__name__ + add_id())
            elif field == "input": return WIRNode([val if val is not None else "" + add_id() for val in (ast_node.lower, ast_node.upper, ast_node.step)])
        case "List":
            if field == "input": return WIRNode(ast_node.elts)
            elif field == "operation": return WIRNode(ast_node.__class__.__name__ + add_id())
        case "Expr":
            if field == "output": return WIRNode(ast_node.value)
        case "For":
            if field == "input": return WIRNode(ast_node.iter)
            elif field == "operation": return WIRNode(ast_node.__class__.__name__ + add_id())
            elif field == "output": return WIRNode(ast_node.target)
        case "Compare":
            if field == "input": return WIRNode([ast_node.left] + ast_node.comparators)
            elif field == "operation": return WIRNode(ast_node.ops)
        case "BinOp":
            if field == "input": return WIRNode([ast_node.left, ast_node.right])
            elif field == "operation": return WIRNode(ast_node.op)
        case "Lambda":
            if field == "operation": return WIRNode(ast_node.__class__.__name__ + add_id())
        case "keyword":
            if field == "input": return WIRNode(ast_node.value)
            elif field == "output": 
                if ast_node.arg is not None: return WIRNode(ast_node.arg + add_id())
            elif field == "operation": return WIRNode(ast_node.__class__.__name__ + add_id())
        case "ListComp":
            if field == "operation": return WIRNode(ast_node.__class__.__name__ + add_id())
            elif field == "input": return WIRNode(ast_node.generators + [ast_node.elt])
        case "UnaryOp":
            if field == "operation": return WIRNode(ast_node.__class__.__name__ + add_id())
            elif field == "input": return WIRNode(ast_node.operand)
        case "comprehension":
            if field == "input": return WIRNode(ast_node.iter)
            elif field == "output": return WIRNode(ast_node.target)
            elif field == "operation": return WIRNode(ast_node.__class__.__name__ + add_id())
        case "IfExp":
            if field == "operation": return WIRNode(ast_node.__class__.__name__ + add_id())
            elif field == "input": return WIRNode([ast_node.test, ast_node.test, ast_node.orelse])
        case "Dict":
            if field == "input": return WIRNode([ast.Tuple([key, value]) for key, value in zip(ast_node.keys, ast_node.values)])
            elif field == "operation": return WIRNode(ast_node.__class__.__name__ + add_id())
        case "Add" | "Sub" | "Mult" | "Div" | "Eq" | "Lt" | "Gt" | "GtE" | "LtE" | "BitAnd" | "Mod":
            if field == "operation": return WIRNode(ast_node.__class__.__name__ + add_id())
        case _:
            pass
            
    return WIRNode(None)


def GenPR(v: WIRNode, PRs: list[PRType]) -> Tuple[WIRNode, list[PRType]]:
    """
    Recursively processes an AST node to generate WIR variables and append to PRs.
    """
    isAttribute = v.isAttribute
    if v.node is None:
        return WIRNode(None), PRs

    if isinstance(v.node, list):
        os_list = []
        for node in v.node:
            o, PRs = GenPR(WIRNode(node), PRs)
            os_list.append(o.node)
        return WIRNode(list(flatten(os_list))), PRs

    if isinstance(v.node, (str, int, float, bool, type(None))):
        if isinstance(v.node, str):
            return_name = v.node.replace("\n", "")
            if isAttribute:
                return_name = return_name + ":meth"
            v = WIRNode(return_name)
        return v, PRs

    p, PRs = GenPR(extract_from_node(v, "operation"), PRs)
    I, PRs = GenPR(extract_from_node(v, "input"), PRs)
    c, PRs = GenPR(extract_from_node(v, "caller"), PRs)
    O, PRs = GenPR(extract_from_node(v, "output"), PRs)

    if O.node is None:
        if is_empty_or_none_list(I.node) and c.node is None:
            O = p
            return O, PRs
        elif isinstance(p.node, list):
            O = WIRNode([(op + add_id()) for op in p.node])
        else:
            O = WIRNode(p.node + add_id())

    input_nodes = I.node if isinstance(I.node, list) else [I.node]
    output_nodes = O.node if isinstance(O.node, list) else [O.node]
    caller_nodes = c.node if isinstance(c.node, list) else [c.node]
    operation_nodes = p.node if isinstance(p.node, list) else [p.node]

    for _i, _c, _p, _o in itertools.product(input_nodes, caller_nodes, operation_nodes, output_nodes):
        PRs.append((_i, _c, _p, _o))
        
    return O, PRs