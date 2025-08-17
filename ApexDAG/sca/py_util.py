import ast
from typing import Optional

def get_operator_description(node: ast.AST) -> Optional[str]:
    try:
        if isinstance(node, ast.Compare):
            operator = node.ops[0].__class__.__name__.lower()
        elif isinstance(node, ast.BoolOp):
            operator = node.op.__class__.__name__.lower()
        else:
            return None # Or raise an error, depending on desired behavior for unhandled types

        operator_translation = {
            "eq": "equal",
            "not_eq": "not equal",
            "noteq": "not equal",
            "lt": "less than",
            "lte": "less than or equal",
            "gt": "greater than",
            "gte": "greater than or equal",
            "is_not": "is not",
            "isnot": "is not",
            "not_in": "not in",
            "notin": "not in",
            "in": "in",
            "is": "is",
            "not": "not",
            "and": "and",
            "or": "or"
        }
        operator = operator_translation[operator]

    except (AttributeError, KeyError):
        operator = None

    return operator

def flatten_list(input_list):
    result = []
    for item in input_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result