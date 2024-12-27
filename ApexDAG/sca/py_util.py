import ast
from typing import Optional

def get_operator_description(node: ast.AST) -> Optional[str]:
    try:
        operator =  node.ops[0].__class__.__name__.lower()
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
        
    except AttributeError:
        operator = None
        
    return operator