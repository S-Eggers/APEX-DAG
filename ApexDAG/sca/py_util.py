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

def flatten_list(input_list):
    result = []
    
    def process_element(element):
        if isinstance(element, list):
            if len(element) == 1: 
                # Melt down single-element lists
                process_element(element[0])
            else:  
                # Preserve lists with more than one element
                result.append([process_element(e) for e in element])
        else:
            result.append(element)

    for item in input_list:
        process_element(item)
    
    return result