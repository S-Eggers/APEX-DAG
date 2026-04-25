import ast
import re


def get_operator_description(node: ast.AST) -> str | None:
    try:
        if isinstance(node, ast.Compare) or isinstance(node, ast.BoolOp):
            operator = node.ops[0].__class__.__name__.lower()
        elif isinstance(node, ast.In):
            operator = node.__doc__.lower()
        else:
            return None

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
            "or": "or",
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

def tokenize_method(method: str, imported_names: dict, import_from_modules: dict) -> str:
    """Formats camelCase and snake_case method names, removing library aliases."""
    if not method: return ""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1 \2", method)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1 \2", s1)
    tokens = re.split(r"[._\s]", s2)
    for library_alias in imported_names:
        if library_alias in tokens and library_alias not in import_from_modules:
            tokens.remove(library_alias)
    return " ".join(tokens).lower()

def tokenize_literal(literal: str) -> str:
    """Cleans up literal strings for UI display."""
    clean_literal = re.sub(r"['\"\[\]\(\)\{\}]", "", literal)
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1 \2", clean_literal)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1 \2", s1)
    tokens = re.split(r"[._\s\-/]", s2)
    return " ".join(filter(None, tokens)).lower()

def get_names(node: ast.AST, code_buffer: str = None) -> list[str] | None:
    """Extracts raw string names from various AST node types."""
    match node:
        case ast.Name():
            return [node.id]
        case ast.Attribute():
            name = get_names(node.value, code_buffer)
            return ([name[0] + "." + node.attr]) if name else [node.attr]
        case ast.Tuple() | ast.List() | ast.Set():
            return [get_names(elt, code_buffer) for elt in node.elts if get_names(elt, code_buffer)]
        case ast.Dict():
            names = []
            for value in node.values:
                if v_names := get_names(value, code_buffer): names.extend(v_names)
            for key in node.keys:
                if k_names := get_names(key, code_buffer): names.extend(k_names)
            return names
        case ast.FunctionDef() | ast.AsyncFunctionDef():
            if isinstance(node.parent, ast.ClassDef):
                return [f"{node.parent.name}.{node.name}"]
            return [node.name]
        case ast.Subscript():
            return get_names(node.value, code_buffer)
        case ast.Call():
            return get_names(node.func, code_buffer)
        case ast.If() | ast.IfExp():
            return ["If"]
        case ast.While() | ast.For():
            return ["Loop"]
        case ast.Lambda():
            return ["Lambda"]
        case ast.Starred():
            return get_names(node.value, code_buffer)
        case (ast.BinOp() | ast.Compare() | ast.BoolOp() | ast.UnaryOp() | ast.JoinedStr() | ast.Constant()):
            return None
        case _:
            return None

def get_target_components(raw_target_list: list) -> list[list[str]]:
    """Cleans nested tuple target structures."""
    components = []
    if not raw_target_list: return []
    if isinstance(raw_target_list[0], list):
        for sub_list in raw_target_list:
            components.extend(get_target_components(sub_list))
    else:
        components.append(raw_target_list)
    return components

def get_lr_values(left: ast.AST, right: ast.AST, code_buffer: str = None) -> tuple[str | None, str | None]:
    """Extracts base names for left/right binary operations."""
    def get_name(node: ast.AST) -> str | None:
        names = get_names(node, code_buffer)
        if not names: return None
        flat_names = flatten_list(names)
        return flat_names[0] if flat_names else None
    return get_name(left), get_name(right)

def get_base_name(node: ast.AST) -> str | None:
    """Recursively resolves the base identifier of an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Constant):
        return str(node.value)
    elif isinstance(node, (ast.Attribute, ast.Subscript)):
        return get_base_name(node.value)
    elif isinstance(node, ast.Call):
        return get_base_name(node.func)
    return None

def process_arguments(node: ast.arguments) -> dict[str, list]:
    """Extracts positional arguments and defaults from function definitions."""
    arguments = {"args": [], "defaults": []}
    for arg in node.args:
        arguments["args"].append(arg.arg)
    for default in node.defaults:
        arguments["defaults"].append(default)
    return arguments
