from dataclasses import asdict, dataclass


@dataclass
class ElementMetadata:
    name: str
    category: str
    label: str
    color: str
    border_style: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


DEFAULT_NODE_COLOR: str = "#B0E0E6"
DEFAULT_EDGE_COLOR: str = "#d3d3d3"


def _build_metadata_dict(category: str, color: str, border_style: str, items: list[tuple[str, int, str]]) -> dict[int, dict[str, str]]:
    """Builds a typed dictionary of metadata for a specific category of elements."""
    result: dict[int, dict[str, str]] = {}
    for name, uid, label in items:
        final_label = label if label else name.replace("_", " ")
        result[uid] = ElementMetadata(name=name, category=category, label=final_label, color=color, border_style=border_style).to_dict()
    return result


DATAFLOW_NODES: dict[int, dict[str, str]] = {
    **_build_metadata_dict("State", "#B0C4DE", "solid", [("VARIABLE", 0, "Variable")]),
    **_build_metadata_dict("Definitions", "#FFC0CB", "solid", [("IMPORT", 1, "Import")]),
    **_build_metadata_dict("Data", "#c4deb0", "solid", [("DATASET", 2, "Dataset / Table")]),
    **_build_metadata_dict("Definitions", "#c4deb0", "solid", [("FUNCTION", 3, "Function")]),
    **_build_metadata_dict("State", "#b3b0de", "solid", [("INTERMEDIATE", 4, "Intermediate")]),
    **_build_metadata_dict("Control Flow", "#DEDAB0", "solid", [("IF", 5, "If")]),
    **_build_metadata_dict("Control Flow", "#B0DEB9", "solid", [("LOOP", 6, "Loop")]),
    **_build_metadata_dict("Definitions", "#DEB0DE", "solid", [("CLASS", 7, "Class")]),
    **_build_metadata_dict("State", "#B0E0E6", "solid", [("LITERAL", 8, "Literal")]),
}

DATAFLOW_EDGES: dict[int, dict[str, str]] = {
    **_build_metadata_dict("Edges", "#FFA07A", "solid", [("CALLER", 0, "Caller")]),
    **_build_metadata_dict("Edges", "#98FB98", "solid", [("INPUT", 1, "Input")]),
    **_build_metadata_dict("Edges", "#d3d3d3", "dashed", [("OMITTED", 2, "Reassign")]),
    **_build_metadata_dict("Edges", "#B0E0E6", "solid", [("BRANCH", 3, "Branch")]),
    **_build_metadata_dict("Edges", "#FFDAB9", "solid", [("LOOP", 4, "Loop")]),
    **_build_metadata_dict("Edges", "#c4deb0", "solid", [("FUNCTION_CALL", 5, "Function Call")]),
    **_build_metadata_dict("Edges", DEFAULT_EDGE_COLOR, "solid", [("CLASS_CALL", 6, "Class Call")]),
}

DOMAIN_NODES: dict[int, dict[str, str]] = {
    **_build_metadata_dict("Entities", "#B0C4DE", "solid", [("VARIABLE", 0, "Variable")]),
    **_build_metadata_dict("Entities", "#FFC0CB", "solid", [("LIBRARY", 1, "Library / Origin")]),
    **_build_metadata_dict("Data", "#c4deb0", "solid", [("DATASET", 2, "Dataset / Table")]),
    **_build_metadata_dict("Entities", "#b3b0de", "solid", [("UDF", 3, "UDF (Custom Logic)")]),
    **_build_metadata_dict("Entities", "#FFC0CB", "solid", [("MODEL", 4, "ML Model")]),
    **_build_metadata_dict("Data", "#B0C4DE", "solid", [("COLUMN", 5, "Column")]),
    **_build_metadata_dict("Data", "#a5f3fc", "dashed", [("HYPERPARAMETER", 6, "Hyperparameter")]),
    **_build_metadata_dict("Data", "#B0E0E6", "solid", [("LITERAL", 7, "Static Value")]),
    **_build_metadata_dict("Fallback", DEFAULT_NODE_COLOR, "solid", [("NOT_RELEVANT", 8, "Not Relevant")]),
}

DOMAIN_EDGES: dict[int, dict[str, str]] = {
    **_build_metadata_dict("Operations", "#FFA07A", "solid", [("MODEL_OPERATION", 0, "Model Op (Train/Pred)")]),
    **_build_metadata_dict("Operations", "#98FB98", "solid", [("DATA_IMPORT_EXTRACTION", 1, "Data Load")]),
    **_build_metadata_dict("Operations", "#d3d3d3", "solid", [("DATA_TRANSFORM", 2, "Transformation")]),
    **_build_metadata_dict("Operations", "#B0E0E6", "dotted", [("EDA", 3, "EDA / Inspection")]),
    **_build_metadata_dict("Operations", "#FFDAB9", "solid", [("DATA_EXPORT", 4, "Environment Export")]),
    **_build_metadata_dict("Fallback", DEFAULT_EDGE_COLOR, "solid", [("NOT_RELEVANT", 5, "Not Relevant")]),
}

AST_NODES: dict[int, dict[str, str]] = {
    # Modules & Imports (Reds)
    **_build_metadata_dict(
        "Modules & Imports", "#fca5a5", "solid", [("Module", 100, ""), ("Interactive", 101, ""), ("Expression", 102, ""), ("FunctionType", 103, ""), ("Import", 219, ""), ("ImportFrom", 220, "")]
    ),
    # Definitions (Purples)
    **_build_metadata_dict("Definitions", "#d8b4fe", "solid", [("FunctionDef", 200, ""), ("AsyncFunctionDef", 201, ""), ("Lambda", 304, "")]),
    **_build_metadata_dict("Definitions", "#fbcfe8", "solid", [("ClassDef", 202, "")]),
    # Control Flow (Yellows)
    **_build_metadata_dict(
        "Control Flow",
        "#fde68a",
        "solid",
        [
            ("Return", 203, ""),
            ("For", 208, ""),
            ("AsyncFor", 209, ""),
            ("While", 210, ""),
            ("If", 211, ""),
            ("With", 212, ""),
            ("AsyncWith", 213, ""),
            ("Match", 214, ""),
            ("Try", 216, ""),
            ("TryStar", 217, ""),
            ("Break", 225, ""),
            ("Continue", 226, ""),
            ("Await", 312, ""),
            ("Yield", 313, ""),
            ("YieldFrom", 314, ""),
        ],
    ),
    # Statements / Variables & Access (Blues)
    **_build_metadata_dict(
        "Statements",
        "#bfdbfe",
        "solid",
        [
            ("Delete", 204, ""),
            ("Assign", 205, ""),
            ("AugAssign", 206, ""),
            ("AnnAssign", 207, ""),
            ("Raise", 215, ""),
            ("Assert", 218, ""),
            ("Global", 221, ""),
            ("Nonlocal", 222, ""),
            ("Expr", 223, ""),
            ("Pass", 224, ""),
        ],
    ),
    **_build_metadata_dict("Variables & Access", "#bfdbfe", "solid", [("Attribute", 320, ""), ("Subscript", 321, ""), ("Starred", 322, ""), ("Name", 323, ""), ("Slice", 326, "")]),
    # Data Structures & Constants (Cyans)
    **_build_metadata_dict(
        "Data Structures", "#a5f3fc", "solid", [("Dict", 306, ""), ("Set", 307, ""), ("List", 324, ""), ("Tuple", 325, ""), ("Constant", 319, ""), ("FormattedValue", 317, ""), ("JoinedStr", 318, "")]
    ),
    # Calls, Comprehensions & Operations (Greens)
    **_build_metadata_dict("Calls & Comprehensions", "#bbf7d0", "solid", [("Call", 316, ""), ("ListComp", 308, ""), ("SetComp", 309, ""), ("DictComp", 310, ""), ("GeneratorExp", 311, "")]),
    **_build_metadata_dict(
        "Operations",
        "#bbf7d0",
        "solid",
        [
            ("BoolOp", 300, ""),
            ("NamedExpr", 301, ""),
            ("BinOp", 302, ""),
            ("UnaryOp", 303, ""),
            ("IfExp", 305, ""),
            ("Compare", 315, ""),
            ("Load", 400, ""),
            ("Store", 401, ""),
            ("Del", 402, ""),
            ("And", 410, ""),
            ("Or", 411, ""),
            ("Add", 412, ""),
            ("Sub", 413, ""),
            ("Mult", 414, ""),
            ("MatMult", 415, ""),
            ("Div", 416, ""),
            ("Mod", 417, ""),
            ("Pow", 418, ""),
            ("LShift", 419, ""),
            ("RShift", 420, ""),
            ("BitOr", 421, ""),
            ("BitXor", 422, ""),
            ("BitAnd", 423, ""),
            ("FloorDiv", 424, ""),
            ("Invert", 430, ""),
            ("Not", 431, ""),
            ("UAdd", 432, ""),
            ("USub", 433, ""),
            ("Eq", 440, ""),
            ("NotEq", 441, ""),
            ("Lt", 442, ""),
            ("LtE", 443, ""),
            ("Gt", 444, ""),
            ("GtE", 445, ""),
            ("Is", 446, ""),
            ("IsNot", 447, ""),
            ("In", 448, ""),
            ("NotIn", 449, ""),
        ],
    ),
    # Pattern Matching (Yellows)
    **_build_metadata_dict(
        "Pattern Matching",
        "#fde68a",
        "solid",
        [
            ("MatchValue", 500, ""),
            ("MatchSingleton", 501, ""),
            ("MatchSequence", 502, ""),
            ("MatchMapping", 503, ""),
            ("MatchClass", 504, ""),
            ("MatchStar", 505, ""),
            ("MatchAs", 506, ""),
            ("MatchOr", 507, ""),
        ],
    ),
    # Structural & Misc (Blues)
    **_build_metadata_dict(
        "Structural & Misc",
        "#bfdbfe",
        "solid",
        [
            ("arg", 600, ""),
            ("arguments", 601, ""),
            ("keyword", 602, ""),
            ("alias", 603, ""),
            ("withitem", 604, ""),
            ("match_case", 605, ""),
            ("comprehension", 606, ""),
            ("excepthandler", 607, ""),
            ("ExceptHandler", 608, ""),
        ],
    ),
    # Fallback (Red)
    **_build_metadata_dict("Fallback", "#ef4444", "dashed", [("AST_UNKNOWN", 999, "AST Unknown")]),
}

AST_EDGES: dict[int, dict[str, str]] = {
    **_build_metadata_dict("Edges", "#9ca3af", "solid", [("AST_PARENT_CHILD", 0, "Parent/Child")]),
}

VERBOSE = False

"""
Dataflow Node and Edge Types
"""
NODE_TYPES = {element["name"]: uid for uid, element in DATAFLOW_NODES.items()}

REVERSE_NODE_TYPES = {v: k for k, v in NODE_TYPES.items()}

EDGE_TYPES = {element["name"]: uid for uid, element in DATAFLOW_EDGES.items()}

REVERSE_EDGE_TYPES = {v: k for k, v in EDGE_TYPES.items()}

"""
Lineage Node and Edge Types
"""
DOMAIN_NODE_TYPES = {element["name"]: uid for uid, element in DOMAIN_NODES.items()}

REVERSE_DOMAIN_NODE_TYPES = {v: k for k, v in DOMAIN_NODE_TYPES.items()}

DOMAIN_EDGE_TYPES = {element["name"]: uid for uid, element in DOMAIN_EDGES.items()}

REVERSE_DOMAIN_EDGE_TYPES = {v: k for k, v in DOMAIN_EDGE_TYPES.items()}

"""
AST Node and Edge Types
"""
AST_NODE_TYPES = {element["name"]: uid for uid, element in AST_NODES.items()}

REVERSE_AST_NODE_TYPES = {v: k for k, v in AST_NODE_TYPES.items()}

AST_EDGE_TYPES = {element["name"]: uid for uid, element in AST_EDGES.items()}

REVERSE_AST_EDGE_TYPES = {v: k for k, v in AST_EDGE_TYPES.items()}
