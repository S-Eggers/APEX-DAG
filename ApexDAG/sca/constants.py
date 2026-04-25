VERBOSE = False

"""
Dataflow Node and Edge Types
"""
NODE_TYPES = {
    "VARIABLE": 0,
    "IMPORT": 1,
    "DATASET": 2,
    "FUNCTION": 3,
    "INTERMEDIATE": 4,
    "IF": 5,
    "LOOP": 6,
    "CLASS": 7,
    "LITERAL": 8,
}

REVERSE_NODE_TYPES = {v: k for k, v in NODE_TYPES.items()}

EDGE_TYPES = {
    "CALLER": 0,
    "INPUT": 1,
    "OMITTED": 2,
    "BRANCH": 3,
    "LOOP": 4,
    "FUNCTION_CALL": 5,
    "CLASS_CALL": 6,
}

REVERSE_EDGE_TYPES = {v: k for k, v in EDGE_TYPES.items()}

"""
Lineage Node and Edge Types
"""
DOMAIN_NODE_TYPES = {
    "VARIABLE": 0,
    "LIBRARY": 1,
    "DATASET": 2,
    "UDF": 3,
    "MODEL": 4,
    "COLUMN": 5,
    "HYPERPARAMETER": 6,
    "LITERAL": 7,
    "NOT_RELEVANT": 8,
}

REVERSE_DOMAIN_NODE_TYPES = {v: k for k, v in DOMAIN_NODE_TYPES.items()}

DOMAIN_EDGE_TYPES = {
    "MODEL_OPERATION": 0, # Train/Eval/Tune/Predict
    "DATA_IMPORT_EXTRACTION": 1,
    "DATA_TRANSFORM": 2,
    "EDA": 3,
    "DATA_EXPORT": 4,
    "NOT_RELEVANT": 5,
}

REVERSE_DOMAIN_EDGE_TYPES = {v: k for k, v in DOMAIN_EDGE_TYPES.items()}

"""
AST Node and Edge Types
"""
AST_NODE_TYPES = {
    # Modules
    "Module": 100, "Interactive": 101, "Expression": 102, "FunctionType": 103,
    
    # Statements
    "FunctionDef": 200, "AsyncFunctionDef": 201, "ClassDef": 202, "Return": 203,
    "Delete": 204, "Assign": 205, "AugAssign": 206, "AnnAssign": 207,
    "For": 208, "AsyncFor": 209, "While": 210, "If": 211, "With": 212,
    "AsyncWith": 213, "Match": 214, "Raise": 215, "Try": 216, "TryStar": 217,
    "Assert": 218, "Import": 219, "ImportFrom": 220, "Global": 221, "Nonlocal": 222,
    "Expr": 223, "Pass": 224, "Break": 225, "Continue": 226,
    
    # Expressions
    "BoolOp": 300, "NamedExpr": 301, "BinOp": 302, "UnaryOp": 303, "Lambda": 304,
    "IfExp": 305, "Dict": 306, "Set": 307, "ListComp": 308, "SetComp": 309,
    "DictComp": 310, "GeneratorExp": 311, "Await": 312, "Yield": 313,
    "YieldFrom": 314, "Compare": 315, "Call": 316, "FormattedValue": 317,
    "JoinedStr": 318, "Constant": 319, "Attribute": 320, "Subscript": 321,
    "Starred": 322, "Name": 323, "List": 324, "Tuple": 325, "Slice": 326,
    
    # Operators & Contexts
    "Load": 400, "Store": 401, "Del": 402,
    "And": 410, "Or": 411, "Add": 412, "Sub": 413, "Mult": 414, "MatMult": 415,
    "Div": 416, "Mod": 417, "Pow": 418, "LShift": 419, "RShift": 420,
    "BitOr": 421, "BitXor": 422, "BitAnd": 423, "FloorDiv": 424,
    "Invert": 430, "Not": 431, "UAdd": 432, "USub": 433,
    "Eq": 440, "NotEq": 441, "Lt": 442, "LtE": 443, "Gt": 444, "GtE": 445,
    "Is": 446, "IsNot": 447, "In": 448, "NotIn": 449,
    
    # Pattern Matching
    "MatchValue": 500, "MatchSingleton": 501, "MatchSequence": 502,
    "MatchMapping": 503, "MatchClass": 504, "MatchStar": 505,
    "MatchAs": 506, "MatchOr": 507,
    
    # Structural & Misc
    "arg": 600, "arguments": 601, "keyword": 602, "alias": 603,
    "withitem": 604, "match_case": 605, "comprehension": 606,
    "excepthandler": 607, "ExceptHandler": 608,
    
    # Fallback
    "AST_UNKNOWN": 999
}

REVERSE_AST_NODE_TYPES = {v: k for k, v in AST_NODE_TYPES.items()}

AST_EDGE_TYPES = {
    "AST_PARENT_CHILD": 0
}

REVERSE_AST_EDGE_TYPES = {v: k for k, v in AST_EDGE_TYPES.items()}