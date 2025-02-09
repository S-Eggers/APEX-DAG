NODE_TYPES = {
    "VARIABLE": 0,
    "IMPORT": 1,
    "LITERAL": 2,
    "FUNCTION": 3,
    "INTERMEDIATE": 4,
    "IF": 5,
    "LOOP": 6,
    "CLASS": 7,
}

EDGE_TYPES = {
    "CALLER": 0,
    "INPUT": 1,
    "OMITTED": 2,
    "BRANCH": 3,
    "LOOP": 4,
    "FUNCTION_CALL": 5,
}

VERBOSE = False

REVERSE_NODE_TYPES = {v: k for k, v in NODE_TYPES.items()}
REVERSE_EDGE_TYPES = {v: k for k, v in EDGE_TYPES.items()}
