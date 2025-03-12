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

DOMAIN_EDGE_TYPES = {
    "MODEL_TRAIN": 0,
    "MODEL_EVALUATION": 1,
    "HYPERPARAMETER_TUNING": 2,
    "DATA_EXPORT": 3,
    "DATA_IMPORT_EXTRACTION": 4,
    "DATA_TRANSFORM": 5,
    "EDA": 6,
    "ENVIRONMENT": 7,
    "NOT_INTERESTING": 8,
}

VERBOSE = False

REVERSE_NODE_TYPES = {v: k for k, v in NODE_TYPES.items()}
REVERSE_EDGE_TYPES = {v: k for k, v in EDGE_TYPES.items()}
