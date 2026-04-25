// config/GraphTheme.ts

export const DEFAULT_NODE_COLOR = '#B0E0E6';
export const DEFAULT_EDGE_COLOR = '#d3d3d3';

export const SEMANTIC_COLORS = {
  nodes: {
    VARIABLE: '#B0C4DE',
    IMPORT: '#FFC0CB',
    DATASET: '#c4deb0',
    FUNCTION: '#c4deb0',
    UDF: '#b3b0de',
    INTERMEDIATE: '#b3b0de',
    IF: '#DEDAB0',
    LOOP: '#B0DEB9',
    CLASS: '#DEB0DE',
    LITERAL: '#B0E0E6',
    MODEL: '#FFC0CB',
    // AST
    // Control Flow (Yellows/Oranges)
    If: '#fde68a',
    For: '#fde68a',
    While: '#fde68a',
    Try: '#fde68a',
    Match: '#fde68a',
    FunctionDef: '#d8b4fe',
    AsyncFunctionDef: '#d8b4fe',
    ClassDef: '#fbcfe8',

    // Variables & Assignments (Blues)
    Assign: '#bfdbfe',
    Name: '#bfdbfe',
    Constant: '#a5f3fc',

    // Calls & Operations (Greens)
    Call: '#bbf7d0',
    BinOp: '#bbf7d0',
    Compare: '#bbf7d0',

    // 5. Imports (Reds)
    Import: '#fca5a5',
    ImportFrom: '#fca5a5',

    AST_UNKNOWN: '#ef4444',
    NOT_RELEVANT: DEFAULT_EDGE_COLOR
  } as Record<string, string>,

  edges: {
    CALLER: '#FFA07A',
    REASSIGN: '#d3d3d3',
    INPUT: '#98FB98',
    BRANCH: '#B0E0E6',
    LOOP: '#FFDAB9',
    FUNCTION_CALL: '#c4deb0',
    MODEL_TRAIN: '#FFA07A',
    DATA_IMPORT_EXTRACTION: '#98FB98',
    DATA_TRANSFORM: '#d3d3d3',
    ENVIRONMENT_EXPORT: '#FFDAB9',
    AST_PARENT_CHILD: '#9ca3af',
    NOT_RELEVANT: DEFAULT_EDGE_COLOR
  } as Record<string, string>
};
