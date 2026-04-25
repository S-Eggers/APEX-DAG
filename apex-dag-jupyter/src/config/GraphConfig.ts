import { GraphMode, LegendItemType } from '../types/GraphTypes';
import { SEMANTIC_COLORS } from './GraphTheme';

const { nodes: nColors, edges: eColors } = SEMANTIC_COLORS;

const sharedLineageConfig = {
  legends: [
    {
      type: 'node',
      color: SEMANTIC_COLORS.nodes.VARIABLE,
      label: 'Variable',
      borderStyle: 'solid',
      numericType: 0,
      category: 'Entities'
    },
    {
      type: 'node',
      color: SEMANTIC_COLORS.nodes.IMPORT,
      label: 'Library / Origin',
      borderStyle: 'solid',
      numericType: 1,
      category: 'Entities'
    },
    {
      type: 'node',
      color: SEMANTIC_COLORS.nodes.UDF,
      label: 'UDF (Custom Logic)',
      borderStyle: 'solid',
      numericType: 3,
      category: 'Entities'
    },
    {
      type: 'node',
      color: SEMANTIC_COLORS.nodes.MODEL,
      label: 'ML Model',
      borderStyle: 'solid',
      numericType: 4,
      category: 'Entities'
    },
    {
      type: 'node',
      color: SEMANTIC_COLORS.nodes.DATASET,
      label: 'Dataset / Table',
      borderStyle: 'solid',
      numericType: 2,
      category: 'Data'
    },
    {
      type: 'node',
      color: SEMANTIC_COLORS.nodes.Constant,
      label: 'Hyperparameter',
      borderStyle: 'dashed',
      numericType: 6,
      category: 'Data'
    },
    {
      type: 'node',
      color: SEMANTIC_COLORS.nodes.LITERAL,
      label: 'Static Value',
      borderStyle: 'solid',
      numericType: 7,
      category: 'Data'
    },
    {
      type: 'edge',
      color: SEMANTIC_COLORS.edges.MODEL_TRAIN,
      label: 'Model Op (Train/Pred)',
      borderStyle: 'solid',
      numericType: 0,
      category: 'Operations'
    },
    {
      type: 'edge',
      color: SEMANTIC_COLORS.edges.DATA_IMPORT_EXTRACTION,
      label: 'Data Load',
      borderStyle: 'solid',
      numericType: 1,
      category: 'Operations'
    },
    {
      type: 'edge',
      color: SEMANTIC_COLORS.edges.DATA_TRANSFORM,
      label: 'Transformation',
      borderStyle: 'solid',
      numericType: 2,
      category: 'Operations'
    },
    {
      type: 'edge',
      color: SEMANTIC_COLORS.edges.BRANCH,
      label: 'EDA / Inspection',
      borderStyle: 'dotted',
      numericType: 3,
      category: 'Operations'
    }
  ] as LegendItemType[],

  getNodeColor: (type: number) => {
    const map: Record<number, string> = {
      0: SEMANTIC_COLORS.nodes.VARIABLE,
      1: SEMANTIC_COLORS.nodes.IMPORT,
      2: SEMANTIC_COLORS.nodes.DATASET,
      3: SEMANTIC_COLORS.nodes.UDF,
      4: SEMANTIC_COLORS.nodes.MODEL,
      5: SEMANTIC_COLORS.nodes.Name,
      6: SEMANTIC_COLORS.nodes.Constant,
      7: SEMANTIC_COLORS.nodes.LITERAL,
      8: SEMANTIC_COLORS.nodes.NOT_RELEVANT
    };
    return map[type] || SEMANTIC_COLORS.nodes.VARIABLE;
  },

  getEdgeColor: (type: number) => {
    const map: Record<number, string> = {
      0: SEMANTIC_COLORS.edges.MODEL_TRAIN,
      1: SEMANTIC_COLORS.edges.DATA_IMPORT_EXTRACTION,
      2: SEMANTIC_COLORS.edges.DATA_TRANSFORM,
      3: SEMANTIC_COLORS.edges.BRANCH,
      4: SEMANTIC_COLORS.edges.ENVIRONMENT_EXPORT,
      5: SEMANTIC_COLORS.edges.FUNCTION_CALL,
      6: SEMANTIC_COLORS.edges.NOT_RELEVANT
    };
    return map[type] || SEMANTIC_COLORS.edges.DATA_TRANSFORM;
  }
};

const EXHAUSTIVE_AST_LEGENDS: LegendItemType[] = [
  // Modules & Imports
  {
    type: 'node',
    color: nColors.Import,
    label: 'Module',
    borderStyle: 'solid',
    numericType: 100,
    category: 'Modules & Imports'
  },
  {
    type: 'node',
    color: nColors.Import,
    label: 'Interactive',
    borderStyle: 'solid',
    numericType: 101,
    category: 'Modules & Imports'
  },
  {
    type: 'node',
    color: nColors.Import,
    label: 'Expression',
    borderStyle: 'solid',
    numericType: 102,
    category: 'Modules & Imports'
  },
  {
    type: 'node',
    color: nColors.Import,
    label: 'FunctionType',
    borderStyle: 'solid',
    numericType: 103,
    category: 'Modules & Imports'
  },
  {
    type: 'node',
    color: nColors.Import,
    label: 'Import',
    borderStyle: 'solid',
    numericType: 219,
    category: 'Modules & Imports'
  },
  {
    type: 'node',
    color: nColors.ImportFrom,
    label: 'ImportFrom',
    borderStyle: 'solid',
    numericType: 220,
    category: 'Modules & Imports'
  },
  // Definitions
  {
    type: 'node',
    color: nColors.FunctionDef,
    label: 'FunctionDef',
    borderStyle: 'solid',
    numericType: 200,
    category: 'Definitions'
  },
  {
    type: 'node',
    color: nColors.AsyncFunctionDef,
    label: 'AsyncFunctionDef',
    borderStyle: 'solid',
    numericType: 201,
    category: 'Definitions'
  },
  {
    type: 'node',
    color: nColors.ClassDef,
    label: 'ClassDef',
    borderStyle: 'solid',
    numericType: 202,
    category: 'Definitions'
  },
  {
    type: 'node',
    color: nColors.FunctionDef,
    label: 'Lambda',
    borderStyle: 'solid',
    numericType: 304,
    category: 'Definitions'
  },
  // Control Flow
  {
    type: 'node',
    color: nColors.If,
    label: 'Return',
    borderStyle: 'solid',
    numericType: 203,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: nColors.For,
    label: 'For',
    borderStyle: 'solid',
    numericType: 208,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: nColors.For,
    label: 'AsyncFor',
    borderStyle: 'solid',
    numericType: 209,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: nColors.While,
    label: 'While',
    borderStyle: 'solid',
    numericType: 210,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: nColors.If,
    label: 'If',
    borderStyle: 'solid',
    numericType: 211,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: nColors.Try,
    label: 'With',
    borderStyle: 'solid',
    numericType: 212,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: nColors.Try,
    label: 'AsyncWith',
    borderStyle: 'solid',
    numericType: 213,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: nColors.Match,
    label: 'Match',
    borderStyle: 'solid',
    numericType: 214,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: nColors.Try,
    label: 'Try',
    borderStyle: 'solid',
    numericType: 216,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: nColors.Try,
    label: 'TryStar',
    borderStyle: 'solid',
    numericType: 217,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: nColors.If,
    label: 'Break',
    borderStyle: 'solid',
    numericType: 225,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: nColors.If,
    label: 'Continue',
    borderStyle: 'solid',
    numericType: 226,
    category: 'Control Flow'
  },
  // Statements
  {
    type: 'node',
    color: nColors.Assign,
    label: 'Delete',
    borderStyle: 'solid',
    numericType: 204,
    category: 'Statements'
  },
  {
    type: 'node',
    color: nColors.Assign,
    label: 'Assign',
    borderStyle: 'solid',
    numericType: 205,
    category: 'Statements'
  },
  {
    type: 'node',
    color: nColors.Assign,
    label: 'AugAssign',
    borderStyle: 'solid',
    numericType: 206,
    category: 'Statements'
  },
  {
    type: 'node',
    color: nColors.Assign,
    label: 'AnnAssign',
    borderStyle: 'solid',
    numericType: 207,
    category: 'Statements'
  },
  {
    type: 'node',
    color: nColors.Assign,
    label: 'Raise',
    borderStyle: 'solid',
    numericType: 215,
    category: 'Statements'
  },
  {
    type: 'node',
    color: nColors.Assign,
    label: 'Assert',
    borderStyle: 'solid',
    numericType: 218,
    category: 'Statements'
  },
  {
    type: 'node',
    color: nColors.Assign,
    label: 'Global',
    borderStyle: 'solid',
    numericType: 221,
    category: 'Statements'
  },
  {
    type: 'node',
    color: nColors.Assign,
    label: 'Nonlocal',
    borderStyle: 'solid',
    numericType: 222,
    category: 'Statements'
  },
  {
    type: 'node',
    color: nColors.Assign,
    label: 'Expr',
    borderStyle: 'solid',
    numericType: 223,
    category: 'Statements'
  },
  {
    type: 'node',
    color: nColors.Assign,
    label: 'Pass',
    borderStyle: 'solid',
    numericType: 224,
    category: 'Statements'
  },
  // Operations
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'BoolOp',
    borderStyle: 'solid',
    numericType: 300,
    category: 'Operations'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'NamedExpr',
    borderStyle: 'solid',
    numericType: 301,
    category: 'Operations'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'BinOp',
    borderStyle: 'solid',
    numericType: 302,
    category: 'Operations'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'UnaryOp',
    borderStyle: 'solid',
    numericType: 303,
    category: 'Operations'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'IfExp',
    borderStyle: 'solid',
    numericType: 305,
    category: 'Operations'
  },
  {
    type: 'node',
    color: nColors.Call,
    label: 'Call',
    borderStyle: 'solid',
    numericType: 316,
    category: 'Calls & Arguments'
  },
  {
    type: 'node',
    color: nColors.Compare,
    label: 'Compare',
    borderStyle: 'solid',
    numericType: 315,
    category: 'Operations'
  },
  {
    type: 'node',
    color: nColors.If,
    label: 'Await',
    borderStyle: 'solid',
    numericType: 312,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: nColors.If,
    label: 'Yield',
    borderStyle: 'solid',
    numericType: 313,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: nColors.If,
    label: 'YieldFrom',
    borderStyle: 'solid',
    numericType: 314,
    category: 'Control Flow'
  },
  // Data Structures
  {
    type: 'node',
    color: nColors.Constant,
    label: 'Dict',
    borderStyle: 'solid',
    numericType: 306,
    category: 'Data Structures'
  },
  {
    type: 'node',
    color: nColors.Constant,
    label: 'Set',
    borderStyle: 'solid',
    numericType: 307,
    category: 'Data Structures'
  },
  {
    type: 'node',
    color: nColors.Constant,
    label: 'List',
    borderStyle: 'solid',
    numericType: 324,
    category: 'Data Structures'
  },
  {
    type: 'node',
    color: nColors.Constant,
    label: 'Tuple',
    borderStyle: 'solid',
    numericType: 325,
    category: 'Data Structures'
  },
  {
    type: 'node',
    color: nColors.Call,
    label: 'ListComp',
    borderStyle: 'solid',
    numericType: 308,
    category: 'Comprehensions'
  },
  {
    type: 'node',
    color: nColors.Call,
    label: 'SetComp',
    borderStyle: 'solid',
    numericType: 309,
    category: 'Comprehensions'
  },
  {
    type: 'node',
    color: nColors.Call,
    label: 'DictComp',
    borderStyle: 'solid',
    numericType: 310,
    category: 'Comprehensions'
  },
  {
    type: 'node',
    color: nColors.Call,
    label: 'GeneratorExp',
    borderStyle: 'solid',
    numericType: 311,
    category: 'Comprehensions'
  },
  // Variables & Access
  {
    type: 'node',
    color: nColors.Constant,
    label: 'Constant',
    borderStyle: 'solid',
    numericType: 319,
    category: 'Variables & Access'
  },
  {
    type: 'node',
    color: nColors.Constant,
    label: 'FormattedValue',
    borderStyle: 'solid',
    numericType: 317,
    category: 'Variables & Access'
  },
  {
    type: 'node',
    color: nColors.Constant,
    label: 'JoinedStr',
    borderStyle: 'solid',
    numericType: 318,
    category: 'Variables & Access'
  },
  {
    type: 'node',
    color: nColors.Name,
    label: 'Attribute',
    borderStyle: 'solid',
    numericType: 320,
    category: 'Variables & Access'
  },
  {
    type: 'node',
    color: nColors.Name,
    label: 'Subscript',
    borderStyle: 'solid',
    numericType: 321,
    category: 'Variables & Access'
  },
  {
    type: 'node',
    color: nColors.Name,
    label: 'Starred',
    borderStyle: 'solid',
    numericType: 322,
    category: 'Variables & Access'
  },
  {
    type: 'node',
    color: nColors.Name,
    label: 'Name',
    borderStyle: 'solid',
    numericType: 323,
    category: 'Variables & Access'
  },
  {
    type: 'node',
    color: nColors.Name,
    label: 'Slice',
    borderStyle: 'solid',
    numericType: 326,
    category: 'Variables & Access'
  },
  // Operators & Contexts
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'Load',
    borderStyle: 'solid',
    numericType: 400,
    category: 'Contexts'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'Store',
    borderStyle: 'solid',
    numericType: 401,
    category: 'Contexts'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'Del',
    borderStyle: 'solid',
    numericType: 402,
    category: 'Contexts'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'And',
    borderStyle: 'solid',
    numericType: 410,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'Or',
    borderStyle: 'solid',
    numericType: 411,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'Add',
    borderStyle: 'solid',
    numericType: 412,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'Sub',
    borderStyle: 'solid',
    numericType: 413,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'Mult',
    borderStyle: 'solid',
    numericType: 414,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'MatMult',
    borderStyle: 'solid',
    numericType: 415,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'Div',
    borderStyle: 'solid',
    numericType: 416,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'Mod',
    borderStyle: 'solid',
    numericType: 417,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'Pow',
    borderStyle: 'solid',
    numericType: 418,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'LShift',
    borderStyle: 'solid',
    numericType: 419,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'RShift',
    borderStyle: 'solid',
    numericType: 420,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'BitOr',
    borderStyle: 'solid',
    numericType: 421,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'BitXor',
    borderStyle: 'solid',
    numericType: 422,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'BitAnd',
    borderStyle: 'solid',
    numericType: 423,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'FloorDiv',
    borderStyle: 'solid',
    numericType: 424,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'Invert',
    borderStyle: 'solid',
    numericType: 430,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'Not',
    borderStyle: 'solid',
    numericType: 431,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'UAdd',
    borderStyle: 'solid',
    numericType: 432,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.BinOp,
    label: 'USub',
    borderStyle: 'solid',
    numericType: 433,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.Compare,
    label: 'Eq',
    borderStyle: 'solid',
    numericType: 440,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.Compare,
    label: 'NotEq',
    borderStyle: 'solid',
    numericType: 441,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.Compare,
    label: 'Lt',
    borderStyle: 'solid',
    numericType: 442,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.Compare,
    label: 'LtE',
    borderStyle: 'solid',
    numericType: 443,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.Compare,
    label: 'Gt',
    borderStyle: 'solid',
    numericType: 444,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.Compare,
    label: 'GtE',
    borderStyle: 'solid',
    numericType: 445,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.Compare,
    label: 'Is',
    borderStyle: 'solid',
    numericType: 446,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.Compare,
    label: 'IsNot',
    borderStyle: 'solid',
    numericType: 447,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.Compare,
    label: 'In',
    borderStyle: 'solid',
    numericType: 448,
    category: 'Operators'
  },
  {
    type: 'node',
    color: nColors.Compare,
    label: 'NotIn',
    borderStyle: 'solid',
    numericType: 449,
    category: 'Operators'
  },
  // Pattern Matching
  {
    type: 'node',
    color: nColors.Match,
    label: 'MatchValue',
    borderStyle: 'solid',
    numericType: 500,
    category: 'Pattern Matching'
  },
  {
    type: 'node',
    color: nColors.Match,
    label: 'MatchSingleton',
    borderStyle: 'solid',
    numericType: 501,
    category: 'Pattern Matching'
  },
  {
    type: 'node',
    color: nColors.Match,
    label: 'MatchSequence',
    borderStyle: 'solid',
    numericType: 502,
    category: 'Pattern Matching'
  },
  {
    type: 'node',
    color: nColors.Match,
    label: 'MatchMapping',
    borderStyle: 'solid',
    numericType: 503,
    category: 'Pattern Matching'
  },
  {
    type: 'node',
    color: nColors.Match,
    label: 'MatchClass',
    borderStyle: 'solid',
    numericType: 504,
    category: 'Pattern Matching'
  },
  {
    type: 'node',
    color: nColors.Match,
    label: 'MatchStar',
    borderStyle: 'solid',
    numericType: 505,
    category: 'Pattern Matching'
  },
  {
    type: 'node',
    color: nColors.Match,
    label: 'MatchAs',
    borderStyle: 'solid',
    numericType: 506,
    category: 'Pattern Matching'
  },
  {
    type: 'node',
    color: nColors.Match,
    label: 'MatchOr',
    borderStyle: 'solid',
    numericType: 507,
    category: 'Pattern Matching'
  },
  // Structural & Misc
  {
    type: 'node',
    color: nColors.Assign,
    label: 'arg',
    borderStyle: 'solid',
    numericType: 600,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: nColors.Assign,
    label: 'arguments',
    borderStyle: 'solid',
    numericType: 601,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: nColors.Assign,
    label: 'keyword',
    borderStyle: 'solid',
    numericType: 602,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: nColors.Assign,
    label: 'alias',
    borderStyle: 'solid',
    numericType: 603,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: nColors.Try,
    label: 'withitem',
    borderStyle: 'solid',
    numericType: 604,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: nColors.Match,
    label: 'match_case',
    borderStyle: 'solid',
    numericType: 605,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: nColors.Call,
    label: 'comprehension',
    borderStyle: 'solid',
    numericType: 606,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: nColors.Try,
    label: 'excepthandler',
    borderStyle: 'solid',
    numericType: 607,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: nColors.Try,
    label: 'ExceptHandler',
    borderStyle: 'solid',
    numericType: 608,
    category: 'Structural & Misc'
  },
  // Fallback
  {
    type: 'node',
    color: nColors.AST_UNKNOWN,
    label: 'AST_UNKNOWN',
    borderStyle: 'dashed',
    numericType: 999,
    category: 'Fallback'
  },
  // Edges
  {
    type: 'edge',
    color: eColors.AST_PARENT_CHILD,
    label: 'Parent/Child',
    borderStyle: 'solid',
    numericType: 0,
    category: 'Edges'
  }
];

const AST_NODE_COLOR_MAP: Record<number, string> = Object.fromEntries(
  EXHAUSTIVE_AST_LEGENDS.filter(item => item.type === 'node').map(item => [
    item.numericType,
    item.color
  ])
);

export const MODE_CONFIG: Record<
  GraphMode,
  {
    legends: LegendItemType[];
    getNodeColor: (type: number) => string;
    getEdgeColor: (type: number) => string;
  }
> = {
  dataflow: {
    legends: [
      {
        type: 'node',
        color: nColors.VARIABLE,
        label: 'Variable',
        borderStyle: 'solid',
        numericType: 0,
        category: 'State'
      },
      {
        type: 'node',
        color: nColors.INTERMEDIATE,
        label: 'Intermediate',
        borderStyle: 'solid',
        numericType: 4,
        category: 'State'
      },
      {
        type: 'node',
        color: nColors.LITERAL,
        label: 'Literal',
        borderStyle: 'solid',
        numericType: 8,
        category: 'State'
      },
      {
        type: 'node',
        color: nColors.FUNCTION,
        label: 'Function',
        borderStyle: 'solid',
        numericType: 3,
        category: 'Definitions'
      },
      {
        type: 'node',
        color: nColors.CLASS,
        label: 'Class',
        borderStyle: 'solid',
        numericType: 7,
        category: 'Definitions'
      },
      {
        type: 'node',
        color: nColors.IMPORT,
        label: 'Import',
        borderStyle: 'solid',
        numericType: 1,
        category: 'Definitions'
      },
      {
        type: 'node',
        color: nColors.IF,
        label: 'If',
        borderStyle: 'solid',
        numericType: 5,
        category: 'Control Flow'
      },
      {
        type: 'node',
        color: nColors.LOOP,
        label: 'Loop',
        borderStyle: 'solid',
        numericType: 6,
        category: 'Control Flow'
      },
      {
        type: 'edge',
        color: eColors.CALLER,
        label: 'Caller',
        borderStyle: 'solid',
        numericType: 0,
        category: 'Edges'
      },
      {
        type: 'edge',
        color: eColors.INPUT,
        label: 'Input',
        borderStyle: 'solid',
        numericType: 1,
        category: 'Edges'
      },
      {
        type: 'edge',
        color: eColors.REASSIGN,
        label: 'Reassign',
        borderStyle: 'dashed',
        numericType: 2,
        category: 'Edges'
      },
      {
        type: 'edge',
        color: eColors.BRANCH,
        label: 'Branch',
        borderStyle: 'solid',
        numericType: 3,
        category: 'Edges'
      },
      {
        type: 'edge',
        color: eColors.LOOP,
        label: 'Loop',
        borderStyle: 'solid',
        numericType: 4,
        category: 'Edges'
      },
      {
        type: 'edge',
        color: eColors.FUNCTION_CALL,
        label: 'Function',
        borderStyle: 'solid',
        numericType: 5,
        category: 'Edges'
      }
    ],
    getNodeColor: type => {
      const map: Record<number, string> = {
        0: nColors.VARIABLE,
        1: nColors.IMPORT,
        3: nColors.FUNCTION,
        4: nColors.INTERMEDIATE,
        5: nColors.IF,
        6: nColors.LOOP,
        7: nColors.CLASS,
        8: nColors.LITERAL
      };
      return map[type] || nColors.VARIABLE;
    },
    getEdgeColor: type => {
      const map: Record<number, string> = {
        0: eColors.CALLER,
        1: eColors.INPUT,
        2: eColors.REASSIGN,
        3: eColors.BRANCH,
        4: eColors.LOOP,
        5: eColors.FUNCTION_CALL
      };
      return map[type] || eColors.CALLER;
    }
  },

  lineage: sharedLineageConfig,
  vamsa: {
    legends: [
      {
        type: 'node',
        color: nColors.VARIABLE,
        label: 'Data Node',
        borderStyle: 'solid',
        numericType: 0,
        category: 'Vamsa Entities'
      },
      {
        type: 'node',
        color: nColors.FUNCTION,
        label: 'Operation',
        borderStyle: 'solid',
        numericType: 3,
        category: 'Vamsa Entities'
      },
      {
        type: 'edge',
        color: eColors.INPUT,
        label: 'Input Flow',
        borderStyle: 'solid',
        numericType: 1,
        category: 'Vamsa Edges'
      },
      {
        type: 'edge',
        color: eColors.CALLER,
        label: 'Caller Flow',
        borderStyle: 'solid',
        numericType: 0,
        category: 'Vamsa Edges'
      },
      {
        type: 'edge',
        color: eColors.DATA_TRANSFORM,
        label: 'Output Flow',
        borderStyle: 'solid',
        numericType: 2,
        category: 'Vamsa Edges'
      }
    ],
    getNodeColor: type => {
      const map: Record<number, string> = {
        0: nColors.VARIABLE,
        3: nColors.FUNCTION
      };
      return map[type] || nColors.VARIABLE;
    },
    getEdgeColor: type => {
      const map: Record<number, string> = {
        0: eColors.CALLER,
        1: eColors.INPUT,
        2: eColors.DATA_TRANSFORM
      };
      return map[type] || eColors.DATA_TRANSFORM;
    }
  },
  labeling: sharedLineageConfig,

  ast: {
    legends: EXHAUSTIVE_AST_LEGENDS,
    getNodeColor: type => AST_NODE_COLOR_MAP[type] || nColors.AST_UNKNOWN,
    getEdgeColor: type => eColors.AST_PARENT_CHILD
  }
};

export const groupLegendItems = (
  allLegendItems: LegendItemType[]
): Record<string, LegendItemType[]> => {
  const grouped: Record<string, LegendItemType[]> = {};
  allLegendItems.forEach(item => {
    const cat = item.category || 'Uncategorized';
    if (!grouped[cat]) grouped[cat] = [];
    grouped[cat].push(item);
  });
  return grouped;
};

export const filterLegendItems = (
  elements: any[],
  allLegendItems: LegendItemType[]
): LegendItemType[] => {
  if (!elements || elements.length === 0) return [];

  const presentNodeTypes = new Set<number>();
  const presentEdgeTypes = new Set<number>();

  elements.forEach(element => {
    const data = element.data;

    if (data.node_type !== undefined && data.node_type !== null) {
      presentNodeTypes.add(Number(data.node_type));
    } else if (
      data.predicted_label !== undefined &&
      data.predicted_label !== null
    ) {
      presentEdgeTypes.add(Number(data.predicted_label));
    } else if (data.edge_type !== undefined && data.edge_type !== null) {
      presentEdgeTypes.add(Number(data.edge_type));
    }
  });

  return allLegendItems.filter(item => {
    const targetType = Number(item.numericType);
    return item.type === 'node'
      ? presentNodeTypes.has(targetType)
      : presentEdgeTypes.has(targetType);
  });
};
