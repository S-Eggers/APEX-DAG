import { GraphMode, LegendItemType } from '../types/GraphTypes';

export const colors = {
  light_steel_blue: '#B0C4DE',
  very_soft_blue: '#b3b0de',
  pink: '#FFC0CB',
  light_green: '#c4deb0',
  very_soft_yellow: '#DEDAB0',
  very_soft_purple: '#DEB0DE',
  very_soft_lime_green: '#B0DEB9',
  light_salmon: '#FFA07A',
  pale_green: '#98FB98',
  gray: '#d3d3d3',
  powder_blue: '#B0E0E6',
  peach_puff: '#FFDAB9',
  dark_slate: '#2F4F4F'
};

const sharedLineageConfig = {
  legends: [
    {
      type: 'node',
      color: colors.light_steel_blue,
      label: 'Variable',
      borderStyle: 'solid',
      numericType: 0,
      category: 'Entities'
    },
    {
      type: 'node',
      color: colors.pink,
      label: 'Library',
      borderStyle: 'solid',
      numericType: 1,
      category: 'Entities'
    },
    {
      type: 'node',
      color: colors.light_green,
      label: 'Dataset',
      borderStyle: 'solid',
      numericType: 2,
      category: 'Data'
    },
    {
      type: 'node',
      color: colors.very_soft_blue,
      label: 'UDF',
      borderStyle: 'solid',
      numericType: 3,
      category: 'Entities'
    },
    {
      type: 'node',
      color: colors.powder_blue,
      label: 'Literal',
      borderStyle: 'solid',
      numericType: 8,
      category: 'Data'
    },
    {
      type: 'edge',
      color: colors.light_salmon,
      label: 'Model Train/Eval',
      borderStyle: 'solid',
      numericType: 0,
      category: 'Operations'
    },
    {
      type: 'edge',
      color: colors.peach_puff,
      label: 'Environment+Data Export',
      borderStyle: 'solid',
      numericType: 4,
      category: 'Operations'
    },
    {
      type: 'edge',
      color: colors.powder_blue,
      label: 'EDA',
      borderStyle: 'solid',
      numericType: 3,
      category: 'Operations'
    },
    {
      type: 'edge',
      color: colors.pale_green,
      label: 'Data Import',
      borderStyle: 'solid',
      numericType: 1,
      category: 'Operations'
    },
    {
      type: 'edge',
      color: colors.gray,
      label: 'Data Transform',
      borderStyle: 'solid',
      numericType: 2,
      category: 'Operations'
    }
  ] as LegendItemType[],
  getNodeColor: (type: number) => {
    const map: Record<number, string> = {
      0: colors.light_steel_blue,
      1: colors.pink,
      2: colors.light_green,
      3: colors.very_soft_blue,
      4: colors.very_soft_yellow,
      5: colors.very_soft_purple,
      7: colors.very_soft_lime_green,
      8: colors.powder_blue
    };
    return map[type] || colors.light_steel_blue;
  },
  getEdgeColor: (type: number) => {
    const map: Record<number, string> = {
      0: colors.light_salmon,
      1: colors.pale_green,
      2: colors.gray,
      3: colors.powder_blue,
      4: colors.peach_puff,
      5: colors.light_green
    };
    return map[type] || '#000';
  }
};

const EXHAUSTIVE_AST_LEGENDS: LegendItemType[] = [
  // Modules & Imports
  {
    type: 'node',
    color: colors.very_soft_purple,
    label: 'Module',
    borderStyle: 'solid',
    numericType: 100,
    category: 'Modules & Imports'
  },
  {
    type: 'node',
    color: colors.very_soft_purple,
    label: 'Interactive',
    borderStyle: 'solid',
    numericType: 101,
    category: 'Modules & Imports'
  },
  {
    type: 'node',
    color: colors.very_soft_purple,
    label: 'Expression',
    borderStyle: 'solid',
    numericType: 102,
    category: 'Modules & Imports'
  },
  {
    type: 'node',
    color: colors.very_soft_purple,
    label: 'FunctionType',
    borderStyle: 'solid',
    numericType: 103,
    category: 'Modules & Imports'
  },
  {
    type: 'node',
    color: colors.pink,
    label: 'Import',
    borderStyle: 'solid',
    numericType: 219,
    category: 'Modules & Imports'
  },
  {
    type: 'node',
    color: colors.pink,
    label: 'ImportFrom',
    borderStyle: 'solid',
    numericType: 220,
    category: 'Modules & Imports'
  },
  // Definitions
  {
    type: 'node',
    color: colors.light_green,
    label: 'FunctionDef',
    borderStyle: 'solid',
    numericType: 200,
    category: 'Definitions'
  },
  {
    type: 'node',
    color: colors.light_green,
    label: 'AsyncFunctionDef',
    borderStyle: 'solid',
    numericType: 201,
    category: 'Definitions'
  },
  {
    type: 'node',
    color: colors.light_green,
    label: 'ClassDef',
    borderStyle: 'solid',
    numericType: 202,
    category: 'Definitions'
  },
  {
    type: 'node',
    color: colors.light_green,
    label: 'Lambda',
    borderStyle: 'solid',
    numericType: 304,
    category: 'Definitions'
  },
  // Control Flow
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'Return',
    borderStyle: 'solid',
    numericType: 203,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'For',
    borderStyle: 'solid',
    numericType: 208,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'AsyncFor',
    borderStyle: 'solid',
    numericType: 209,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'While',
    borderStyle: 'solid',
    numericType: 210,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'If',
    borderStyle: 'solid',
    numericType: 211,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'With',
    borderStyle: 'solid',
    numericType: 212,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'AsyncWith',
    borderStyle: 'solid',
    numericType: 213,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'Match',
    borderStyle: 'solid',
    numericType: 214,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'Try',
    borderStyle: 'solid',
    numericType: 216,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'TryStar',
    borderStyle: 'solid',
    numericType: 217,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'Break',
    borderStyle: 'solid',
    numericType: 225,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'Continue',
    borderStyle: 'solid',
    numericType: 226,
    category: 'Control Flow'
  },
  // Statements
  {
    type: 'node',
    color: colors.gray,
    label: 'Delete',
    borderStyle: 'solid',
    numericType: 204,
    category: 'Statements'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'Assign',
    borderStyle: 'solid',
    numericType: 205,
    category: 'Statements'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'AugAssign',
    borderStyle: 'solid',
    numericType: 206,
    category: 'Statements'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'AnnAssign',
    borderStyle: 'solid',
    numericType: 207,
    category: 'Statements'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'Raise',
    borderStyle: 'solid',
    numericType: 215,
    category: 'Statements'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'Assert',
    borderStyle: 'solid',
    numericType: 218,
    category: 'Statements'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'Global',
    borderStyle: 'solid',
    numericType: 221,
    category: 'Statements'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'Nonlocal',
    borderStyle: 'solid',
    numericType: 222,
    category: 'Statements'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'Expr',
    borderStyle: 'solid',
    numericType: 223,
    category: 'Statements'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'Pass',
    borderStyle: 'solid',
    numericType: 224,
    category: 'Statements'
  },
  // Operations
  {
    type: 'node',
    color: colors.pink,
    label: 'BoolOp',
    borderStyle: 'solid',
    numericType: 300,
    category: 'Operations'
  },
  {
    type: 'node',
    color: colors.pink,
    label: 'NamedExpr',
    borderStyle: 'solid',
    numericType: 301,
    category: 'Operations'
  },
  {
    type: 'node',
    color: colors.pink,
    label: 'BinOp',
    borderStyle: 'solid',
    numericType: 302,
    category: 'Operations'
  },
  {
    type: 'node',
    color: colors.pink,
    label: 'UnaryOp',
    borderStyle: 'solid',
    numericType: 303,
    category: 'Operations'
  },
  {
    type: 'node',
    color: colors.pink,
    label: 'IfExp',
    borderStyle: 'solid',
    numericType: 305,
    category: 'Operations'
  },
  {
    type: 'node',
    color: colors.very_soft_lime_green,
    label: 'Call',
    borderStyle: 'solid',
    numericType: 316,
    category: 'Calls & Arguments'
  },
  {
    type: 'node',
    color: colors.pink,
    label: 'Compare',
    borderStyle: 'solid',
    numericType: 315,
    category: 'Operations'
  },
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'Await',
    borderStyle: 'solid',
    numericType: 312,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'Yield',
    borderStyle: 'solid',
    numericType: 313,
    category: 'Control Flow'
  },
  {
    type: 'node',
    color: colors.very_soft_yellow,
    label: 'YieldFrom',
    borderStyle: 'solid',
    numericType: 314,
    category: 'Control Flow'
  },
  // Data Structures
  {
    type: 'node',
    color: colors.peach_puff,
    label: 'Dict',
    borderStyle: 'solid',
    numericType: 306,
    category: 'Data Structures'
  },
  {
    type: 'node',
    color: colors.peach_puff,
    label: 'Set',
    borderStyle: 'solid',
    numericType: 307,
    category: 'Data Structures'
  },
  {
    type: 'node',
    color: colors.peach_puff,
    label: 'List',
    borderStyle: 'solid',
    numericType: 324,
    category: 'Data Structures'
  },
  {
    type: 'node',
    color: colors.peach_puff,
    label: 'Tuple',
    borderStyle: 'solid',
    numericType: 325,
    category: 'Data Structures'
  },
  {
    type: 'node',
    color: colors.pale_green,
    label: 'ListComp',
    borderStyle: 'solid',
    numericType: 308,
    category: 'Comprehensions'
  },
  {
    type: 'node',
    color: colors.pale_green,
    label: 'SetComp',
    borderStyle: 'solid',
    numericType: 309,
    category: 'Comprehensions'
  },
  {
    type: 'node',
    color: colors.pale_green,
    label: 'DictComp',
    borderStyle: 'solid',
    numericType: 310,
    category: 'Comprehensions'
  },
  {
    type: 'node',
    color: colors.pale_green,
    label: 'GeneratorExp',
    borderStyle: 'solid',
    numericType: 311,
    category: 'Comprehensions'
  },
  // Variables & Access
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Constant',
    borderStyle: 'solid',
    numericType: 319,
    category: 'Variables & Access'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'FormattedValue',
    borderStyle: 'solid',
    numericType: 317,
    category: 'Variables & Access'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'JoinedStr',
    borderStyle: 'solid',
    numericType: 318,
    category: 'Variables & Access'
  },
  {
    type: 'node',
    color: colors.light_steel_blue,
    label: 'Attribute',
    borderStyle: 'solid',
    numericType: 320,
    category: 'Variables & Access'
  },
  {
    type: 'node',
    color: colors.light_steel_blue,
    label: 'Subscript',
    borderStyle: 'solid',
    numericType: 321,
    category: 'Variables & Access'
  },
  {
    type: 'node',
    color: colors.light_steel_blue,
    label: 'Starred',
    borderStyle: 'solid',
    numericType: 322,
    category: 'Variables & Access'
  },
  {
    type: 'node',
    color: colors.light_steel_blue,
    label: 'Name',
    borderStyle: 'solid',
    numericType: 323,
    category: 'Variables & Access'
  },
  {
    type: 'node',
    color: colors.light_steel_blue,
    label: 'Slice',
    borderStyle: 'solid',
    numericType: 326,
    category: 'Variables & Access'
  },
  // Operators
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Load',
    borderStyle: 'solid',
    numericType: 400,
    category: 'Contexts'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Store',
    borderStyle: 'solid',
    numericType: 401,
    category: 'Contexts'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Del',
    borderStyle: 'solid',
    numericType: 402,
    category: 'Contexts'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'And',
    borderStyle: 'solid',
    numericType: 410,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Or',
    borderStyle: 'solid',
    numericType: 411,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Add',
    borderStyle: 'solid',
    numericType: 412,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Sub',
    borderStyle: 'solid',
    numericType: 413,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Mult',
    borderStyle: 'solid',
    numericType: 414,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'MatMult',
    borderStyle: 'solid',
    numericType: 415,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Div',
    borderStyle: 'solid',
    numericType: 416,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Mod',
    borderStyle: 'solid',
    numericType: 417,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Pow',
    borderStyle: 'solid',
    numericType: 418,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'LShift',
    borderStyle: 'solid',
    numericType: 419,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'RShift',
    borderStyle: 'solid',
    numericType: 420,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'BitOr',
    borderStyle: 'solid',
    numericType: 421,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'BitXor',
    borderStyle: 'solid',
    numericType: 422,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'BitAnd',
    borderStyle: 'solid',
    numericType: 423,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'FloorDiv',
    borderStyle: 'solid',
    numericType: 424,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Invert',
    borderStyle: 'solid',
    numericType: 430,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Not',
    borderStyle: 'solid',
    numericType: 431,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'UAdd',
    borderStyle: 'solid',
    numericType: 432,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'USub',
    borderStyle: 'solid',
    numericType: 433,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Eq',
    borderStyle: 'solid',
    numericType: 440,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'NotEq',
    borderStyle: 'solid',
    numericType: 441,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Lt',
    borderStyle: 'solid',
    numericType: 442,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'LtE',
    borderStyle: 'solid',
    numericType: 443,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Gt',
    borderStyle: 'solid',
    numericType: 444,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'GtE',
    borderStyle: 'solid',
    numericType: 445,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'Is',
    borderStyle: 'solid',
    numericType: 446,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'IsNot',
    borderStyle: 'solid',
    numericType: 447,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'In',
    borderStyle: 'solid',
    numericType: 448,
    category: 'Operators'
  },
  {
    type: 'node',
    color: colors.powder_blue,
    label: 'NotIn',
    borderStyle: 'solid',
    numericType: 449,
    category: 'Operators'
  },
  // Pattern Matching
  {
    type: 'node',
    color: colors.dark_slate,
    label: 'MatchValue',
    borderStyle: 'solid',
    numericType: 500,
    category: 'Pattern Matching'
  },
  {
    type: 'node',
    color: colors.dark_slate,
    label: 'MatchSingleton',
    borderStyle: 'solid',
    numericType: 501,
    category: 'Pattern Matching'
  },
  {
    type: 'node',
    color: colors.dark_slate,
    label: 'MatchSequence',
    borderStyle: 'solid',
    numericType: 502,
    category: 'Pattern Matching'
  },
  {
    type: 'node',
    color: colors.dark_slate,
    label: 'MatchMapping',
    borderStyle: 'solid',
    numericType: 503,
    category: 'Pattern Matching'
  },
  {
    type: 'node',
    color: colors.dark_slate,
    label: 'MatchClass',
    borderStyle: 'solid',
    numericType: 504,
    category: 'Pattern Matching'
  },
  {
    type: 'node',
    color: colors.dark_slate,
    label: 'MatchStar',
    borderStyle: 'solid',
    numericType: 505,
    category: 'Pattern Matching'
  },
  {
    type: 'node',
    color: colors.dark_slate,
    label: 'MatchAs',
    borderStyle: 'solid',
    numericType: 506,
    category: 'Pattern Matching'
  },
  {
    type: 'node',
    color: colors.dark_slate,
    label: 'MatchOr',
    borderStyle: 'solid',
    numericType: 507,
    category: 'Pattern Matching'
  },
  // Structural & Misc
  {
    type: 'node',
    color: colors.gray,
    label: 'arg',
    borderStyle: 'solid',
    numericType: 600,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'arguments',
    borderStyle: 'solid',
    numericType: 601,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'keyword',
    borderStyle: 'solid',
    numericType: 602,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'alias',
    borderStyle: 'solid',
    numericType: 603,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'withitem',
    borderStyle: 'solid',
    numericType: 604,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'match_case',
    borderStyle: 'solid',
    numericType: 605,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'comprehension',
    borderStyle: 'solid',
    numericType: 606,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'excepthandler',
    borderStyle: 'solid',
    numericType: 607,
    category: 'Structural & Misc'
  },
  {
    type: 'node',
    color: colors.gray,
    label: 'ExceptHandler',
    borderStyle: 'solid',
    numericType: 608,
    category: 'Structural & Misc'
  },
  // Fallback
  {
    type: 'node',
    color: colors.gray,
    label: 'AST_UNKNOWN',
    borderStyle: 'dashed',
    numericType: 999,
    category: 'Fallback'
  },
  // Edges
  {
    type: 'edge',
    color: colors.gray,
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
        color: colors.light_steel_blue,
        label: 'Variable',
        borderStyle: 'solid',
        numericType: 0,
        category: 'State'
      },
      {
        type: 'node',
        color: colors.very_soft_blue,
        label: 'Intermediate',
        borderStyle: 'solid',
        numericType: 4,
        category: 'State'
      },
      {
        type: 'node',
        color: colors.powder_blue,
        label: 'Literal',
        borderStyle: 'solid',
        numericType: 8,
        category: 'State'
      },
      {
        type: 'node',
        color: colors.light_green,
        label: 'Function',
        borderStyle: 'solid',
        numericType: 3,
        category: 'Definitions'
      },
      {
        type: 'node',
        color: colors.very_soft_purple,
        label: 'Class',
        borderStyle: 'solid',
        numericType: 7,
        category: 'Definitions'
      },
      {
        type: 'node',
        color: colors.pink,
        label: 'Import',
        borderStyle: 'solid',
        numericType: 1,
        category: 'Definitions'
      },
      {
        type: 'node',
        color: colors.very_soft_yellow,
        label: 'If',
        borderStyle: 'solid',
        numericType: 5,
        category: 'Control Flow'
      },
      {
        type: 'node',
        color: colors.very_soft_lime_green,
        label: 'Loop',
        borderStyle: 'solid',
        numericType: 6,
        category: 'Control Flow'
      },
      {
        type: 'edge',
        color: colors.light_salmon,
        label: 'Caller',
        borderStyle: 'solid',
        numericType: 0,
        category: 'Edges'
      },
      {
        type: 'edge',
        color: colors.pale_green,
        label: 'Input',
        borderStyle: 'solid',
        numericType: 1,
        category: 'Edges'
      },
      {
        type: 'edge',
        color: colors.gray,
        label: 'Reassign',
        borderStyle: 'dashed',
        numericType: 2,
        category: 'Edges'
      },
      {
        type: 'edge',
        color: colors.powder_blue,
        label: 'Branch',
        borderStyle: 'solid',
        numericType: 3,
        category: 'Edges'
      },
      {
        type: 'edge',
        color: colors.peach_puff,
        label: 'Loop',
        borderStyle: 'solid',
        numericType: 4,
        category: 'Edges'
      },
      {
        type: 'edge',
        color: colors.light_green,
        label: 'Function',
        borderStyle: 'solid',
        numericType: 5,
        category: 'Edges'
      }
    ],
    getNodeColor: type => {
      const map: Record<number, string> = {
        0: colors.light_steel_blue,
        1: colors.pink,
        3: colors.light_green,
        4: colors.very_soft_blue,
        5: colors.very_soft_yellow,
        6: colors.very_soft_lime_green,
        7: colors.very_soft_purple,
        8: colors.powder_blue
      };
      return map[type] || '#000';
    },
    getEdgeColor: type => {
      const map: Record<number, string> = {
        0: colors.light_salmon,
        1: colors.pale_green,
        2: colors.gray,
        3: colors.powder_blue,
        4: colors.peach_puff,
        5: colors.light_green
      };
      return map[type] || '#000';
    }
  },

  lineage: sharedLineageConfig,
  vamsa: sharedLineageConfig,
  labeling: sharedLineageConfig,

  ast: {
    legends: EXHAUSTIVE_AST_LEGENDS,
    getNodeColor: type => AST_NODE_COLOR_MAP[type] || colors.powder_blue,
    getEdgeColor: type => colors.gray
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
