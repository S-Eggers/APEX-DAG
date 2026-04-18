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
      numericType: 0
    },
    {
      type: 'node',
      color: colors.pink,
      label: 'Library',
      borderStyle: 'solid',
      numericType: 1
    },
    {
      type: 'node',
      color: colors.light_green,
      label: 'Dataset',
      borderStyle: 'solid',
      numericType: 2
    },
    {
      type: 'node',
      color: colors.very_soft_blue,
      label: 'UDF',
      borderStyle: 'solid',
      numericType: 3
    },
    {
      type: 'edge',
      color: colors.light_salmon,
      label: 'Model Train/Eval',
      borderStyle: 'solid',
      numericType: 0
    },
    {
      type: 'edge',
      color: colors.peach_puff,
      label: 'Environment+Data Export',
      borderStyle: 'solid',
      numericType: 4
    },
    {
      type: 'edge',
      color: colors.powder_blue,
      label: 'EDA',
      borderStyle: 'solid',
      numericType: 3
    },
    {
      type: 'edge',
      color: colors.pale_green,
      label: 'Data Import',
      borderStyle: 'solid',
      numericType: 1
    },
    {
      type: 'edge',
      color: colors.gray,
      label: 'Data Transform',
      borderStyle: 'solid',
      numericType: 2
    }
  ] as LegendItemType[],
  getNodeColor: (type: number) => {
    const map: Record<number, string> = {
      0: colors.light_steel_blue,
      1: colors.pink,
      2: colors.light_green,
      3: colors.very_soft_blue
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
        numericType: 0
      },
      {
        type: 'node',
        color: colors.very_soft_blue,
        label: 'Intermediate',
        borderStyle: 'solid',
        numericType: 4
      },
      {
        type: 'node',
        color: colors.light_green,
        label: 'Function',
        borderStyle: 'solid',
        numericType: 3
      },
      {
        type: 'node',
        color: colors.pink,
        label: 'Import',
        borderStyle: 'solid',
        numericType: 1
      },
      {
        type: 'node',
        color: colors.very_soft_yellow,
        label: 'If',
        borderStyle: 'solid',
        numericType: 5
      },
      {
        type: 'node',
        color: colors.very_soft_lime_green,
        label: 'Loop',
        borderStyle: 'solid',
        numericType: 6
      },
      {
        type: 'node',
        color: colors.very_soft_purple,
        label: 'Class',
        borderStyle: 'solid',
        numericType: 7
      },
      {
        type: 'node',
        color: colors.powder_blue,
        label: 'Literal',
        borderStyle: 'solid',
        numericType: 8
      },
      {
        type: 'edge',
        color: colors.light_salmon,
        label: 'Caller',
        borderStyle: 'solid',
        numericType: 0
      },
      {
        type: 'edge',
        color: colors.gray,
        label: 'Reassign',
        borderStyle: 'dashed',
        numericType: 2
      },
      {
        type: 'edge',
        color: colors.pale_green,
        label: 'Input',
        borderStyle: 'solid',
        numericType: 1
      },
      {
        type: 'edge',
        color: colors.powder_blue,
        label: 'Branch',
        borderStyle: 'solid',
        numericType: 3
      },
      {
        type: 'edge',
        color: colors.peach_puff,
        label: 'Loop',
        borderStyle: 'solid',
        numericType: 4
      },
      {
        type: 'edge',
        color: colors.light_green,
        label: 'Function',
        borderStyle: 'solid',
        numericType: 5
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

  ast: {
    legends: [
      {
        type: 'node',
        color: colors.powder_blue,
        label: 'AST Node',
        borderStyle: 'solid',
        numericType: 0
      },
      {
        type: 'edge',
        color: colors.gray,
        label: 'Parent/Child',
        borderStyle: 'solid',
        numericType: 0
      }
    ],
    getNodeColor: type => colors.powder_blue,
    getEdgeColor: type => colors.gray
  }
};

export const filterLegendItems = (
  elements: any[],
  allLegendItems: LegendItemType[]
): LegendItemType[] => {
  if (!elements || elements.length === 0) return [];

  const presentNodeTypes = new Set<number>();
  const presentEdgeTypes = new Set<number>();

  elements.forEach(element => {
    if (typeof element.data.node_type === 'number')
      presentNodeTypes.add(element.data.node_type);
    else if (typeof element.data.predicted_label === 'number')
      presentEdgeTypes.add(element.data.predicted_label);
    else if (typeof element.data.edge_type === 'number')
      presentEdgeTypes.add(element.data.edge_type);
  });

  return allLegendItems.filter(item =>
    item.type === 'node'
      ? presentNodeTypes.has(item.numericType)
      : presentEdgeTypes.has(item.numericType)
  );
};
