import React, { useCallback, useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';

// Ensure the layout is registered only once per module load
if (!cytoscape.prototype.dagre) {
  cytoscape.use(dagre);
}

const colors = {
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
  peach_puff: '#FFDAB9'
};

interface LegendItemType {
  type: 'node' | 'edge';
  color: string;
  label: string;
  borderStyle: 'solid' | 'dashed';
  numericType: number;
}

const initialLegendItems: LegendItemType[] = [
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
];

const initialLegendItemsLineage: LegendItemType[] = [
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
];

interface GraphProps {
  graphData: { elements: any[] };
  mode: string;
  resetTrigger: number;
}

const filterLegendItems = (
  elements: any[],
  allLegendItems: LegendItemType[]
): LegendItemType[] => {
  if (!elements || elements.length === 0) return [];

  const presentNodeTypes = new Set<number>();
  const presentEdgeTypes = new Set<number>();

  elements.forEach(element => {
    if (typeof element.data.node_type === 'number') {
      presentNodeTypes.add(element.data.node_type);
    } else if (typeof element.data.predicted_label === 'number') {
      presentEdgeTypes.add(element.data.predicted_label);
    } else if (typeof element.data.edge_type === 'number') {
      presentEdgeTypes.add(element.data.edge_type);
    }
  });

  return allLegendItems.filter(item =>
    item.type === 'node'
      ? presentNodeTypes.has(item.numericType)
      : presentEdgeTypes.has(item.numericType)
  );
};

export default function Graph({ graphData, mode, resetTrigger }: GraphProps) {
  const layout = { name: 'dagre', rankDir: 'TB', animate: false, fit: false };
  const graphRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  const [activeLegendItems, setActiveLegendItems] = useState<LegendItemType[]>(
    []
  );

  // Bypassing brittle Cytoscape type definitions with `any` for structural stability
  const edgeType = useCallback(
    (element: any) => {
      const caseType =
        mode === 'dataflow'
          ? element.data('edge_type')
          : element.data('predicted_label');
      switch (caseType) {
        case 0:
          return colors.light_salmon;
        case 1:
          return colors.pale_green;
        case 2:
          return colors.gray;
        case 3:
          return colors.powder_blue;
        case 4:
          return colors.peach_puff;
        case 5:
          return colors.light_green;
        default:
          return '#000';
      }
    },
    [mode]
  );

  const nodeType = useCallback(
    (element: any) => {
      const caseType = element.data('node_type');
      if (mode === 'lineage') {
        switch (caseType) {
          case 0:
            return colors.light_steel_blue;
          case 1:
            return colors.pink;
          case 2:
            return colors.light_green;
          case 3:
            return colors.very_soft_blue;
          default:
            return colors.light_steel_blue;
        }
      }
      switch (caseType) {
        case 0:
          return colors.light_steel_blue;
        case 1:
          return colors.pink;
        case 3:
          return colors.light_green;
        case 4:
          return colors.very_soft_blue;
        case 5:
          return colors.very_soft_yellow;
        case 6:
          return colors.very_soft_lime_green;
        case 7:
          return colors.very_soft_purple;
        case 8:
          return colors.powder_blue;
        default:
          return '#000';
      }
    },
    [mode]
  );

  const lineType = useCallback((element: any) => {
    const caseType = element.data('edge_type');
    return caseType === 2 ? 'dashed' : 'solid';
  }, []);

  useEffect(() => {
    if (!graphRef.current) return;

    cyRef.current = cytoscape({
      container: graphRef.current,
      elements: [],
      layout: layout,
      panningEnabled: true,
      zoomingEnabled: true,
      wheelSensitivity: 0.2
    });

    return () => {
      cyRef.current?.destroy();
      cyRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!cyRef.current) return;

    cyRef.current.style([
      {
        selector: 'node',
        style: {
          shape: 'round-rectangle',
          'background-color': (ele: any) => nodeType(ele),
          label: 'data(label)',
          width: '60px',
          height: '35px',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '12px',
          color: '#333',
          'text-wrap': 'ellipsis',
          'text-max-width': '80px'
        }
      },
      {
        selector: 'edge',
        style: {
          width: 2,
          'line-color': (ele: any) => edgeType(ele),
          'target-arrow-shape': 'triangle',
          'target-arrow-color': (ele: any) => edgeType(ele),
          'curve-style': 'bezier',
          label: 'data(label)',
          'line-style': (ele: any) => lineType(ele) as 'solid' | 'dashed'
        }
      }
    ] as cytoscape.StylesheetStyle[]);

    // CRITICAL FIX: Explicit bulk removal and addition correctly resolves order-dependent edge references
    cyRef.current.elements().remove();
    cyRef.current.add(graphData.elements);

    const prevTrigger = cyRef.current.data('prevResetTrigger');
    if (prevTrigger !== undefined && resetTrigger !== prevTrigger) {
      cyRef.current.fit(undefined, 50);
    }
    cyRef.current.data('prevResetTrigger', resetTrigger);

    setActiveLegendItems(
      filterLegendItems(
        graphData.elements,
        mode === 'dataflow' ? initialLegendItems : initialLegendItemsLineage
      )
    );

    const layoutInstance = cyRef.current.layout(layout);
    layoutInstance.on('layoutstop', () => {
      if (cyRef.current) {
        cyRef.current.fit(undefined, 50);
      }
    });
    layoutInstance.run();
  }, [graphData, mode, resetTrigger, nodeType, edgeType, lineType]);

  return (
    <div className="flex flex-col h-full bg-white overflow-hidden">
      <div
        id="cy"
        className="w-full grow min-h-0 relative border-b border-gray-200 bg-[#fafafa]"
        ref={graphRef}
      />

      <ul className="flex flex-row justify-center items-center bg-white text-gray-700 list-none p-4 m-0 flex-wrap shrink-0 shadow-inner z-10">
        {activeLegendItems.map((item, index) => (
          <li
            key={index}
            className="mr-6 flex flex-col justify-center items-center last:mr-0 min-w-[60px]"
          >
            <div
              className={
                item.type === 'node'
                  ? 'w-[50px] h-[25px] rounded m-1 inline-block border border-gray-300/50'
                  : 'w-full h-0 inline-block border-t-[3px] bg-transparent mb-2 mt-2'
              }
              style={{
                backgroundColor:
                  item.type === 'node' ? item.color : 'transparent',
                borderColor: item.color,
                borderStyle: item.borderStyle
              }}
            />
            <span className="text-xs font-medium tracking-wide">
              {item.label}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
