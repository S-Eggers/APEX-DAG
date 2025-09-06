import React, { useCallback, useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';

cytoscape.use(dagre);

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
  numericType: number; // Corresponds to node_type or edge_type
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

interface ElementData {
  id: string;
  label?: string;
  node_type?: number;
  edge_type?: number;
  source?: string;
  target?: string;
  predicted_label?: number;
}

interface CytoscapeElement {
  data: ElementData;
  group: 'nodes' | 'edges';
}

interface GraphData {
  elements: CytoscapeElement[];
}

interface GraphProps {
  graphData?: GraphData;
  mode?: string;
  eventTarget: EventTarget;
}

const filterLegendItems = (
  elements: CytoscapeElement[],
  allLegendItems: LegendItemType[]
): LegendItemType[] => {
  if (!elements || elements.length === 0) {
    return []; // No elements in graph, so no legend items to show
  }

  const presentNodeTypes = new Set<number>();
  const presentEdgeTypes = new Set<number>();

  elements.forEach(element => {
    if (
      element.data.node_type !== undefined &&
      typeof element.data.node_type === 'number'
    ) {
      presentNodeTypes.add(element.data.node_type);
    } else if (
      element.data.predicted_label !== undefined &&
      typeof element.data.predicted_label === 'number'
    ) {
      presentEdgeTypes.add(element.data.predicted_label);
    } else if (
      element.data.edge_type !== undefined &&
      typeof element.data.edge_type === 'number'
    ) {
      presentEdgeTypes.add(element.data.edge_type);
    }
  });

  return allLegendItems.filter(item => {
    if (item.type === 'node') {
      return presentNodeTypes.has(item.numericType);
    } else if (item.type === 'edge') {
      return presentEdgeTypes.has(item.numericType);
    }
    return false;
  });
};

export default function Graph({
  graphData = { elements: [] },
  mode = 'dataflow',
  eventTarget
}: GraphProps) {
  const layout = {
    name: 'dagre',
    rankDir: 'TB'
  };

  const edgeType = (element: cytoscape.SingularElementReturnValue) => {
    const caseType =
      mode == 'dataflow'
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
  };

  const nodeType = (element: cytoscape.SingularElementReturnValue) => {
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
      /*case 2: return colors.light_green; <- not used yet*/
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
      default:
        return '#000';
    }
  };

  const lineType = (element: cytoscape.SingularElementReturnValue) => {
    const caseType = element.data('edge_type');
    switch (caseType) {
      case 2:
        return 'dashed';
      default:
        return 'solid';
    }
  };

  const style = [
    {
      selector: 'node',
      style: {
        shape: 'round-rectangle',
        'background-color': (element: cytoscape.SingularElementReturnValue) =>
          nodeType(element),
        label: 'data(label)',
        width: '60px',
        height: '35px',
        'text-valign': 'center' as 'center' | 'top' | 'bottom',
        'text-halign': 'center' as 'center' | 'left' | 'right',
        'font-size': '12px',
        color: '#333'
      }
    },
    {
      selector: 'edge',
      style: {
        width: 2,
        'line-color': (element: cytoscape.SingularElementReturnValue) =>
          edgeType(element),
        'target-arrow-shape': 'triangle',
        'target-arrow-color': (element: cytoscape.SingularElementReturnValue) =>
          edgeType(element),
        'curve-style': 'bezier',
        label: 'data(label)',
        'line-style': (element: cytoscape.SingularElementReturnValue) =>
          lineType(element) as 'solid' | 'dashed'
      }
    }
  ];

  const graphRef = useRef(null);
  const [pan, setPan] = useState<cytoscape.Position | null>(null);
  const [zoom, setZoom] = useState(1);
  const [activeLegendItems, setActiveLegendItems] =
    useState<LegendItemType[]>(initialLegendItems);

  const cyRef = useRef<cytoscape.Core | null>(null);

  const drawGraph = () => {
    const cy = cytoscape({
      container: graphRef.current,
      style: style,
      layout: layout,
      elements: graphData.elements
    });

    cyRef.current = cy; // Store the cy instance

    if (pan) {
      cy.pan(pan);
    } else {
      console.log('Centering the graph');
      cy.center();
    }
    cy.zoom(zoom);
    cy.on('pan', () => {
      setPan(cy.pan());
    });

    cy.on('zoom', () => {
      setZoom(cy.zoom());
    });

    setPan(cy.pan());
    setZoom(cy.zoom());
  };

  useEffect(() => {
    drawGraph();
  }, [graphData]);

  useEffect(() => {
    const filtered = filterLegendItems(
      graphData.elements,
      mode === 'dataflow' ? initialLegendItems : initialLegendItemsLineage
    );
    console.log('Filtered legend items:', filtered);
    setActiveLegendItems(filtered);
  }, [graphData]);

  const handleResetView = useCallback(() => {
    if (cyRef.current) {
      console.log('Resetting Cytoscape view (fit and zoom 1)');
      cyRef.current.fit(); // Fit the graph to the viewport
      cyRef.current.zoom(1); // Reset zoom to 1
      setPan(cyRef.current.pan()); // Update React state
      setZoom(cyRef.current.zoom()); // Update React state
    }
  }, [setPan, setZoom]); // Add setPan and setZoom to dependencies for useCallback

  useEffect(() => {
    eventTarget.addEventListener('reset-view', handleResetView);

    return () => {
      eventTarget.removeEventListener('reset-view', handleResetView);
    };
  }, [eventTarget, handleResetView]); // Add handleResetView to dependencies for useEffect

  return (
    <>
      <div id="cy" className={'cy'} ref={graphRef}></div>
      <ul className={'legend'}>
        {activeLegendItems.map((item, index) => (
          <li key={index}>
            <div
              className={item.type}
              style={{
                backgroundColor: item.color,
                borderColor: item.color,
                borderStyle: item.borderStyle as 'solid' | 'dashed'
              }}
            ></div>{' '}
            {item.label}
          </li>
        ))}
      </ul>
    </>
  );
}
