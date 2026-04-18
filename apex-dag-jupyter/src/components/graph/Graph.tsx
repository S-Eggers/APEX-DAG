import React, { useCallback, useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import { GraphProps, LegendItemType } from '../../types/GraphTypes';
import { MODE_CONFIG, filterLegendItems } from '../../config/GraphConfig';

if (!cytoscape.prototype.dagre) {
  cytoscape.use(dagre);
}

export default function Graph({ graphData, mode, resetTrigger }: GraphProps) {
  const layout = { name: 'dagre', rankDir: 'TB', animate: false, fit: false };
  const graphRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  const [activeLegendItems, setActiveLegendItems] = useState<LegendItemType[]>(
    []
  );

  const edgeTypeColor = useCallback(
    (element: any) => {
      const caseType =
        mode === 'dataflow'
          ? element.data('edge_type')
          : element.data('predicted_label');
      return MODE_CONFIG[mode].getEdgeColor(caseType);
    },
    [mode]
  );

  const nodeTypeColor = useCallback(
    (element: any) => {
      return MODE_CONFIG[mode].getNodeColor(element.data('node_type'));
    },
    [mode]
  );

  const lineTypeStyle = useCallback((element: any) => {
    return element.data('edge_type') === 2 ? 'dashed' : 'solid';
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
          'background-color': (ele: any) => nodeTypeColor(ele),
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
          'line-color': (ele: any) => edgeTypeColor(ele),
          'target-arrow-shape': 'triangle',
          'target-arrow-color': (ele: any) => edgeTypeColor(ele),
          'curve-style': 'bezier',
          label: 'data(label)',
          'line-style': (ele: any) => lineTypeStyle(ele) as 'solid' | 'dashed'
        }
      }
    ] as cytoscape.StylesheetStyle[]);

    cyRef.current.elements().remove();
    cyRef.current.add(graphData.elements);

    const prevTrigger = cyRef.current.data('prevResetTrigger');
    if (prevTrigger !== undefined && resetTrigger !== prevTrigger) {
      cyRef.current.fit(undefined, 50);
    }
    cyRef.current.data('prevResetTrigger', resetTrigger);

    setActiveLegendItems(
      filterLegendItems(graphData.elements, MODE_CONFIG[mode].legends)
    );

    const layoutInstance = cyRef.current.layout(layout);
    layoutInstance.on('layoutstop', () => {
      if (cyRef.current) {
        cyRef.current.fit(undefined, 50);
      }
    });
    layoutInstance.run();
  }, [
    graphData,
    mode,
    resetTrigger,
    nodeTypeColor,
    edgeTypeColor,
    lineTypeStyle
  ]);

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
