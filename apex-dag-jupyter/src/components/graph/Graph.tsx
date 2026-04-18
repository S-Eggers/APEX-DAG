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
  const [selectedNode, setSelectedNode] = useState<any | null>(null);

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

    cyRef.current.on('tap', 'node', evt => {
      const nodeData = evt.target.data();
      setSelectedNode(nodeData);

      cyRef.current?.elements().removeClass('selected');
      evt.target.addClass('selected');
    });

    cyRef.current.on('tap', evt => {
      if (evt.target === cyRef.current) {
        setSelectedNode(null);
        cyRef.current?.elements().removeClass('selected');
      }
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
          'text-max-width': '80px',
          'border-width': 1,
          'border-color': '#ccc'
        }
      },
      {
        selector: 'node.selected',
        style: {
          'border-width': 3,
          'border-color': '#2563eb',
          'shadow-blur': 10,
          'shadow-color': '#2563eb',
          'shadow-opacity': 0.5
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

    setSelectedNode(null);

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
    <div className="flex flex-col h-full bg-white overflow-hidden relative">
      <div
        id="cy"
        className="w-full grow min-h-0 relative bg-[#fafafa]"
        ref={graphRef}
      />

      {selectedNode && (
        <div className="absolute top-0 right-0 w-80 h-full bg-white border-l border-gray-200 shadow-xl flex flex-col z-20 transform transition-transform duration-300">
          <div className="p-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
            <h3
              className="font-bold text-gray-800 truncate"
              title={selectedNode.label}
            >
              {selectedNode.label || 'Node Details'}
            </h3>

            <button
              onClick={() => {
                setSelectedNode(null);
                cyRef.current?.elements().removeClass('selected');
              }}
              className="text-gray-400 hover:text-gray-800 hover:bg-gray-200 p-1 rounded transition-colors focus:outline-none"
              aria-label="Close panel"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          <div className="p-4 overflow-y-auto grow">
            <div className="mb-6">
              <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">
                Type ID
              </span>
              <p className="text-sm font-mono text-gray-800 mt-1">
                {selectedNode.node_type}
              </p>
            </div>
            {selectedNode.base_inputs && (
              <div className="mb-6">
                <span className="text-xs font-bold text-gray-400 uppercase tracking-wider block mb-2">
                  Parameters / Imports
                </span>
                <div className="bg-green-50 p-2 rounded text-xs text-green-800 border border-green-200 font-mono overflow-x-auto whitespace-pre-wrap">
                  {selectedNode.base_inputs}
                </div>
              </div>
            )}
            <div className="mb-6">
              <span className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2 block">
                Transformation History
              </span>
              {selectedNode.transform_history &&
              selectedNode.transform_history.length > 0 ? (
                <div className="space-y-3">
                  {selectedNode.transform_history.map(
                    (step: any, idx: number) => (
                      <div
                        key={idx}
                        className="bg-blue-50 p-3 rounded border border-blue-100 flex flex-col gap-2"
                      >
                        <div className="flex justify-between items-center border-b border-blue-200 pb-1">
                          <span className="text-xs font-bold text-blue-800 uppercase truncate pr-2">
                            {step.operation}
                          </span>
                          <span className="text-xs text-blue-600 font-mono whitespace-nowrap">
                            → {step.target_node}
                          </span>
                        </div>

                        {step.transform_code && (
                          <div className="text-xs text-gray-700 font-mono bg-white p-2 rounded border border-gray-200 overflow-x-auto whitespace-pre-wrap">
                            {step.transform_code}
                          </div>
                        )}
                      </div>
                    )
                  )}
                </div>
              ) : (
                <div className="bg-gray-50 p-3 rounded text-xs border border-gray-200 border-dashed text-gray-500 italic">
                  No linear transformations were contracted into this node.
                </div>
              )}
            </div>

            <div className="mb-6">
              <span className="text-xs font-bold text-gray-400 uppercase tracking-wider block mb-2">
                Original Assignment Code
              </span>
              {selectedNode.code ? (
                <pre className="bg-gray-100 p-2 rounded text-xs text-gray-700 overflow-x-auto whitespace-pre-wrap border border-gray-200">
                  {selectedNode.code}
                </pre>
              ) : (
                <div className="bg-gray-50 p-3 rounded text-xs border border-gray-200 border-dashed text-gray-500 italic">
                  Source code unavailable for this node.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <ul className="flex flex-row justify-center items-center bg-white text-gray-700 list-none p-4 m-0 flex-wrap shrink-0 shadow-inner z-10 relative">
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
