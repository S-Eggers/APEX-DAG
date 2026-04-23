import React, { useRef, useState } from 'react';
import { GraphProps, LegendItemType } from '../../types/GraphTypes';
import {
  groupLegendItems,
  filterLegendItems,
  MODE_CONFIG
} from '../../config/GraphConfig';
import callBackend from '../../utils/callBackend';
import { useCytoscape } from '../../hooks/useCytoscape';

import NodeDetailsPanel from './panels/NodeDetailsPanel';
import EdgeDetailsPanel from './panels/EdgeDetailsPanel';
import GraphLegend from './panels/GraphLegend';

export default function Graph({
  graphData,
  mode,
  resetTrigger,
  taxonomy,
  notebookName,
  notebookCode,
  onLocateCell
}: GraphProps) {
  const graphRef = useRef<HTMLDivElement>(null);

  const [selectedNode, setSelectedNode] = useState<any | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<any | null>(null);

  const [groupedLegendItems, setGroupedLegendItems] = useState<
    Record<string, LegendItemType[]>
  >({});

  const updateLegend = () => {
    if (!cyRef.current) return;
    const liveElements = cyRef.current
      .elements()
      .map(ele => ({ data: ele.data() }));
    const authoritativeLegends = MODE_CONFIG[mode].legends;

    const filtered = filterLegendItems(liveElements, authoritativeLegends);
    setGroupedLegendItems(groupLegendItems(filtered));
  };

  const cyRef = useCytoscape(
    graphRef,
    graphData,
    mode,
    resetTrigger,
    nodeData => {
      setSelectedNode(nodeData);
      if (nodeData && nodeData.cell_id && onLocateCell) {
        onLocateCell(nodeData.cell_id);
      }
    },
    edgeData => {
      if (!edgeData || typeof edgeData.data !== 'function') {
        setSelectedEdge(null);
        return;
      }
      const data = edgeData.data();
      setSelectedEdge(edgeData);
      if (data.cell_id && onLocateCell) {
        onLocateCell(data.cell_id);
      }
    },
    () => {
      updateLegend();
    },
    taxonomy.getNodeColor,
    taxonomy.getEdgeColor
  );

  const handleEdgeLabelChange = (newLabelValue: number) => {
    if (!selectedEdge || !cyRef.current) return;

    selectedEdge.data('predicted_label', newLabelValue);
    const labelStr = taxonomy.edgeLabelOptions.find(
      (opt: any) => opt.value === newLabelValue
    )?.label;
    if (labelStr) selectedEdge.data('domain_label', labelStr);

    setSelectedEdge(cyRef.current.getElementById(selectedEdge.id()));
    updateLegend();
  };

  const handleNodeLabelChange = (newLabelValue: number) => {
    if (!selectedNode || !cyRef.current) return;

    const cyNode = cyRef.current.getElementById(selectedNode.id);
    if (!cyNode) return;

    cyNode.data('node_type', newLabelValue);
    const labelStr = taxonomy.nodeLabelOptions.find(
      (opt: any) => opt.value === newLabelValue
    )?.label;
    if (labelStr) cyNode.data('domain_label', labelStr);

    setSelectedNode({ ...cyNode.data() });
    updateLegend();
  };

  const handleSaveAnnotations = async () => {
    if (!cyRef.current) return;
    const currentGraphJson = (cyRef.current.json() as any).elements;
    const rawPythonCode = notebookCode.map((c: any) => c.source).join('\n');

    try {
      const result = await callBackend('labeling/save', {
        filename: notebookName,
        graph: currentGraphJson,
        code: rawPythonCode
      });

      if (result.success) {
        alert('Annotations saved securely.');
      } else {
        console.error('Save failed: ', result.message);
      }
    } catch (e) {
      console.error('Network or Auth error during save: ', e);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white overflow-hidden relative">
      {mode === 'labeling' && (
        <div className="absolute top-4 left-4 z-30 flex items-center gap-3 bg-white p-2 rounded shadow-md border border-gray-200">
          <span className="text-xs text-gray-500 font-mono">
            {notebookName}_annotated.gml
          </span>
          <a
            onClick={handleSaveAnnotations}
            className="bg-blue-600 !text-white px-4 py-2 rounded shadow hover:bg-blue-700 transition hover:cursor-pointer"
          >
            Save GML
          </a>
        </div>
      )}

      <div
        id="cy"
        className="w-full grow min-h-0 relative bg-[#fafafa]"
        ref={graphRef}
      />

      {selectedNode && (
        <NodeDetailsPanel
          node={selectedNode}
          mode={mode}
          options={taxonomy.nodeLabelOptions}
          onChange={handleNodeLabelChange}
          onClose={() => {
            setSelectedNode(null);
            cyRef.current?.elements().removeClass('selected');
          }}
        />
      )}

      {selectedEdge && (
        <EdgeDetailsPanel
          edge={selectedEdge}
          mode={mode}
          options={taxonomy.edgeLabelOptions}
          onChange={handleEdgeLabelChange}
          onClose={() => {
            setSelectedEdge(null);
            cyRef.current?.elements().removeClass('selected');
          }}
        />
      )}

      <GraphLegend groupedItems={groupedLegendItems} />
    </div>
  );
}
