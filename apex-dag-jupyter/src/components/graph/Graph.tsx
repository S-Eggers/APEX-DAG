import React, { useRef, useState } from 'react';
import { GraphProps, LegendItemType } from '../../types/GraphTypes';
import {
  groupLegendItems,
  filterLegendItems,
  MODE_CONFIG
} from '../../config/GraphConfig';
import { callBackend } from '../../utils/callBackend';
import { useCytoscape } from '../../hooks/useCytoscape';

import NodeDetailsPanel from './panels/NodeDetailsPanel';
import EdgeDetailsPanel from './panels/EdgeDetailsPanel';
import GraphLegend from './panels/GraphLegend';

type ActionState = 'idle' | 'loading' | 'success' | 'error';

export default function Graph({
  graphData,
  mode,
  resetTrigger,
  taxonomy,
  notebookName,
  notebookCode,
  rawDatasetPath,
  onLocateCell,
  onNextNotebook
}: GraphProps) {
  const graphRef = useRef<HTMLDivElement>(null);

  const [selectedNode, setSelectedNode] = useState<any | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<any | null>(null);
  const [saveState, setSaveState] = useState<ActionState>('idle');
  const [nextState, setNextState] = useState<ActionState>('idle');

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
    MODE_CONFIG[mode].getNodeColor,
    MODE_CONFIG[mode].getEdgeColor
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
    setSaveState('loading');

    const currentGraphJson = (cyRef.current.json() as any).elements;
    const rawPythonCode = notebookCode.map((c: any) => c.source).join('\n');

    try {
      const result = await callBackend('labeling/save', {
        filename: notebookName,
        graph: currentGraphJson,
        code: rawPythonCode
      });

      if (result.success) {
        setSaveState('success');
      } else {
        console.error('Save failed: ', result.message);
        setSaveState('error');
      }
    } catch (e) {
      console.error('Network or Auth error during save: ', e);
      setSaveState('error');
    } finally {
      setTimeout(() => setSaveState('idle'), 2000);
    }
  };

  const handleNextNotebook = async () => {
    setNextState('loading');
    try {
      const result = await callBackend('labeling/next', {
        datasetPath: rawDatasetPath || 'raw_dataset'
      });

      if (result.success && result.path) {
        setNextState('success');
        if (onNextNotebook) {
          onNextNotebook(result.path);
        }
      } else {
        console.error(result.message || 'No more notebooks available.');
        setNextState('error');
      }
    } catch (e) {
      console.error('Failed to fetch next notebook:', e);
      setNextState('error');
    } finally {
      setTimeout(() => setNextState('idle'), 2000);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white overflow-hidden relative">
      {mode === 'labeling' && (
        <div className="absolute top-4 left-4 z-30 flex items-center gap-3 bg-white p-2 rounded shadow-md border border-gray-200">
          <span className="text-xs text-gray-500 font-mono">
            {notebookName}.json
          </span>

          <a
            onClick={handleSaveAnnotations}
            className={`px-4 py-2 rounded shadow font-medium block cursor-pointer transition-colors duration-200 !text-white ${
              saveState === 'success'
                ? 'bg-green-600'
                : saveState === 'error'
                  ? 'bg-red-600'
                  : saveState === 'loading'
                    ? 'bg-blue-400 !cursor-wait'
                    : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {saveState === 'idle' && 'Save JSON'}
            {saveState === 'loading' && 'Saving...'}
            {saveState === 'success' && 'Saved ✓'}
            {saveState === 'error' && 'Error ✗'}
          </a>

          <a
            onClick={handleNextNotebook}
            className={`px-4 py-2 rounded shadow cursor-pointer block font-medium !text-white transition-colors duration-200 ${
              nextState === 'success'
                ? 'bg-green-600'
                : nextState === 'error'
                  ? 'bg-red-600'
                  : nextState === 'loading'
                    ? 'bg-gray-600 !cursor-wait'
                    : 'bg-gray-800 hover:bg-gray-900'
            }`}
          >
            {nextState === 'idle' && 'Next'}
            {nextState === 'loading' && 'Loading...'}
            {nextState === 'success' && 'Opening ✓'}
            {nextState === 'error' && 'Empty ✗'}
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
