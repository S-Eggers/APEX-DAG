import React, { useRef, useState } from 'react';
import {
  GraphProps,
  LegendItemType,
  GraphNodeData,
  CyElement,
  GraphEdgeData,
  GraphElementPayload
} from '../../types/GraphTypes';
import {
  groupLegendItems,
  filterLegendItems
} from '../../hooks/useGraphTaxonomy';
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

  const [selectedNode, setSelectedNode] = useState<GraphNodeData | null>(null);
  const [selectedEdge, setSelectedEdge] =
    useState<CyElement<GraphEdgeData> | null>(null);

  const [saveState, setSaveState] = useState<ActionState>('idle');
  const [nextState, setNextState] = useState<ActionState>('idle');
  const [flagState, setFlagState] = useState<ActionState>('idle');
  const [showFlagMenu, setShowFlagMenu] = useState(false);

  const [groupedLegendItems, setGroupedLegendItems] = useState<
    Record<string, LegendItemType[]>
  >({});

  const updateLegend = () => {
    if (!cyRef.current || !taxonomy.isLoaded) return;

    const liveElements: GraphElementPayload[] = cyRef.current
      .elements()
      .map((ele: CyElement<GraphNodeData & GraphEdgeData>) => ({
        data: ele.data() as GraphElementPayload['data']
      }));

    const authoritativeLegends = taxonomy.legends;
    const filtered = filterLegendItems(liveElements, authoritativeLegends);
    setGroupedLegendItems(groupLegendItems(filtered));
  };
  const cyRef = useCytoscape(
    graphRef,
    graphData,
    mode,
    resetTrigger,
    (nodeData: GraphNodeData) => {
      setSelectedNode(nodeData);
      if (nodeData && nodeData.cell_id && onLocateCell) {
        onLocateCell(nodeData.cell_id);
      }
    },
    (edgeData: CyElement<GraphEdgeData> | null) => {
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
      opt => opt.value === newLabelValue
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
      opt => opt.value === newLabelValue
    )?.label;

    if (labelStr) cyNode.data('domain_label', labelStr);

    setSelectedNode({ ...(cyNode.data() as GraphNodeData) });
    updateLegend();
  };

  const handleSaveAnnotations = async () => {
    if (!cyRef.current) return;
    setSaveState('loading');

    const currentGraphJson = (cyRef.current.json() as { elements: unknown })
      .elements;
    const rawPythonCode = notebookCode
      .map((c: { source: string }) => c.source)
      .join('\n');

    try {
      const result = await callBackend('labeling/save', {
        filename: notebookName,
        graph: currentGraphJson,
        code: rawPythonCode
      });

      if (result.success) {
        setSaveState('success');
      } else {
        setSaveState('error');
      }
    } catch (e) {
      setSaveState('error');
    } finally {
      setTimeout(() => setSaveState('idle'), 2000);
    }
  };

  const handleNextNotebook = async () => {
    setNextState('loading');
    const currentNbFile = notebookName ? `${notebookName}.ipynb` : undefined;

    try {
      const result = await callBackend('labeling/next', {
        datasetPath: rawDatasetPath || 'raw_dataset',
        current_filename: currentNbFile
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

  const handleFlagNotebook = async (reason: string) => {
    setFlagState('loading');
    setShowFlagMenu(false);

    const nbFilename = `${notebookName}.ipynb`;

    try {
      const result = await callBackend('labeling/flag', {
        filename: nbFilename,
        reason: reason
      });

      if (result.success) {
        setFlagState('success');
        await handleNextNotebook();
      } else {
        setFlagState('error');
      }
    } catch (e) {
      setFlagState('error');
    } finally {
      setTimeout(() => setFlagState('idle'), 2000);
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

          <div className="flex relative items-stretch rounded shadow-md">
            <a
              onClick={handleNextNotebook}
              className={`flex items-center justify-center px-5 py-2 rounded-l cursor-pointer font-medium !text-white no-underline transition-colors duration-200 ${
                nextState === 'success'
                  ? 'bg-green-600'
                  : nextState === 'error'
                    ? 'bg-red-600'
                    : nextState === 'loading'
                      ? 'bg-gray-600 !cursor-wait'
                      : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              {nextState === 'idle' && 'Next'}
              {nextState === 'loading' && 'Loading...'}
              {nextState === 'success' && 'Opening ✓'}
              {nextState === 'error' && 'Empty ✗'}
            </a>

            <a
              onClick={() => setShowFlagMenu(!showFlagMenu)}
              className={`flex items-center justify-center px-3 py-2 rounded-r cursor-pointer font-medium !text-white no-underline transition-colors duration-200 border-l border-gray-600 ${
                flagState === 'loading'
                  ? 'bg-orange-400 !cursor-wait'
                  : flagState === 'error'
                    ? 'bg-red-600'
                    : 'bg-gray-800 hover:bg-gray-700'
              }`}
              title="Flag & Skip"
            >
              Flag
            </a>

            {showFlagMenu && (
              <div className="absolute top-full right-0 mt-2 w-56 bg-white border border-gray-200 rounded shadow-xl z-50 overflow-hidden">
                <ul className="m-0 p-0 list-none text-sm text-gray-700 py-1">
                  <li className="m-0 p-0">
                    <a
                      onClick={() => handleFlagNotebook('Bug in Dataflow')}
                      className="flex items-center w-full px-4 py-2 hover:bg-gray-100 cursor-pointer no-underline !text-gray-800 transition-colors"
                    >
                      <span className="mr-3">🚩</span> Bug in Dataflow
                    </a>
                  </li>
                  <li className="m-0 p-0">
                    <a
                      onClick={() => handleFlagNotebook('Not Relevant')}
                      className="flex items-center w-full px-4 py-2 hover:bg-gray-100 cursor-pointer no-underline !text-gray-800 transition-colors"
                    >
                      <span className="mr-3">🗑️</span> Not Relevant
                    </a>
                  </li>
                  <li className="m-0 p-0 border-t border-gray-100">
                    <a
                      onClick={() => handleFlagNotebook('Must Revisit')}
                      className="flex items-center w-full px-4 py-2 hover:bg-gray-100 cursor-pointer no-underline !text-gray-800 transition-colors"
                    >
                      <span className="mr-3">⏳</span> Must Revisit
                    </a>
                  </li>
                </ul>
              </div>
            )}
          </div>
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
