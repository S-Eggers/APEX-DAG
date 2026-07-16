import { useEffect, useRef, useCallback } from 'react';
import cytoscape, {
  NodeSingular,
  EdgeSingular,
  StylesheetStyle
} from 'cytoscape';
import dagre from 'cytoscape-dagre';
import {
  GraphMode,
  GraphNodeData,
  GraphEdgeData,
  CyElement
} from '../types/GraphTypes';
import { adjustForDarkMode } from '../utils/colorTheme';

if (!cytoscape.prototype.dagre) {
  cytoscape.use(dagre);
}

const MAX_EDGE_LABEL_CHARS = 32;

function truncateLabel(text: string, max: number): string {
  if (!text) return '';
  return text.length > max ? `${text.slice(0, max - 1)}...` : text;
}

export function useCytoscape(
  containerRef: React.RefObject<HTMLDivElement>,
  graphData: { elements: unknown[] },
  mode: GraphMode,
  resetTrigger: number,
  hubTypes: Set<number>,
  isDark: boolean,
  onNodeSelect: (nodeData: GraphNodeData | null) => void,
  onEdgeSelect: (edge: CyElement<GraphEdgeData> | null) => void,
  onSyncLegend: () => void,
  getNodeColor: (type: number, isHub: boolean, isDomain?: boolean) => string,
  getEdgeColor: (type: number, semantic?: boolean) => string,
  getGoldColor: (goldKey: string | undefined | null) => string
) {
  const cyRef = useRef<cytoscape.Core | null>(null);

  const callbacksRef = useRef({ onNodeSelect, onEdgeSelect, onSyncLegend });
  useEffect(() => {
    callbacksRef.current = { onNodeSelect, onEdgeSelect, onSyncLegend };
  });

  const edgeTypeColor = useCallback(
    (element: EdgeSingular) => {
      const predicted = element.data('predicted_label');
      if (predicted !== undefined && predicted !== null) {
        return getEdgeColor(Number(predicted), true);
      }

      const edgeType = element.data('edge_type');
      if (edgeType !== undefined && edgeType !== null) {
        const numType = Number(edgeType);
        if (!isNaN(numType)) {
          return getEdgeColor(numType, false);
        }
      }

      const label = String(element.data('label') || '').toLowerCase();
      const rawEdgeType = String(edgeType || '').toLowerCase();
      const semanticString = label + rawEdgeType;

      if (mode === 'vamsa_wir' || mode === 'vamsa_lineage') {
        if (semanticString.includes('caller')) return getEdgeColor(0);
        if (semanticString.includes('input')) return getEdgeColor(1);
        if (
          semanticString.includes('output') ||
          semanticString.includes('transform')
        )
          return getEdgeColor(2);
      }

      return '#d3d3d3';
    },
    [getEdgeColor, mode]
  );

  const nodeTypeColor = useCallback(
    (element: NodeSingular) => {
      const structuralType = element.data('node_type');
      const predictedLabel = element.data('predicted_label');
      const isDomain = element.data('domain_node') === true;
      const isHub =
        !isDomain &&
        structuralType !== undefined &&
        hubTypes.has(Number(structuralType));

      let raw: string;
      if (isHub) {
        raw = getNodeColor(predictedLabel != null ? predictedLabel : -1, true);
      } else if (isDomain) {
        raw = getNodeColor(structuralType, false, true);
      } else {
        raw = getNodeColor(predictedLabel ?? structuralType, false);
      }

      return isDark ? adjustForDarkMode(raw) : raw;
    },
    [getNodeColor, hubTypes, isDark]
  );

  const goldNodeColor = useCallback(
    (element: NodeSingular) => {
      const raw = getGoldColor(element.data('leakage_gold'));
      return isDark ? adjustForDarkMode(raw) : raw;
    },
    [getGoldColor, isDark]
  );

  const nodeShape = useCallback(
    (element: NodeSingular): string => {
      const isDomain = element.data('domain_node') === true;
      const nodeType = element.data('node_type');
      return !isDomain &&
        nodeType !== undefined &&
        hubTypes.has(Number(nodeType))
        ? 'hexagon'
        : 'round-rectangle';
    },
    [hubTypes]
  );

  const lineTypeStyle = useCallback(
    (element: EdgeSingular) => {
      if (mode === 'dataflow' && element.data('edge_type') === 2) {
        return 'dashed';
      }
      return 'solid';
    },
    [mode]
  );

  useEffect(() => {
    if (!containerRef.current) return;

    const cy = cytoscape({
      container: containerRef.current,
      elements: [],
      layout: { name: 'preset' },
      panningEnabled: true,
      zoomingEnabled: true,
      wheelSensitivity: 0.2
    });

    cyRef.current = cy;

    cy.on('tap', 'node', evt => {
      callbacksRef.current.onEdgeSelect(null);
      callbacksRef.current.onNodeSelect(evt.target.data() as GraphNodeData);
      cy.elements().removeClass('selected');
      evt.target.addClass('selected');
    });

    cy.on('tap', 'edge', evt => {
      callbacksRef.current.onNodeSelect(null);
      callbacksRef.current.onEdgeSelect(
        evt.target as unknown as CyElement<GraphEdgeData>
      );
      cy.elements().removeClass('selected');
      evt.target.addClass('selected');
    });

    cy.on('tap', evt => {
      if (evt.target === cy) {
        callbacksRef.current.onNodeSelect(null);
        callbacksRef.current.onEdgeSelect(null);
        cy.elements().removeClass('selected');
      }
    });

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    cy.style([
      {
        selector: 'node',
        style: {
          shape: (ele: NodeSingular) => nodeShape(ele),
          'background-color': (ele: NodeSingular) => nodeTypeColor(ele),
          label: (ele: NodeSingular) => {
            const baseLabel = ele.data('label') || '';
            if (mode === 'vamsa_wir' || mode === 'vamsa_lineage') {
              const annotations = ele.data('annotations');
              if (Array.isArray(annotations) && annotations.length > 0) {
                return `${baseLabel}\n[${annotations.join(', ')}]`;
              }
            }
            return baseLabel;
          },
          width: '60px',
          height: '35px',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '12px',
          color: isDark ? '#e8e8e8' : '#333',
          'text-wrap': 'wrap',
          'text-max-width': '120px',
          'border-width': 1,
          'border-color': isDark ? '#555' : '#ccc'
        }
      },
      ...(mode === 'leakage'
        ? [
            {
              selector: 'node[leakage_gold]',
              style: {
                'background-color': (ele: NodeSingular) => goldNodeColor(ele)
              }
            }
          ]
        : []),
      {
        selector: 'node[?has_leakage]',
        style: {
          'border-width': 3,
          'border-color': '#ef4444',
          'border-style': 'dashed',
          'shadow-blur': 12,
          'shadow-color': '#ef4444',
          'shadow-opacity': 0.55
        }
      },
      {
        selector: 'node.selected',
        style: {
          'border-width': 3,
          'border-color': '#2563eb',
          'border-style': 'solid',
          'shadow-blur': 10,
          'shadow-color': '#2563eb',
          'shadow-opacity': 0.5
        }
      },
      {
        selector: 'node.tuple-subject',
        style: {
          'border-width': 4,
          'border-color': '#16a34a',
          'border-style': 'solid',
          'shadow-blur': 12,
          'shadow-color': '#16a34a',
          'shadow-opacity': 0.6
        }
      },
      {
        selector: 'node.tuple-object',
        style: {
          'border-width': 4,
          'border-color': '#d97706',
          'border-style': 'solid',
          'shadow-blur': 12,
          'shadow-color': '#d97706',
          'shadow-opacity': 0.6
        }
      },
      {
        selector: 'edge',
        style: {
          width: 2,
          'line-color': (ele: EdgeSingular) => edgeTypeColor(ele),
          'target-arrow-shape': 'triangle',
          'target-arrow-color': (ele: EdgeSingular) => edgeTypeColor(ele),
          'curve-style': 'bezier',
          label: (ele: EdgeSingular) =>
            truncateLabel(
              String(ele.data('label') || ''),
              MAX_EDGE_LABEL_CHARS
            ),
          'font-size': '10px',
          color: isDark ? '#d1d5db' : '#444',
          'text-rotation': 'autorotate',
          'text-wrap': 'wrap',
          'text-max-width': '160px',
          'text-background-color': isDark ? '#0d1117' : '#eef2f8',
          'text-background-opacity': 0.85,
          'text-background-shape': 'roundrectangle',
          'text-background-padding': '2px',
          'line-style': (ele: EdgeSingular) => lineTypeStyle(ele)
        }
      },
      {
        selector: 'edge.selected',
        style: {
          width: 3.5,
          label: 'data(label)',
          'text-background-opacity': 1,
          'z-index': 999
        }
      }
    ] as StylesheetStyle[]);

    cy.elements().remove();
    cy.add(graphData.elements as cytoscape.ElementDefinition[]);

    callbacksRef.current.onNodeSelect(null);
    callbacksRef.current.onEdgeSelect(null);

    callbacksRef.current.onSyncLegend();

    const prevTrigger = cy.data('prevResetTrigger');
    if (prevTrigger !== undefined && resetTrigger !== prevTrigger) {
      cy.fit(undefined, 50);
    }
    cy.data('prevResetTrigger', resetTrigger);

    const isLineage = mode === 'lineage' || mode === 'vamsa_lineage';
    const layoutInstance = cy.layout({
      name: 'dagre',
      rankDir: 'TB',
      nodeSep: isLineage ? 80 : 40,
      edgeSep: isLineage ? 40 : 20,
      rankSep: isLineage ? 160 : 70,
      ranker: 'network-simplex'
    } as cytoscape.LayoutOptions);

    layoutInstance.on('layoutstop', () => {
      cy.fit(undefined, 50);
    });

    layoutInstance.run();
  }, [
    graphData,
    mode,
    resetTrigger,
    isDark,
    nodeShape,
    nodeTypeColor,
    goldNodeColor,
    edgeTypeColor,
    lineTypeStyle
  ]);

  return cyRef;
}
