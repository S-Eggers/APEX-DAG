import { useEffect, useRef, useCallback } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import { GraphMode } from '../types/GraphTypes';

if (!cytoscape.prototype.dagre) {
  cytoscape.use(dagre);
}

export function useCytoscape(
  containerRef: React.RefObject<HTMLDivElement>,
  graphData: { elements: any[] },
  mode: GraphMode,
  resetTrigger: number,
  onNodeSelect: (nodeData: any | null) => void,
  onEdgeSelect: (edge: any | null) => void,
  onSyncLegend: () => void,
  getNodeColor: (type: number) => string,
  getEdgeColor: (type: number) => string
) {
  const cyRef = useRef<cytoscape.Core | null>(null);

  const callbacksRef = useRef({ onNodeSelect, onEdgeSelect, onSyncLegend });
  useEffect(() => {
    callbacksRef.current = { onNodeSelect, onEdgeSelect, onSyncLegend };
  });

  const edgeTypeColor = useCallback(
    (element: any) => {
      const predicted = element.data('predicted_label');
      if (predicted !== undefined && predicted !== null) {
        return getEdgeColor(Number(predicted));
      }

      const edgeType = element.data('edge_type');
      if (edgeType !== undefined && edgeType !== null) {
        const numType = Number(edgeType);
        if (!isNaN(numType)) {
          return getEdgeColor(numType);
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
    (element: any) => {
      return getNodeColor(element.data('node_type'));
    },
    [getNodeColor]
  );

  const lineTypeStyle = useCallback(
    (element: any) => {
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
      callbacksRef.current.onNodeSelect(evt.target.data());
      cy.elements().removeClass('selected');
      evt.target.addClass('selected');
    });

    cy.on('tap', 'edge', evt => {
      callbacksRef.current.onNodeSelect(null);
      callbacksRef.current.onEdgeSelect(evt.target);
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
          shape: 'round-rectangle',
          'background-color': (ele: any) => nodeTypeColor(ele),
          label: (ele: any) => {
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
          color: '#333',

          'text-wrap': 'wrap',
          'text-max-width': '120px',

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

    cy.elements().remove();
    cy.add(graphData.elements);

    callbacksRef.current.onNodeSelect(null);
    callbacksRef.current.onEdgeSelect(null);

    callbacksRef.current.onSyncLegend();

    const prevTrigger = cy.data('prevResetTrigger');
    if (prevTrigger !== undefined && resetTrigger !== prevTrigger) {
      cy.fit(undefined, 50);
    }
    cy.data('prevResetTrigger', resetTrigger);

    const layoutInstance = cy.layout({
      name: 'dagre'
    });

    layoutInstance.on('layoutstop', () => {
      cy.fit(undefined, 50);
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

  return cyRef;
}
