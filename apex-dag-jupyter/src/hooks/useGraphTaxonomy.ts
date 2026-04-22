// hooks/useGraphTaxonomy.ts
import { useState, useEffect, useMemo } from 'react';
import { getBackend } from '../utils/callBackend';
import { GraphMode, LegendItemType, LabelOption } from '../types/GraphTypes';
import {
  SEMANTIC_COLORS,
  DEFAULT_NODE_COLOR,
  DEFAULT_EDGE_COLOR
} from '../config/GraphTheme';

interface TaxonomyResponse {
  nodes: Record<string, string>;
  edges: Record<string, string>;
}

export function useGraphTaxonomy(mode: GraphMode) {
  const [taxonomy, setTaxonomy] = useState<TaxonomyResponse | null>(null);

  useEffect(() => {
    getBackend('constants')
      .then(data => {
        if (data.success && data.taxonomy) {
          setTaxonomy(data.taxonomy[mode]);
        }
      })
      .catch(err => console.error('Failed to fetch taxonomy', err));
  }, [mode]);

  return useMemo(() => {
    if (!taxonomy) {
      return {
        isLoaded: false,
        legends: [],
        getNodeColor: () => DEFAULT_NODE_COLOR,
        getEdgeColor: () => DEFAULT_EDGE_COLOR,
        nodeLabelOptions: [],
        edgeLabelOptions: []
      };
    }

    const getNodeColor = (numericType: number | undefined | null) => {
      if (numericType === undefined || numericType === null) {
        return DEFAULT_NODE_COLOR;
      }

      const stringKey = taxonomy.nodes[numericType.toString()];
      return stringKey
        ? SEMANTIC_COLORS.nodes[stringKey] || DEFAULT_NODE_COLOR
        : DEFAULT_NODE_COLOR;
    };

    const getEdgeColor = (numericType: number | undefined | null) => {
      if (numericType === undefined || numericType === null) {
        return DEFAULT_EDGE_COLOR;
      }

      const stringKey = taxonomy.edges[numericType.toString()];
      return stringKey
        ? SEMANTIC_COLORS.edges[stringKey] || DEFAULT_EDGE_COLOR
        : DEFAULT_EDGE_COLOR;
    };

    const legends: LegendItemType[] = [];

    Object.entries(taxonomy.nodes).forEach(([idStr, stringKey]) => {
      legends.push({
        type: 'node',
        numericType: parseInt(idStr, 10),
        label: stringKey.replace(/_/g, ' '),
        color: SEMANTIC_COLORS.nodes[stringKey] || DEFAULT_NODE_COLOR,
        borderStyle: 'solid'
      });
    });

    Object.entries(taxonomy.edges).forEach(([idStr, stringKey]) => {
      legends.push({
        type: 'edge',
        numericType: parseInt(idStr, 10),
        label: stringKey.replace(/_/g, ' '),
        color: SEMANTIC_COLORS.edges[stringKey] || DEFAULT_EDGE_COLOR,
        borderStyle: stringKey === 'REASSIGN' ? 'dashed' : 'solid'
      });
    });

    const nodeLabelOptions: LabelOption[] = Object.entries(taxonomy.nodes).map(
      ([id, key]) => ({
        value: parseInt(id, 10),
        label: key.replace(/_/g, ' ')
      })
    );

    const edgeLabelOptions: LabelOption[] = Object.entries(taxonomy.edges).map(
      ([id, key]) => ({
        value: parseInt(id, 10),
        label: key.replace(/_/g, ' ')
      })
    );

    return {
      isLoaded: true,
      legends,
      getNodeColor,
      getEdgeColor,
      nodeLabelOptions,
      edgeLabelOptions
    };
  }, [taxonomy]);
}
