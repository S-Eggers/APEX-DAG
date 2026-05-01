import { useState, useEffect, useMemo } from 'react';
import { getBackend } from '../utils/callBackend';
import {
  GraphMode,
  LegendItemType,
  LabelOption,
  TaxonomyAPIResponse,
  TaxonomyModeData,
  GraphElementPayload
} from '../types/GraphTypes';

const DEFAULT_COLOR = '#B0E0E6';

export function useGraphTaxonomy(mode: GraphMode) {
  const [taxonomy, setTaxonomy] = useState<TaxonomyModeData | null>(null);

  useEffect(() => {
    let isMounted = true;

    getBackend('constants')
      .then((data: TaxonomyAPIResponse) => {
        if (isMounted && data.success && data.taxonomy) {
          setTaxonomy(data.taxonomy[mode]);
        }
      })
      .catch((err: unknown) => {
        console.error(`Failed to fetch taxonomy for mode: ${mode}`, err);
      });

    return () => {
      isMounted = false;
    };
  }, [mode]);

  return useMemo(() => {
    if (!taxonomy) {
      return {
        isLoaded: false,
        legends: [] as LegendItemType[],
        getNodeColor: (_type: number | null | undefined) => DEFAULT_COLOR,
        getEdgeColor: (_type: number | null | undefined) => DEFAULT_COLOR,
        nodeLabelOptions: [] as LabelOption[],
        edgeLabelOptions: [] as LabelOption[]
      };
    }

    const legends: LegendItemType[] = [];
    const nodeLabelOptions: LabelOption[] = [];
    const edgeLabelOptions: LabelOption[] = [];

    // Process Nodes
    Object.entries(taxonomy.nodes).forEach(([idStr, meta]) => {
      const numericType = parseInt(idStr, 10);
      legends.push({
        type: 'node',
        numericType,
        name: meta.name,
        label: meta.label,
        color: meta.color,
        borderStyle: meta.border_style,
        category: meta.category
      });
      nodeLabelOptions.push({ value: numericType, label: meta.label });
    });

    // Process Edges
    Object.entries(taxonomy.edges).forEach(([idStr, meta]) => {
      const numericType = parseInt(idStr, 10);
      legends.push({
        type: 'edge',
        numericType,
        name: meta.name,
        label: meta.label,
        color: meta.color,
        borderStyle: meta.border_style,
        category: meta.category
      });
      edgeLabelOptions.push({ value: numericType, label: meta.label });
    });

    const getNodeColor = (numericType: number | undefined | null): string => {
      if (numericType == null) return DEFAULT_COLOR;
      const meta = taxonomy.nodes[numericType.toString()];
      return meta ? meta.color : DEFAULT_COLOR;
    };

    const getEdgeColor = (numericType: number | undefined | null): string => {
      if (numericType == null) return DEFAULT_COLOR;
      const meta = taxonomy.edges[numericType.toString()];
      return meta ? meta.color : DEFAULT_COLOR;
    };

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

// Data Utility functions
export const groupLegendItems = (
  allLegendItems: LegendItemType[]
): Record<string, LegendItemType[]> => {
  return allLegendItems.reduce(
    (acc, item) => {
      const cat = item.category || 'Uncategorized';
      acc[cat] = acc[cat] || [];
      acc[cat].push(item);
      return acc;
    },
    {} as Record<string, LegendItemType[]>
  );
};

export const filterLegendItems = (
  elements: GraphElementPayload[],
  allLegendItems: LegendItemType[]
): LegendItemType[] => {
  if (!elements || elements.length === 0) return [];

  const presentNodeTypes = new Set<number>();
  const presentEdgeTypes = new Set<number>();

  elements.forEach(({ data }) => {
    if (data.node_type != null) presentNodeTypes.add(Number(data.node_type));
    if (data.predicted_label != null)
      presentEdgeTypes.add(Number(data.predicted_label));
    if (data.edge_type != null) presentEdgeTypes.add(Number(data.edge_type));
  });

  return allLegendItems.filter(item => {
    const target = Number(item.numericType);
    return item.type === 'node'
      ? presentNodeTypes.has(target)
      : presentEdgeTypes.has(target);
  });
};
