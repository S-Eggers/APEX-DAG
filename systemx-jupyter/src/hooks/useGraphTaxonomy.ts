import { useState, useEffect, useMemo } from 'react';
import { getBackend } from '../utils/callBackend';
import {
  GraphMode,
  LegendItemType,
  LabelOption,
  TaxonomyAPIResponse,
  TaxonomyModeData,
  TaxonomyState,
  GraphElementPayload
} from '../types/GraphTypes';

const DEFAULT_COLOR = '#B0E0E6';

export function useGraphTaxonomy(mode: GraphMode): TaxonomyState {
  const [taxonomy, setTaxonomy] = useState<TaxonomyModeData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    setError(null);

    getBackend<TaxonomyAPIResponse>('constants')
      .then(data => {
        if (!isMounted) return;
        if (data.success && data.taxonomy) {
          setTaxonomy(data.taxonomy[mode]);
        } else {
          setError('Could not load the graph taxonomy from the backend.');
        }
      })
      .catch((err: Error) => {
        console.error(
          `Failed to fetch taxonomy for mode: ${mode}`,
          err.message
        );
        if (isMounted) {
          setError('Could not reach the SystemX backend to load the legend.');
        }
      });

    return () => {
      isMounted = false;
    };
  }, [mode]);

  return useMemo(() => {
    if (!taxonomy) {
      return {
        isLoaded: false,
        error,
        legends: [],
        getNodeColor: (_type: number | null | undefined) => DEFAULT_COLOR,
        getEdgeColor: (_type: number | null | undefined) => DEFAULT_COLOR,
        getNodeLabel: (_type: number | null | undefined) => undefined,
        getEdgeLabel: (_type: number | null | undefined) => undefined,
        nodeLabelOptions: [],
        edgeLabelOptions: [],
        hubLabelOptions: [],
        hubTypes: new Set<number>(),
        hasHubs: false,
        leakageGold: [],
        getGoldColor: (_gold: string | null | undefined) => DEFAULT_COLOR,
        getGoldEntry: (_gold: string | null | undefined) => undefined
      };
    }

    const leakageGold = taxonomy.gold ?? [];
    const goldByKey = new Map(leakageGold.map(entry => [entry.key, entry]));
    const getGoldEntry = (gold: string | null | undefined) =>
      gold != null ? goldByKey.get(gold) : undefined;
    const getGoldColor = (gold: string | null | undefined) =>
      getGoldEntry(gold)?.color ?? DEFAULT_COLOR;

    const legends: LegendItemType[] = [];
    const nodeLabelOptions: LabelOption[] = [];
    const edgeLabelOptions: LabelOption[] = [];
    const hubLabelOptions: LabelOption[] = [];

    const rawHubTypes = taxonomy.hub_types as unknown;
    let hubTypesArray: number[] = [];
    if (Array.isArray(rawHubTypes)) {
      hubTypesArray = rawHubTypes;
    } else if (typeof rawHubTypes === 'number') {
      hubTypesArray = [rawHubTypes];
    }
    const hubTypes = new Set<number>(hubTypesArray);

    Object.entries(taxonomy.nodes).forEach(([idStr, meta]) => {
      const numericType = parseInt(idStr, 10);
      legends.push({
        type: 'node',
        numericType,
        name: meta.name,
        label: meta.label,
        color: meta.color,
        borderStyle: meta.border_style,
        category: meta.category,
        space: 'structural'
      });
      nodeLabelOptions.push({ value: numericType, label: meta.label });
    });

    if (taxonomy.domain_nodes) {
      Object.entries(taxonomy.domain_nodes).forEach(([idStr, meta]) => {
        const numericType = parseInt(idStr, 10);
        legends.push({
          type: 'node',
          numericType,
          name: meta.name,
          label: meta.label,
          color: meta.color,
          borderStyle: meta.border_style,
          category: meta.category,
          space: 'domain'
        });
      });
    }

    if (taxonomy.hubs) {
      Object.entries(taxonomy.hubs).forEach(([idStr, meta]) => {
        const numericType = parseInt(idStr, 10);

        legends.push({
          type: 'node',
          numericType,
          name: meta.name,
          label: meta.label,
          color: meta.color,
          borderStyle: meta.border_style,
          category: 'Hub Nodes'
        });
        hubLabelOptions.push({ value: numericType, label: meta.label });
      });
    }

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

    const getNodeColor = (
      numericType: number | undefined | null,
      isHub: boolean = false,
      isDomain: boolean = false
    ): string => {
      if (numericType == null) return DEFAULT_COLOR;

      const idStr = numericType.toString();

      if (isHub && taxonomy.hubs && taxonomy.hubs[idStr]) {
        return taxonomy.hubs[idStr].color;
      }
      if (isDomain && taxonomy.domain_nodes && taxonomy.domain_nodes[idStr]) {
        return taxonomy.domain_nodes[idStr].color;
      }

      const meta = taxonomy.nodes[idStr];
      return meta ? meta.color : DEFAULT_COLOR;
    };

    const hasHubs = !!taxonomy.hubs && Object.keys(taxonomy.hubs).length > 0;

    const edgeMeta = (numericType: number, semantic: boolean) => {
      const idStr = numericType.toString();
      const table = semantic && hasHubs ? taxonomy.hubs! : taxonomy.edges;
      return table[idStr];
    };

    const getEdgeColor = (
      numericType: number | undefined | null,
      semantic: boolean = false
    ): string => {
      if (numericType == null) return DEFAULT_COLOR;
      return edgeMeta(numericType, semantic)?.color ?? DEFAULT_COLOR;
    };

    const getNodeLabel = (
      numericType: number | undefined | null,
      isHub: boolean = false,
      isDomain: boolean = false
    ): string | undefined => {
      if (numericType == null) return undefined;
      const idStr = numericType.toString();
      if (isHub && taxonomy.hubs && taxonomy.hubs[idStr]) {
        return taxonomy.hubs[idStr].label;
      }
      if (isDomain && taxonomy.domain_nodes && taxonomy.domain_nodes[idStr]) {
        return taxonomy.domain_nodes[idStr].label;
      }
      return taxonomy.nodes[idStr]?.label;
    };

    const getEdgeLabel = (
      numericType: number | undefined | null,
      semantic: boolean = false
    ): string | undefined => {
      if (numericType == null) return undefined;
      return edgeMeta(numericType, semantic)?.label;
    };

    return {
      isLoaded: true,
      error: null,
      legends,
      getNodeColor,
      getEdgeColor,
      getNodeLabel,
      getEdgeLabel,
      nodeLabelOptions,
      edgeLabelOptions,
      hubLabelOptions,
      hubTypes,
      hasHubs,
      leakageGold,
      getGoldColor,
      getGoldEntry
    };
  }, [taxonomy, mode, error]);
}

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
  allLegendItems: LegendItemType[],
  hubTypes: Set<number> = new Set(),
  hasHubTable: boolean = false
): LegendItemType[] => {
  if (!elements || elements.length === 0) return [];

  const standardNodeCounts = new Map<number, number>();
  const hubNodeCounts = new Map<number, number>();
  const domainNodeCounts = new Map<number, number>();
  const structuralEdgeCounts = new Map<number, number>();
  const semanticEdgeCounts = new Map<number, number>();

  const bump = (m: Map<number, number>, k: number) =>
    m.set(k, (m.get(k) || 0) + 1);

  elements.forEach(({ data }) => {
    const structuralType =
      data.node_type != null ? Number(data.node_type) : null;
    const predictedLabel =
      data.predicted_label != null ? Number(data.predicted_label) : null;
    const edgeType = data.edge_type != null ? Number(data.edge_type) : null;
    const isDomainNode = data.domain_node === true;

    if (structuralType != null) {
      const isHub = hubTypes.has(structuralType) && !isDomainNode;
      if (isHub) {
        bump(hubNodeCounts, predictedLabel != null ? predictedLabel : -1);
      } else if (isDomainNode) {
        bump(domainNodeCounts, structuralType);
      } else {
        bump(
          standardNodeCounts,
          predictedLabel != null ? predictedLabel : structuralType
        );
      }
    } else if (edgeType != null) {
      if (predictedLabel != null) {
        bump(semanticEdgeCounts, predictedLabel);
      } else {
        bump(structuralEdgeCounts, edgeType);
      }
    }
  });

  return allLegendItems
    .filter(item => {
      const target = Number(item.numericType);
      if (item.type === 'edge') {
        return (
          structuralEdgeCounts.has(target) ||
          (!hasHubTable && semanticEdgeCounts.has(target))
        );
      }
      if (item.type === 'node') {
        if (item.category === 'Hub Nodes') {
          return (
            hubNodeCounts.has(target) ||
            (hasHubTable && semanticEdgeCounts.has(target))
          );
        }
        if (item.space === 'domain') {
          return domainNodeCounts.has(target);
        }
        return standardNodeCounts.has(target);
      }
      return false;
    })
    .map(item => {
      const target = Number(item.numericType);
      let count = 0;

      if (item.type === 'edge') {
        count =
          (structuralEdgeCounts.get(target) || 0) +
          (!hasHubTable ? semanticEdgeCounts.get(target) || 0 : 0);
      } else if (item.category === 'Hub Nodes') {
        count =
          (hubNodeCounts.get(target) || 0) +
          (hasHubTable ? semanticEdgeCounts.get(target) || 0 : 0);
      } else if (item.space === 'domain') {
        count = domainNodeCounts.get(target) || 0;
      } else {
        count = standardNodeCounts.get(target) || 0;
      }

      return { ...item, count };
    });
};
