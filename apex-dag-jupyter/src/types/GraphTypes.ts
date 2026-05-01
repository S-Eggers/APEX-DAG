import { ExtractedCell } from './NotebookTypes';

export type GraphMode = 'dataflow' | 'lineage' | 'ast' | 'vamsa' | 'labeling';
export type TableMode = 'environment';

export type BorderStyle = 'solid' | 'dashed' | 'dotted';

export interface TransformStep {
  operation: string;
  target_node: string;
  transform_code?: string;
}

export interface GraphNodeData {
  id: string;
  cell_id?: string;
  node_type?: number;
  domain_label?: string;
  label?: string;
  base_inputs?: string;
  transform_history?: TransformStep[];
  code?: string;
}

export interface GraphEdgeData {
  id: string;
  cell_id?: string;
  edge_type?: number;
  predicted_label?: number;
  domain_label?: string;
  label?: string;
  raw_code?: string;
  source?: string;
  target?: string;
}

export interface CyElement<T> {
  id(): string;
  data(): T;
  data(key: string, value: string | number): void;
  source?(): CyElement<GraphNodeData>;
  target?(): CyElement<GraphNodeData>;
}

export interface ElementMetadata {
  name: string;
  category: string;
  label: string;
  color: string;
  border_style: BorderStyle;
}

export interface TaxonomyModeData {
  nodes: Record<string, ElementMetadata>;
  edges: Record<string, ElementMetadata>;
}

export interface TaxonomyAPIResponse {
  success: boolean;
  taxonomy: Record<GraphMode, TaxonomyModeData>;
}

export interface LegendItemType {
  type: 'node' | 'edge';
  numericType: number;
  label: string;
  color: string;
  borderStyle: BorderStyle;
  category: string;
  name: string;
}

export interface LabelOption {
  value: number;
  label: string;
}

export interface GraphElementPayload {
  data: {
    node_type?: number | string;
    predicted_label?: number | string;
    edge_type?: number | string;
  };
}

export interface TaxonomyState {
  isLoaded: boolean;
  legends: LegendItemType[];
  getNodeColor: (type: number | undefined | null) => string;
  getEdgeColor: (type: number | undefined | null) => string;
  nodeLabelOptions: LabelOption[];
  edgeLabelOptions: LabelOption[];
}

export interface GraphProps {
  graphData: { elements: unknown[] };
  mode: GraphMode;
  resetTrigger: number;
  taxonomy: TaxonomyState;
  notebookName: string;
  notebookCode: ExtractedCell[];
  rawDatasetPath: string;
  onLocateCell?: (cellId: string) => void;
  onNextNotebook?: (path: string) => void;
}

export interface GraphComponentProps {
  graphData: { elements: any[] };
  mode: GraphMode;
  resetTrigger: number;
  notebookName: string;
  notebookCode: ExtractedCell[];
  rawDatasetPath: string;
  onLocateCell?: (cellId: string) => void;
  onNextNotebook?: (path: string) => void;
}

export interface LabelOption {
  value: number;
  label: string;
}
