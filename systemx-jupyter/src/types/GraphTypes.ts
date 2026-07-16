import { ExtractedCell } from './NotebookTypes';

export interface LineageTuple {
  tuple_type: '<D, D>' | '<M, D>' | '<D, Empty>';
  subject_id: string;
  object_id: string;
}

export type GraphMode =
  | 'dataflow'
  | 'lineage'
  | 'lineage_annotation'
  | 'ast'
  | 'vamsa_wir'
  | 'vamsa_lineage'
  | 'labeling'
  | 'leakage';
export type TableMode = 'environment';

export type BorderStyle = 'solid' | 'dashed' | 'dotted';

export interface TransformStep {
  operation: string;
  target_node: string;
  transform_code?: string;
}

export interface FeatureDimImportance {
  index: number;
  name: string;
  score: number;
  value?: number;
}

export interface FeatureScalarImportance {
  name: string;
  description?: string;
  score: number;
  value?: number;
}

export interface FeatureGroupImportance {
  key: string;
  name: string;
  description?: string;
  score: number;
  dims?: FeatureDimImportance[];
  scalars?: FeatureScalarImportance[];
}

export interface FeatureImportance {
  model?: string;
  predicted_class?: number;
  groups: FeatureGroupImportance[];
}

export interface GraphNodeData {
  id: string;
  cell_id?: string;
  node_type?: number;
  domain_node?: boolean;
  domain_label?: string;
  predicted_label?: number;
  label?: string;
  base_inputs?: string;
  transform_history?: TransformStep[];
  code?: string;
  feature_importance?: FeatureImportance;
  has_leakage?: boolean;
  leakage_class?: string;
  leakage_gold?: string;
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

export interface LeakageGoldEntry {
  key: string;
  label: string;
  category: string;
  color: string;
  description: string;
  detector?: string | null;
}

export interface TaxonomyModeData {
  nodes: Record<string, ElementMetadata>;
  edges: Record<string, ElementMetadata>;
  hubs?: Record<string, ElementMetadata>;
  hub_types?: number[];
  domain_nodes?: Record<string, ElementMetadata>;
  gold?: LeakageGoldEntry[];
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
  count?: number;
  space?: 'structural' | 'domain';
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
    domain_node?: boolean;
  };
}

export interface TaxonomyState {
  isLoaded: boolean;
  error?: string | null;
  legends: LegendItemType[];
  getNodeColor: (
    type: number | undefined | null,
    isHub?: boolean,
    isDomain?: boolean
  ) => string;
  getEdgeColor: (type: number | undefined | null, semantic?: boolean) => string;
  getNodeLabel: (
    type: number | undefined | null,
    isHub?: boolean,
    isDomain?: boolean
  ) => string | undefined;
  getEdgeLabel: (
    type: number | undefined | null,
    semantic?: boolean
  ) => string | undefined;
  nodeLabelOptions: LabelOption[];
  edgeLabelOptions: LabelOption[];
  hubLabelOptions: LabelOption[];
  hubTypes: Set<number>;
  hasHubs: boolean;
  leakageGold: LeakageGoldEntry[];
  getGoldColor: (goldKey: string | undefined | null) => string;
  getGoldEntry: (
    goldKey: string | undefined | null
  ) => LeakageGoldEntry | undefined;
}

export interface GraphProps {
  graphData: { elements: unknown[] };
  mode: GraphMode;
  resetTrigger: number;
  taxonomy: TaxonomyState;
  notebookName: string;
  notebookCode: ExtractedCell[];
  labelingConfig: LabelingConfig;
  nnBackend?: string;
  featurePreset?: string;
  modelVariant?: string;
  explainFeatureImportance?: boolean;
  backendError?: string | null;
  onLocateCell?: (cellId: string) => void;
  onNextNotebook?: (path: string) => void;
  onRelabel?: (elements: unknown[]) => void;
  onFocusNode?: (focusFn: (nodeId: string) => void) => void;
  onVariantTrained?: (variantKey: string) => void;
  tuples?: LineageTuple[];
}

export interface GraphComponentProps {
  graphData: { elements: any[] };
  mode: GraphMode;
  resetTrigger: number;
  notebookName: string;
  notebookCode: ExtractedCell[];
  labelingConfig: LabelingConfig;
  nnBackend?: string;
  featurePreset?: string;
  modelVariant?: string;
  explainFeatureImportance?: boolean;
  backendError?: string | null;
  onLocateCell?: (cellId: string) => void;
  onNextNotebook?: (path: string) => void;
  onRelabel?: (elements: unknown[]) => void;
  onFocusNode?: (focusFn: (nodeId: string) => void) => void;
  onVariantTrained?: (variantKey: string) => void;
  tuples?: LineageTuple[];
}

export interface LabelOption {
  value: number;
  label: string;
}

export interface LabelingConfig {
  rawDatasetPath: string;
  defaultFetchMode: 'unannotated' | 'annotated' | 'flagged';
  defaultFlagType: string;
}
