import { ExtractedCell } from './NotebookTypes';

export type GraphMode = 'dataflow' | 'lineage' | 'ast' | 'vamsa' | 'labeling';
export type TableMode = 'environment';

export interface LegendItemType {
  type: 'node' | 'edge';
  color: string;
  label: string;
  borderStyle: string;
  numericType: number;
  category?: string;
}

export interface GraphProps {
  graphData: { elements: any[] };
  mode: GraphMode;
  resetTrigger: number;
  taxonomy: any;
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
