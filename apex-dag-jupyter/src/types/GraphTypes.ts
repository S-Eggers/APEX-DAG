export type GraphMode = 'dataflow' | 'lineage' | 'ast' | 'vamsa';

export interface LegendItemType {
  type: 'node' | 'edge';
  color: string;
  label: string;
  borderStyle: 'solid' | 'dashed';
  numericType: number;
}

export interface GraphProps {
  graphData: { elements: any[] };
  mode: GraphMode;
  resetTrigger: number;
}
