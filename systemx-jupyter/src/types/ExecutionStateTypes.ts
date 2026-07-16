export type CellStatus =
  | 'fresh'
  | 'stale'
  | 'dirty'
  | 'unexecuted'
  | 'missing-deps'
  | 'unknown';

export interface CellStateInfo {
  cell_id: string;
  status: CellStatus;
  reasons: string[];
  predicted_rank: number;
  confidence: number;
  unsafe_reorder: boolean;
  hidden_state_risk: boolean;
}

export interface DependencyEdgeInfo {
  src_cell: string;
  dst_cell: string;
  kind:
    | 'DATA'
    | 'IMPORT'
    | 'FUNCTION_DEF'
    | 'CLASS_DEF'
    | 'FILE_ARTIFACT'
    | 'OPAQUE';
  name: string;
  def_lineno: number;
  use_lineno: number;
  ambiguous: boolean;
  candidate_def_cells: string[];
  out_of_order: boolean;
  confidence: number;
}

export interface CellGraphNodeInfo {
  cell_id: string;
  document_index: number;
  execution_count: number | null;
  defined_names: string[];
  free_uses: string[];
  undefined_uses: string[];
  has_opaque_effects: boolean;
  is_dirty: boolean | null;
  source_preview: string;
}

export interface ExecutionStateData {
  predicted_order: string[];
  constraints: DependencyEdgeInfo[];
  ambiguities: DependencyEdgeInfo[];
  cell_states: Record<string, CellStateInfo>;
  notebook_flags: {
    safe_to_run_top_to_bottom: boolean;
    hidden_state_risk: boolean;
    out_of_order_evidence: boolean;
    dropped_cycle_edges: DependencyEdgeInfo[];
  };
  cell_graph: {
    cells: CellGraphNodeInfo[];
    edges: DependencyEdgeInfo[];
    minimal_constraints: DependencyEdgeInfo[];
    ambiguities: DependencyEdgeInfo[];
  };
}

export const STATUS_COLORS: Record<CellStatus, string> = {
  fresh: '#22c55e',
  stale: '#f59e0b',
  dirty: '#f97316',
  unexecuted: '#94a3b8',
  'missing-deps': '#8b5cf6',
  unknown: '#ef4444'
};
