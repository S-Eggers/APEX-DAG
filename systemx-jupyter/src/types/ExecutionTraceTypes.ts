import { ExtractedCell } from './NotebookTypes';
import {
  CellGraphNodeInfo,
  DependencyEdgeInfo
} from './ExecutionStateTypes';

export interface TraceEvent {
  seq: number;
  cell_id: string;
  document_index: number;
  execution_count: number | null;
  source_hash: string;
  timestamp: string;
  success: boolean | null;
}

export type SessionEndReason =
  | 'restart'
  | 'kernel-changed'
  | 'shutdown'
  | 'reload'
  | null;

export interface TraceSession {
  session_id: string;
  kernel_id: string | null;
  started_at: string;
  ended_reason: SessionEndReason;
  events: TraceEvent[];
}

export interface NotebookTrace {
  schema_version: 1;
  notebook_path: string;
  saved_at: string;
  sessions: TraceSession[];
  sources: Record<string, string>;
  cells_snapshot: ExtractedCell[];
}

export type RerunKind = 'first' | 'identical' | 'edited';

export interface AnnotatedTraceEvent {
  seq: number;
  cell_id: string;
  document_index: number;
  execution_count: number | null;
  timestamp: string;
  success: boolean | null;
  in_notebook: boolean;
  rerun_kind: RerunKind;
  out_of_doc_order: boolean;
  stale_input_names: string[];
}

export type ConstraintViolation = 'reversed' | 'missing-upstream';

export type ViolatedConstraint = DependencyEdgeInfo & {
  violation: ConstraintViolation;
};

export interface DeletedCellRead {
  cell_id: string;
  names: string[];
  defined_names: string[];
  deleted_source_hash: string | null;
}

export interface SessionAnalysis {
  session_id: string;
  kernel_id: string | null;
  started_at: string;
  ended_reason: SessionEndReason;
  observed_order: string[];
  is_linear_extension: boolean;
  violated_constraints: ViolatedConstraint[];
  kendall_tau_vs_predicted: number | null;
  kendall_tau_vs_document: number | null;
  deleted_cell_reads: DeletedCellRead[];
  events: AnnotatedTraceEvent[];
}

export type ReproducibilityRiskKind =
  | 'deleted-cell-state'
  | 'out-of-order-binding'
  | 'edited-after-run'
  | 'ambiguous-binding-divergence'
  | 'opaque-effects'
  | 'never-executed';

export interface ReproducibilityRisk {
  kind: ReproducibilityRiskKind;
  cell_id: string | null;
  detail: string;
}

export interface ReproducibilityReport {
  run_all_reproduces: boolean;
  risks: ReproducibilityRisk[];
}

export type CellFreshness = 'fresh' | 'edited' | 'unexecuted' | 'stale-input';

export interface TraceCellGraph {
  cells: CellGraphNodeInfo[];
  edges: DependencyEdgeInfo[];
  minimal_constraints: DependencyEdgeInfo[];
  ambiguities: DependencyEdgeInfo[];
}

export interface TraceAnalysisData {
  predicted_order: string[];
  cell_graph: TraceCellGraph;
  sessions: SessionAnalysis[];
  current_session: SessionAnalysis | null;
  freshness: Record<string, CellFreshness>;
  replay_sets: Record<string, string[]>;
  reproducibility: ReproducibilityReport;
}

export interface LiveWarning {
  kind: 'stale-input' | 'out-of-order';
  cell_id: string;
  detail: string;
}

export const FRESHNESS_COLORS: Record<CellFreshness, string> = {
  fresh: '#22c55e',
  edited: '#f97316',
  unexecuted: '#94a3b8',
  'stale-input': '#f59e0b'
};
