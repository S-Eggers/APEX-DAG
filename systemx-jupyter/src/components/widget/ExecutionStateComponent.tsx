import React from 'react';
import { useDarkMode } from '../../hooks/useDarkMode';
import {
  CellGraphNodeInfo,
  CellStateInfo,
  CellStatus,
  DependencyEdgeInfo,
  ExecutionStateData,
  STATUS_COLORS
} from '../../types/ExecutionStateTypes';

interface ExecutionStateComponentProps {
  data: ExecutionStateData | null;
  onLocateCell: (cellId: string) => void;
}

const STATUS_LABELS: Record<CellStatus, string> = {
  fresh: 'Fresh',
  stale: 'Stale',
  dirty: 'Edited',
  unexecuted: 'Not run',
  'missing-deps': 'Missing deps',
  unknown: 'Unknown'
};

const SectionHeader = ({ children }: { children: React.ReactNode }) => (
  <h2 className="text-xs font-bold text-gray-800 dark:text-gray-200 uppercase tracking-widest mb-3 border-b border-gray-200 dark:border-gray-700 pb-1">
    {children}
  </h2>
);

const StatusBadge = ({ status }: { status: CellStatus }) => (
  <span
    className="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider text-white"
    style={{ backgroundColor: STATUS_COLORS[status] }}
  >
    {STATUS_LABELS[status]}
  </span>
);

const FlagPill = ({
  ok,
  labelOk,
  labelBad
}: {
  ok: boolean;
  labelOk: string;
  labelBad: string;
}) => (
  <span
    className={`px-2 py-1 rounded text-[11px] font-semibold border ${
      ok
        ? 'bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300 border-green-200 dark:border-green-800'
        : 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300 border-red-200 dark:border-red-800'
    }`}
  >
    {ok ? `✓ ${labelOk}` : `✗ ${labelBad}`}
  </span>
);

const CellCard = ({
  rank,
  node,
  state,
  onLocate
}: {
  rank: number;
  node: CellGraphNodeInfo | undefined;
  state: CellStateInfo;
  onLocate: () => void;
}) => (
  <div
    className="bg-white dark:bg-[#21262d] border border-gray-200 dark:border-gray-700 rounded shadow-sm p-2 flex flex-col gap-1 cursor-pointer hover:border-blue-400 dark:hover:border-blue-500 transition-colors"
    style={{ borderLeft: `4px solid ${STATUS_COLORS[state.status]}` }}
    onClick={onLocate}
    title="Click to locate this cell in the notebook"
  >
    <div className="flex items-center gap-2">
      <span className="text-[11px] font-mono font-bold text-gray-400 dark:text-gray-500 w-6 text-right">
        {rank + 1}.
      </span>
      <span className="text-[11px] font-mono text-gray-500 dark:text-gray-400">
        [{node?.execution_count ?? ' '}]
      </span>
      <code className="flex-1 truncate text-[11px] text-gray-800 dark:text-gray-200">
        {node?.source_preview || state.cell_id}
      </code>
      <StatusBadge status={state.status} />
      {state.unsafe_reorder && (
        <span
          className="text-red-500 text-[11px] font-bold"
          title="Depends on a cell further down the notebook"
        >
          ↺
        </span>
      )}
      {state.hidden_state_risk && (
        <span
          className="text-violet-500 text-[11px] font-bold"
          title="Reads a name no current cell defines (hidden state)"
        >
          ⚠
        </span>
      )}
    </div>
    {state.reasons.length > 0 && (
      <ul className="ml-8 text-[10px] text-gray-500 dark:text-gray-400 list-disc list-inside">
        {state.reasons.map((reason, i) => (
          <li key={i}>{reason}</li>
        ))}
      </ul>
    )}
  </div>
);

const ConstraintRow = ({
  edge,
  cellLabel,
  onLocateCell
}: {
  edge: DependencyEdgeInfo;
  cellLabel: (cellId: string) => string;
  onLocateCell: (cellId: string) => void;
}) => (
  <div className="flex items-center gap-1.5 text-[11px] font-mono py-0.5">
    <button
      className="text-blue-600 dark:text-blue-400 hover:underline bg-transparent border-none cursor-pointer p-0"
      onClick={() => onLocateCell(edge.src_cell)}
    >
      {cellLabel(edge.src_cell)}
    </button>
    <span
      className={
        edge.ambiguous
          ? 'text-amber-500 border-b border-dashed border-amber-500'
          : 'text-gray-400'
      }
      title={
        edge.ambiguous
          ? `Ambiguous: ${edge.candidate_def_cells.length} cells define \`${edge.name}\``
          : undefined
      }
    >
      {edge.out_of_order ? '⤺' : '→'}
    </span>
    <button
      className="text-blue-600 dark:text-blue-400 hover:underline bg-transparent border-none cursor-pointer p-0"
      onClick={() => onLocateCell(edge.dst_cell)}
    >
      {cellLabel(edge.dst_cell)}
    </button>
    <span className="text-gray-600 dark:text-gray-300">{edge.name}</span>
    <span className="text-gray-400 dark:text-gray-500 text-[10px]">
      {edge.kind !== 'DATA' ? edge.kind.toLowerCase().replace('_', ' ') : ''}
    </span>
  </div>
);

const ExecutionStateComponent: React.FC<ExecutionStateComponentProps> = ({
  data,
  onLocateCell
}) => {
  const isDark = useDarkMode();

  if (!data) {
    return (
      <div
        className={`${isDark ? 'dark ' : ''}flex h-full w-full items-center justify-center bg-[#eef2f8] dark:bg-[#0d1117] text-gray-500 dark:text-gray-400 italic text-sm`}
      >
        No execution-state analysis yet. Open a notebook with code cells.
      </div>
    );
  }

  const nodeById = new Map(data.cell_graph.cells.map(c => [c.cell_id, c]));
  const cellLabel = (cellId: string): string => {
    const node = nodeById.get(cellId);
    return node ? `#${node.document_index + 1}` : cellId.slice(0, 6);
  };
  const flags = data.notebook_flags;
  const staleCount = Object.values(data.cell_states).filter(s =>
    ['stale', 'dirty'].includes(s.status)
  ).length;

  return (
    <div
      className={`${isDark ? 'dark ' : ''}h-full w-full box-border overflow-auto bg-[#eef2f8] dark:bg-[#0d1117] p-4 flex flex-col gap-5`}
    >
      <div className="flex flex-wrap gap-2">
        <FlagPill
          ok={flags.safe_to_run_top_to_bottom}
          labelOk="Safe to run top-to-bottom"
          labelBad="NOT reproducible top-to-bottom"
        />
        <FlagPill
          ok={!flags.hidden_state_risk}
          labelOk="No hidden state detected"
          labelBad="Hidden state risk"
        />
        {staleCount > 0 && (
          <span className="px-2 py-1 rounded text-[11px] font-semibold border bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800">
            {staleCount} stale/edited cell{staleCount > 1 ? 's' : ''}
          </span>
        )}
      </div>

      <div>
        <SectionHeader>Predicted execution order</SectionHeader>
        <div className="flex flex-col gap-1.5">
          {data.predicted_order.map((cellId, rank) => {
            const state = data.cell_states[cellId];
            if (!state) return null;
            return (
              <CellCard
                key={cellId}
                rank={rank}
                node={nodeById.get(cellId)}
                state={state}
                onLocate={() => onLocateCell(cellId)}
              />
            );
          })}
        </div>
      </div>

      <div>
        <SectionHeader>
          Ordering constraints ({data.constraints.length})
        </SectionHeader>
        <div className="bg-white dark:bg-[#21262d] border border-gray-200 dark:border-gray-700 rounded shadow-sm p-2">
          {data.constraints.length === 0 ? (
            <span className="text-[11px] text-gray-400 italic">
              No inter-cell dependencies found.
            </span>
          ) : (
            data.constraints.map((edge, i) => (
              <ConstraintRow
                key={i}
                edge={edge}
                cellLabel={cellLabel}
                onLocateCell={onLocateCell}
              />
            ))
          )}
        </div>
      </div>

      {data.ambiguities.length > 0 && (
        <div>
          <SectionHeader>
            Ambiguous bindings ({data.ambiguities.length})
          </SectionHeader>
          <p className="text-[10px] text-gray-500 dark:text-gray-400 mb-2">
            These names are defined in several cells - which definition a
            reader sees depends on the actual execution order.
          </p>
          <div className="bg-white dark:bg-[#21262d] border border-gray-200 dark:border-gray-700 rounded shadow-sm p-2">
            {data.ambiguities.map((edge, i) => (
              <ConstraintRow
                key={i}
                edge={edge}
                cellLabel={cellLabel}
                onLocateCell={onLocateCell}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ExecutionStateComponent;
