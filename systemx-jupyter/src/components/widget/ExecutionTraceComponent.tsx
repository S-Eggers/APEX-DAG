import React, { useState } from 'react';
import { useDarkMode } from '../../hooks/useDarkMode';
import { CellGraphNodeInfo } from '../../types/ExecutionStateTypes';
import {
  AnnotatedTraceEvent,
  CellFreshness,
  FRESHNESS_COLORS,
  LiveWarning,
  RerunKind,
  SessionAnalysis,
  TraceAnalysisData,
  ViolatedConstraint
} from '../../types/ExecutionTraceTypes';

interface ExecutionTraceComponentProps {
  data: TraceAnalysisData | null;
  liveWarnings: LiveWarning[];
  onLocateCell: (cellId: string) => void;
  onSave: () => void;
}

const FRESHNESS_LABELS: Record<CellFreshness, string> = {
  fresh: 'Fresh',
  edited: 'Edited',
  unexecuted: 'Not run',
  'stale-input': 'Stale input'
};

const RERUN_STYLES: Record<RerunKind, { label: string; className: string }> = {
  first: {
    label: 'first',
    className:
      'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800'
  },
  identical: {
    label: 'rerun',
    className:
      'bg-gray-50 dark:bg-gray-800 text-gray-500 dark:text-gray-400 border-gray-200 dark:border-gray-700'
  },
  edited: {
    label: 'edited + rerun',
    className:
      'bg-orange-50 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 border-orange-200 dark:border-orange-800'
  }
};

const SectionHeader = ({ children }: { children: React.ReactNode }) => (
  <h2 className="text-xs font-bold text-gray-800 dark:text-gray-200 uppercase tracking-widest mb-3 border-b border-gray-200 dark:border-gray-700 pb-1">
    {children}
  </h2>
);

const formatTau = (tau: number | null): string =>
  tau === null ? '-' : tau.toFixed(2);

const formatTime = (iso: string): string => {
  const date = new Date(iso);
  return isNaN(date.getTime()) ? iso : date.toLocaleTimeString();
};

const EventRow = ({
  event,
  node,
  isViolationEndpoint,
  onLocate
}: {
  event: AnnotatedTraceEvent;
  node: CellGraphNodeInfo | undefined;
  isViolationEndpoint: boolean;
  onLocate: () => void;
}) => {
  const rerun = RERUN_STYLES[event.rerun_kind];
  return (
    <div
      className={`flex items-center gap-2 text-[11px] py-1 px-1.5 rounded ${
        event.in_notebook
          ? 'cursor-pointer hover:bg-blue-50 dark:hover:bg-blue-900/20'
          : 'opacity-60'
      }`}
      onClick={event.in_notebook ? onLocate : undefined}
      title={
        event.in_notebook
          ? 'Click to locate this cell in the notebook'
          : 'This cell was deleted after it ran'
      }
    >
      <span className="font-mono font-bold text-gray-400 dark:text-gray-500 w-6 text-right">
        {event.seq + 1}.
      </span>
      <span className="font-mono text-gray-500 dark:text-gray-400 w-8">
        [{event.execution_count ?? ' '}]
      </span>
      <code className="flex-1 truncate text-gray-800 dark:text-gray-200">
        {event.in_notebook
          ? node?.source_preview || `cell #${event.document_index + 1}`
          : `(deleted cell ${event.cell_id.slice(0, 6)})`}
      </code>
      <span
        className={`px-1.5 py-0.5 rounded text-[10px] font-semibold border ${rerun.className}`}
      >
        {rerun.label}
      </span>
      {event.success === false && (
        <span className="text-red-500 font-bold" title="Execution raised an error">
          ✗
        </span>
      )}
      {event.out_of_doc_order && (
        <span
          className="text-red-500 font-bold"
          title="Executed above a cell that had already run (out of document order)"
        >
          ⤺
        </span>
      )}
      {event.stale_input_names.length > 0 && (
        <span
          className="text-amber-500 font-bold"
          title={`Inputs not produced yet at this point: ${event.stale_input_names.join(', ')}`}
        >
          ⚠
        </span>
      )}
      {isViolationEndpoint && (
        <span
          className="w-2 h-2 rounded-full bg-red-500 shrink-0"
          title="Part of a violated dependency constraint this session"
        />
      )}
    </div>
  );
};

const ViolationRow = ({
  edge,
  cellLabel,
  onLocateCell
}: {
  edge: ViolatedConstraint;
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
    <span className="text-red-500">→</span>
    <button
      className="text-blue-600 dark:text-blue-400 hover:underline bg-transparent border-none cursor-pointer p-0"
      onClick={() => onLocateCell(edge.dst_cell)}
    >
      {cellLabel(edge.dst_cell)}
    </button>
    <span className="text-gray-600 dark:text-gray-300">{edge.name}</span>
    <span className="px-1 rounded text-[10px] font-semibold bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-300">
      {edge.violation === 'reversed'
        ? 'producer ran after reader'
        : 'producer never ran'}
    </span>
  </div>
);

const SessionSection = ({
  session,
  isCurrent,
  nodeById,
  cellLabel,
  onLocateCell
}: {
  session: SessionAnalysis;
  isCurrent: boolean;
  nodeById: Map<string, CellGraphNodeInfo>;
  cellLabel: (cellId: string) => string;
  onLocateCell: (cellId: string) => void;
}) => {
  const [expanded, setExpanded] = useState(isCurrent);
  const violationEndpoints = new Set(
    session.violated_constraints.flatMap(v => [v.src_cell, v.dst_cell])
  );

  return (
    <div className="bg-white dark:bg-[#21262d] border border-gray-200 dark:border-gray-700 rounded shadow-sm">
      <div
        className="flex items-center gap-2 p-2 cursor-pointer select-none"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="text-gray-400 text-[10px] w-3">
          {expanded ? '▼' : '▶'}
        </span>
        <span className="text-[11px] font-semibold text-gray-700 dark:text-gray-300">
          {isCurrent ? 'Current session' : `Session ${formatTime(session.started_at)}`}
        </span>
        <span className="text-[10px] text-gray-400">
          {session.events.length} run{session.events.length !== 1 ? 's' : ''}
          {session.ended_reason ? ` · ended: ${session.ended_reason}` : ''}
        </span>
        <span className="flex-1" />
        <span
          className={`px-1.5 py-0.5 rounded text-[10px] font-semibold border ${
            session.is_linear_extension
              ? 'bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300 border-green-200 dark:border-green-800'
              : 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300 border-red-200 dark:border-red-800'
          }`}
          title="Whether the observed order respects every dependency constraint"
        >
          {session.is_linear_extension ? 'order valid' : 'order violated'}
        </span>
        <span
          className="text-[10px] font-mono text-gray-500 dark:text-gray-400"
          title="Kendall tau of the observed order vs. the predicted safe order / vs. document order (1 = identical, -1 = reversed)"
        >
          τ {formatTau(session.kendall_tau_vs_predicted)} /{' '}
          {formatTau(session.kendall_tau_vs_document)}
        </span>
      </div>

      {expanded && (
        <div className="px-2 pb-2 flex flex-col gap-2">
          {session.events.length === 0 ? (
            <span className="text-[11px] text-gray-400 italic px-1">
              No executions recorded in this session yet.
            </span>
          ) : (
            <div className="flex flex-col">
              {session.events.map(event => (
                <EventRow
                  key={event.seq}
                  event={event}
                  node={nodeById.get(event.cell_id)}
                  isViolationEndpoint={violationEndpoints.has(event.cell_id)}
                  onLocate={() => onLocateCell(event.cell_id)}
                />
              ))}
            </div>
          )}

          {session.violated_constraints.length > 0 && (
            <div className="border-t border-gray-100 dark:border-gray-800 pt-1.5">
              <span className="text-[10px] font-bold uppercase tracking-wider text-red-500">
                Violated constraints
              </span>
              {session.violated_constraints.map((edge, i) => (
                <ViolationRow
                  key={i}
                  edge={edge}
                  cellLabel={cellLabel}
                  onLocateCell={onLocateCell}
                />
              ))}
            </div>
          )}

          {session.deleted_cell_reads.length > 0 && (
            <div className="border-t border-gray-100 dark:border-gray-800 pt-1.5">
              <span className="text-[10px] font-bold uppercase tracking-wider text-violet-500">
                Deleted-cell state
              </span>
              {session.deleted_cell_reads.map(read => (
                <div
                  key={read.cell_id}
                  className="text-[11px] text-gray-600 dark:text-gray-300 py-0.5"
                >
                  A deleted cell defined{' '}
                  <code className="text-violet-600 dark:text-violet-400">
                    {(read.names.length > 0
                      ? read.names
                      : read.defined_names
                    ).join(', ') || '(unknown names)'}
                  </code>
                  {read.names.length > 0
                    ? ' - still consumed by current cells.'
                    : '.'}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const ExecutionTraceComponent: React.FC<ExecutionTraceComponentProps> = ({
  data,
  liveWarnings,
  onLocateCell,
  onSave
}) => {
  const isDark = useDarkMode();
  const [replayTarget, setReplayTarget] = useState<string | null>(null);

  if (!data) {
    return (
      <div
        className={`${isDark ? 'dark ' : ''}flex h-full w-full items-center justify-center bg-[#eef2f8] dark:bg-[#0d1117] text-gray-500 dark:text-gray-400 italic text-sm`}
      >
        No execution trace yet. Run a cell to start recording.
      </div>
    );
  }

  const nodeById = new Map(data.cell_graph.cells.map(c => [c.cell_id, c]));
  const cellLabel = (cellId: string): string => {
    const node = nodeById.get(cellId);
    return node ? `#${node.document_index + 1}` : cellId.slice(0, 6);
  };

  const repro = data.reproducibility;
  const orderedSessions = [...data.sessions].reverse(); // newest first
  const replaySet = replayTarget ? (data.replay_sets[replayTarget] ?? []) : [];

  return (
    <div
      className={`${isDark ? 'dark ' : ''}h-full w-full box-border overflow-auto bg-[#eef2f8] dark:bg-[#0d1117] p-4 flex flex-col gap-5`}
    >
      {/* Reproducibility banner */}
      <div
        className={`rounded border p-2.5 ${
          repro.run_all_reproduces
            ? 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-800'
            : 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-800'
        }`}
      >
        <div className="flex items-center gap-2">
          <span
            className={`text-[12px] font-bold ${
              repro.run_all_reproduces
                ? 'text-green-700 dark:text-green-300'
                : 'text-red-700 dark:text-red-300'
            }`}
          >
            {repro.run_all_reproduces
              ? '✓ Restart & Run All reproduces this state'
              : '✗ Restart & Run All would NOT reproduce this state'}
          </span>
          <span className="flex-1" />
          <button
            className="px-2 py-1 rounded text-[11px] font-semibold border bg-white dark:bg-[#21262d] text-gray-700 dark:text-gray-300 border-gray-300 dark:border-gray-600 cursor-pointer hover:border-blue-400"
            onClick={onSave}
            title="Persist the recorded trace as a sidecar JSON on the server"
          >
            Save trace
          </button>
        </div>
        {repro.risks.length > 0 && (
          <ul className="mt-1.5 text-[11px] text-red-700 dark:text-red-300 list-disc list-inside">
            {repro.risks.map((risk, i) => (
              <li key={i}>
                {risk.cell_id && nodeById.has(risk.cell_id) ? (
                  <button
                    className="text-blue-600 dark:text-blue-400 hover:underline bg-transparent border-none cursor-pointer p-0 font-mono"
                    onClick={() => onLocateCell(risk.cell_id!)}
                  >
                    {cellLabel(risk.cell_id)}
                  </button>
                ) : null}{' '}
                {risk.detail}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Instant warnings from the recorder (pre-analysis) */}
      {liveWarnings.length > 0 && (
        <div className="rounded border p-2 bg-amber-50 dark:bg-amber-900/30 border-amber-200 dark:border-amber-800">
          {liveWarnings.map((warning, i) => (
            <div
              key={i}
              className="text-[11px] text-amber-700 dark:text-amber-300"
            >
              ⚠{' '}
              <button
                className="text-blue-600 dark:text-blue-400 hover:underline bg-transparent border-none cursor-pointer p-0 font-mono"
                onClick={() => onLocateCell(warning.cell_id)}
              >
                {cellLabel(warning.cell_id)}
              </button>{' '}
              {warning.detail}
            </div>
          ))}
        </div>
      )}

      {/* Per-cell freshness + replay sets */}
      <div>
        <SectionHeader>Cell freshness</SectionHeader>
        <div className="flex flex-wrap gap-1.5">
          {data.cell_graph.cells.map(node => {
            const freshness = data.freshness[node.cell_id] ?? 'unexecuted';
            const selected = replayTarget === node.cell_id;
            return (
              <button
                key={node.cell_id}
                className={`px-2 py-1 rounded text-[11px] font-semibold border cursor-pointer bg-white dark:bg-[#21262d] text-gray-700 dark:text-gray-300 ${
                  selected
                    ? 'border-blue-500'
                    : 'border-gray-200 dark:border-gray-700 hover:border-blue-400'
                }`}
                style={{ borderLeft: `4px solid ${FRESHNESS_COLORS[freshness]}` }}
                onClick={() =>
                  setReplayTarget(selected ? null : node.cell_id)
                }
                onDoubleClick={() => onLocateCell(node.cell_id)}
                title={`${FRESHNESS_LABELS[freshness]} - click for the minimal replay set, double-click to locate`}
              >
                {cellLabel(node.cell_id)}{' '}
                <span className="font-normal text-[10px] text-gray-400">
                  {FRESHNESS_LABELS[freshness]}
                </span>
              </button>
            );
          })}
        </div>
        {replayTarget && (
          <div className="mt-2 text-[11px] bg-white dark:bg-[#21262d] border border-gray-200 dark:border-gray-700 rounded shadow-sm p-2 text-gray-700 dark:text-gray-300">
            {replaySet.length === 0 ? (
              <>
                <code>{cellLabel(replayTarget)}</code> and everything it depends
                on is fresh - nothing to re-run.
              </>
            ) : (
              <>
                Minimal replay for <code>{cellLabel(replayTarget)}</code>, in
                order:{' '}
                {replaySet.map((cellId, i) => (
                  <React.Fragment key={cellId}>
                    {i > 0 && <span className="text-gray-400"> → </span>}
                    <button
                      className="text-blue-600 dark:text-blue-400 hover:underline bg-transparent border-none cursor-pointer p-0 font-mono"
                      onClick={() => onLocateCell(cellId)}
                    >
                      {cellLabel(cellId)}
                    </button>
                  </React.Fragment>
                ))}
              </>
            )}
          </div>
        )}
      </div>

      {/* Session timeline */}
      <div>
        <SectionHeader>
          Recorded sessions ({orderedSessions.length})
        </SectionHeader>
        {orderedSessions.length === 0 ? (
          <span className="text-[11px] text-gray-400 italic">
            No executions recorded yet - run a cell.
          </span>
        ) : (
          <div className="flex flex-col gap-2">
            {orderedSessions.map((session, i) => (
              <SessionSection
                key={session.session_id}
                session={session}
                isCurrent={i === 0}
                nodeById={nodeById}
                cellLabel={cellLabel}
                onLocateCell={onLocateCell}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ExecutionTraceComponent;
