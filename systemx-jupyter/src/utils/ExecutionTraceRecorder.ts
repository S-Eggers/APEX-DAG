import { ISessionContext } from '@jupyterlab/apputils';
import { Cell, ICodeCellModel } from '@jupyterlab/cells';
import { NotebookPanel } from '@jupyterlab/notebook';
import { Signal } from '@lumino/signaling';

import { AppSettings } from '../settings/AppSettings';
import {
  LiveWarning,
  NotebookTrace,
  SessionEndReason,
  TraceCellGraph,
  TraceEvent,
  TraceSession
} from '../types/ExecutionTraceTypes';
import callBackend from './callBackend';
import { getDirtyState, hashSource } from './ExecutionTracker';
import { getNotebookCode } from './getNotebookCode';

export interface TraceChange {
  path: string;
  event: TraceEvent | null;
  warnings: LiveWarning[];
}

class ExecutionTraceRecorder {
  readonly changed = new Signal<this, TraceChange>(this);

  private traces = new Map<string, NotebookTrace>();
  private graphs = new Map<string, TraceCellGraph>();
  private panels = new Map<string, NotebookPanel>();
  private attachedPanels = new WeakSet<NotebookPanel>();
  private autosaveTimers = new Map<string, ReturnType<typeof setTimeout>>();
  private settings: AppSettings | null = null;

  attach(panel: NotebookPanel, settings: AppSettings): void {
    this.settings = settings;
    if (this.attachedPanels.has(panel)) return;
    this.attachedPanels.add(panel);

    const path = panel.context.path;
    this.panels.set(path, panel);
    this.getTrace(path); // ensure the trace + first session exist

    panel.sessionContext.statusChanged.connect((ctx, status) => {
      if (status === 'restarting' || status === 'autorestarting') {
        this.beginSession(path, this.kernelId(ctx), 'restart');
      }
    });

    panel.sessionContext.kernelChanged.connect((ctx, args) => {
      const newId = args.newValue?.id ?? null;
      const open = this.openSession(path);
      if (open && open.events.length === 0) {
        open.kernel_id = newId;
        return;
      }
      if (open && open.kernel_id !== newId) {
        this.beginSession(path, newId, 'kernel-changed');
      }
    });

    panel.disposed.connect(() => {
      this.panels.delete(path);
      const timer = this.autosaveTimers.get(path);
      if (timer) clearTimeout(timer);
      this.autosaveTimers.delete(path);
      if (this.settings?.traceEnabled && this.settings.traceAutosave !== 'off') {
        void this.save(path);
      }
    });
  }

  record(cell: Cell | undefined, panel: NotebookPanel, success?: boolean): void {
    if (!this.settings?.traceEnabled) return;
    const model = cell?.model;
    if (!model || model.type !== 'code') return;

    const path = panel.context.path;
    const trace = this.getTrace(path);
    const source = model.sharedModel.getSource();
    const executionCount = (model as ICodeCellModel).executionCount ?? null;

    let session = this.openSession(path) ?? this.beginSession(path, this.kernelSessionId(panel), null);

    const maxCount = Math.max(
      0,
      ...session.events.map(e => e.execution_count ?? 0)
    );
    if (executionCount === 1 && maxCount > 1) {
      session = this.beginSession(path, this.kernelSessionId(panel), 'restart');
    }

    const event: TraceEvent = {
      seq: session.events.length > 0 ? session.events[session.events.length - 1].seq + 1 : 0,
      cell_id: model.id,
      document_index: panel.content.widgets.findIndex(w => w.model.id === model.id),
      execution_count: executionCount,
      source_hash: hashSource(source),
      timestamp: new Date().toISOString(),
      success: success ?? null
    };
    session.events.push(event);
    trace.sources[event.source_hash] = source;

    this.scheduleAutosave(path);
    this.changed.emit({ path, event, warnings: this.liveWarnings(event, session, panel) });
  }

  setCellGraph(path: string, graph: TraceCellGraph): void {
    this.graphs.set(path, graph);
  }

  getTrace(path: string): NotebookTrace {
    let trace = this.traces.get(path);
    if (!trace) {
      trace = {
        schema_version: 1,
        notebook_path: path,
        saved_at: '',
        sessions: [],
        sources: {},
        cells_snapshot: []
      };
      this.traces.set(path, trace);
    }
    const cells = this.panels.get(path)?.content.model?.cells;
    if (cells) {
      trace.cells_snapshot = getNotebookCode(cells, true);
    }
    return trace;
  }

  hydrate(path: string, saved: NotebookTrace | null): void {
    if (!saved || !Array.isArray(saved.sessions)) return;
    const trace = this.getTrace(path);
    const liveIds = new Set(trace.sessions.map(s => s.session_id));

    const restored = saved.sessions
      .filter(s => s?.session_id && !liveIds.has(s.session_id))
      .map(s => ({
        ...s,
        ended_reason: (s.ended_reason ?? 'reload') as SessionEndReason
      }));
    trace.sessions = [...restored, ...trace.sessions];
    trace.sources = { ...(saved.sources ?? {}), ...trace.sources };
    this.changed.emit({ path, event: null, warnings: [] });
  }

  save(path: string): Promise<void> {
    const settings = this.settings;
    if (!settings) return Promise.resolve();
    const trace = this.getTrace(path);
    trace.saved_at = new Date().toISOString();
    return callBackend<{ success: boolean; message?: string }>(
      'execution-trace/save',
      {
        filename: path,
        base_path: settings.traceStoragePath,
        trace
      }
    )
      .then(response => {
        if (!response.success) {
          console.warn('[EXECUTION-TRACE] Save rejected:', response.message);
        }
      })
      .catch(error => {
        console.warn('[EXECUTION-TRACE] Save failed:', error);
      });
  }

  private beginSession(
    path: string,
    kernelId: string | null,
    endPreviousAs: SessionEndReason
  ): TraceSession {
    const trace = this.getTrace(path);
    const open = this.openSession(path);
    if (open) {
      open.ended_reason = endPreviousAs ?? 'shutdown';
      if (this.settings?.traceEnabled && this.settings.traceAutosave === 'session-end') {
        void this.save(path);
      }
    }
    const session: TraceSession = {
      session_id: `${kernelId ?? 'nokernel'}:${Date.now()}`,
      kernel_id: kernelId,
      started_at: new Date().toISOString(),
      ended_reason: null,
      events: []
    };
    trace.sessions.push(session);
    return session;
  }

  private openSession(path: string): TraceSession | null {
    const sessions = this.traces.get(path)?.sessions ?? [];
    const last = sessions[sessions.length - 1];
    return last && last.ended_reason === null ? last : null;
  }

  private kernelId(ctx: ISessionContext): string | null {
    return ctx.session?.kernel?.id ?? null;
  }

  private kernelSessionId(panel: NotebookPanel): string | null {
    return this.kernelId(panel.sessionContext);
  }

  private liveWarnings(
    event: TraceEvent,
    session: TraceSession,
    panel: NotebookPanel
  ): LiveWarning[] {
    const warnings: LiveWarning[] = [];
    const graph = this.graphs.get(panel.context.path);
    if (!graph) return warnings;

    const executed = new Set(session.events.map(e => e.cell_id));
    for (const edge of graph.edges) {
      if (edge.dst_cell !== event.cell_id) continue;
      if (!executed.has(edge.src_cell)) {
        warnings.push({
          kind: 'stale-input',
          cell_id: event.cell_id,
          detail: `\`${edge.name}\` comes from a cell that has not run this session`
        });
        continue;
      }
      const srcWidget = panel.content.widgets.find(
        w => w.model.id === edge.src_cell
      );
      if (
        srcWidget?.model.type === 'code' &&
        getDirtyState(srcWidget.model as ICodeCellModel) === true
      ) {
        warnings.push({
          kind: 'stale-input',
          cell_id: event.cell_id,
          detail: `\`${edge.name}\` comes from a cell edited after its last run`
        });
      }
    }

    const maxPriorIndex = Math.max(
      -1,
      ...session.events
        .filter(e => e.seq < event.seq)
        .map(e => e.document_index)
    );
    if (event.document_index >= 0 && event.document_index < maxPriorIndex) {
      warnings.push({
        kind: 'out-of-order',
        cell_id: event.cell_id,
        detail: 'executed above a cell that already ran (out of document order)'
      });
    }
    return warnings;
  }

  private scheduleAutosave(path: string): void {
    const settings = this.settings;
    if (!settings || settings.traceAutosave !== 'debounced') return;
    const pending = this.autosaveTimers.get(path);
    if (pending) clearTimeout(pending);
    this.autosaveTimers.set(
      path,
      setTimeout(() => {
        this.autosaveTimers.delete(path);
        void this.save(path);
      }, settings.traceAutosaveDelay)
    );
  }
}

export const traceRecorder = new ExecutionTraceRecorder();
