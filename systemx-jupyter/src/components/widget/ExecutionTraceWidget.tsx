import { ReactWidget } from '@jupyterlab/apputils';
import { NotebookPanel } from '@jupyterlab/notebook';
import React from 'react';
import ExecutionTraceComponent from './ExecutionTraceComponent';
import { LiveWarning, TraceAnalysisData } from '../../types/ExecutionTraceTypes';
import { traceRecorder, TraceChange } from '../../utils/ExecutionTraceRecorder';

const MAX_LIVE_WARNINGS = 6;

export class ExecutionTraceWidget extends ReactWidget {
  private analysis: TraceAnalysisData | null = null;
  private nbPanel: NotebookPanel | null = null;
  private liveWarnings: LiveWarning[] = [];

  constructor() {
    super();
    this.addClass('jp-react-widget');
    this.id = 'systemx-execution-trace-widget';
    this.title.label = 'Execution Trace';
    this.title.closable = true;
    traceRecorder.changed.connect(this.onTraceChanged, this);
  }

  render(): JSX.Element {
    return (
      <ExecutionTraceComponent
        data={this.analysis}
        liveWarnings={this.liveWarnings}
        onLocateCell={this.handleLocateCell}
        onSave={this.handleSave}
      />
    );
  }

  updateAnalysis(data: TraceAnalysisData): void {
    this.analysis = data;
    this.liveWarnings = [];
    this.update();
  }

  trackNotebook(panel: NotebookPanel | null): void {
    this.nbPanel = panel;
  }

  private onTraceChanged = (_: unknown, change: TraceChange): void => {
    if (change.path !== this.nbPanel?.context.path) return;
    if (change.warnings.length > 0) {
      this.liveWarnings = [...this.liveWarnings, ...change.warnings].slice(
        -MAX_LIVE_WARNINGS
      );
      this.update();
    }
  };

  private handleSave = (): void => {
    const path = this.nbPanel?.context.path;
    if (path) void traceRecorder.save(path);
  };

  private handleLocateCell = (cellId: string): void => {
    const notebook = this.nbPanel?.content;
    if (!notebook) return;

    const cellIndex = notebook.widgets.findIndex(c => c.model.id === cellId);
    if (cellIndex === -1) return;

    notebook.activeCellIndex = cellIndex;
    const cellWidget = notebook.widgets[cellIndex];
    cellWidget.node.scrollIntoView({ behavior: 'smooth', block: 'center' });

    const originalBg = cellWidget.node.style.backgroundColor;
    cellWidget.node.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
    cellWidget.node.style.transition = 'background-color 0.5s ease-out';
    setTimeout(() => {
      cellWidget.node.style.backgroundColor = originalBg;
    }, 1500);
  };

  dispose(): void {
    traceRecorder.changed.disconnect(this.onTraceChanged, this);
    super.dispose();
  }
}
