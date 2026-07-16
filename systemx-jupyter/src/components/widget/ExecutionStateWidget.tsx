import { ReactWidget } from '@jupyterlab/apputils';
import { NotebookPanel } from '@jupyterlab/notebook';
import React from 'react';
import ExecutionStateComponent from './ExecutionStateComponent';
import { ExecutionStateData } from '../../types/ExecutionStateTypes';
import {
  clearHighlights,
  unregisterHighlightHoverHandler
} from '../../utils/CodeHighlighter';

export class ExecutionStateWidget extends ReactWidget {
  private executionState: ExecutionStateData | null = null;
  private nbPanel: NotebookPanel | null = null;

  constructor() {
    super();
    this.addClass('jp-react-widget');
    this.id = 'systemx-execution-state-widget';
    this.title.label = 'Execution State';
    this.title.closable = true;
  }

  render(): JSX.Element {
    return (
      <ExecutionStateComponent
        data={this.executionState}
        onLocateCell={this.handleLocateCell}
      />
    );
  }

  updateExecutionState(data: ExecutionStateData): void {
    this.executionState = data;
    this.update();
  }

  trackNotebook(panel: NotebookPanel | null): void {
    this.nbPanel = panel;
  }

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
    clearHighlights(this.nbPanel);
    unregisterHighlightHoverHandler(this.nbPanel);
    super.dispose();
  }
}
