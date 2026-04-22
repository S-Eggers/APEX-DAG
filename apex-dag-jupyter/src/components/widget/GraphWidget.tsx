import { ReactWidget } from '@jupyterlab/apputils';
import { NotebookPanel } from '@jupyterlab/notebook';
import React from 'react';
import GraphComponent from './GraphComponent';
import { GraphMode } from '../../types/GraphTypes';
import { ExtractedCell } from '../../types/NotebookTypes';

export class GraphWidget extends ReactWidget {
  private mode: GraphMode;
  private graphData: any = { elements: [] };
  private resetTrigger: number = 0;
  private notebookName: string = 'untitled';
  private notebookCode: ExtractedCell[] = [];
  private nbPanel: NotebookPanel | null = null;

  constructor(mode: GraphMode = 'dataflow') {
    super();
    this.mode = mode;
    this.addClass('jp-react-widget');
  }

  setNotebookContext(
    name: string,
    code: ExtractedCell[],
    panel: NotebookPanel
  ) {
    this.notebookName = name;
    this.notebookCode = code;
    this.nbPanel = panel;
  }

  private handleLocateCell = (cellId: string) => {
    if (!this.nbPanel || !this.nbPanel.content) {
      console.warn('Notebook panel is not available.');
      return;
    }

    const notebook = this.nbPanel.content;
    const cellIndex = notebook.widgets.findIndex(c => c.model.id === cellId);

    if (cellIndex !== -1) {
      notebook.activeCellIndex = cellIndex;

      const cellWidget = notebook.widgets[cellIndex];
      cellWidget.node.scrollIntoView({ behavior: 'smooth', block: 'center' });

      const originalBg = cellWidget.node.style.backgroundColor;
      cellWidget.node.style.backgroundColor = 'rgba(59, 130, 246, 0.1)'; // Tailwind blue-500 at 10%
      cellWidget.node.style.transition = 'background-color 0.5s ease-out';

      setTimeout(() => {
        cellWidget.node.style.backgroundColor = originalBg;
      }, 1500);
    } else {
      console.warn(`Cell ID ${cellId} not found in the current notebook.`);
    }
  };

  render(): JSX.Element {
    return (
      <GraphComponent
        graphData={this.graphData}
        mode={this.mode}
        resetTrigger={this.resetTrigger}
        notebookName={this.notebookName}
        notebookCode={this.notebookCode}
        onLocateCell={this.handleLocateCell}
      />
    );
  }

  updateGraphData(graphData: any): void {
    console.debug('Updating graph data via Lumino state');
    this.graphData = JSON.parse(graphData);
    this.update();
  }

  resetView(): void {
    console.debug('Resetting graph view');
    this.resetTrigger += 1;
    this.update();
  }
}
