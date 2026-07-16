import { ReactWidget } from '@jupyterlab/apputils';
import { NotebookPanel } from '@jupyterlab/notebook';
import { CommandRegistry } from '@lumino/commands';
import React from 'react';
import GraphComponent from './GraphComponent';
import {
  GraphMode,
  LabelingConfig,
  LineageTuple
} from '../../types/GraphTypes';
import { ExtractedCell } from '../../types/NotebookTypes';
import {
  applyHighlights,
  clearHighlights,
  registerHighlightClickHandler,
  unregisterHighlightClickHandler,
  registerHighlightHoverHandler,
  unregisterHighlightHoverHandler
} from '../../utils/CodeHighlighter';
import { extractHighlights } from '../../utils/extractHighlights';
import { loadHighlightResolvers } from '../../utils/loadTaxonomy';

export class GraphWidget extends ReactWidget {
  private mode: GraphMode;
  private graphData: { elements: unknown[] } = { elements: [] };
  private resetTrigger: number = 0;
  private notebookName: string = 'untitled';
  private notebookCode: ExtractedCell[] = [];
  private nbPanel: NotebookPanel | null = null;
  private commands?: CommandRegistry;

  private labelingConfig: LabelingConfig = {
    rawDatasetPath: 'raw_dataset',
    defaultFetchMode: 'unannotated',
    defaultFlagType: 'needs_review'
  };

  private nnBackend: string = 'hgt';
  private featurePreset: string = 'standard';
  private modelVariant: string = '';
  private explainFeatureImportance: boolean = false;
  private backendError: string | null = null;
  private _focusNodeFn: ((nodeId: string) => void) | null = null;
  private _tuples: LineageTuple[] = [];
  private _registerFocusNode = (fn: (nodeId: string) => void): void => {
    this._focusNodeFn = fn;
  };

  constructor(mode: GraphMode = 'dataflow', commands?: CommandRegistry) {
    super();
    this.mode = mode;
    this.addClass('jp-react-widget');
    this.commands = commands;
  }

  setLabelingConfig(config: LabelingConfig) {
    this.labelingConfig = config;
    this.update();
  }

  setNNBackend(backend: string) {
    this.nnBackend = backend;
    this.update();
  }

  setFeaturePreset(preset: string) {
    this.featurePreset = preset;
    this.update();
  }

  setModelVariant(variant: string) {
    this.modelVariant = variant;
    this.update();
  }

  setExplainFeatureImportance(explain: boolean) {
    this.explainFeatureImportance = explain;
    this.update();
  }

  setBackendError(error: string | null) {
    this.backendError = error;
    this.update();
  }

  setNotebookContext(
    name: string,
    code: ExtractedCell[],
    panel: NotebookPanel
  ) {
    this.notebookName = name;
    this.notebookCode = code;
    this.nbPanel = panel;
    registerHighlightClickHandler(panel, nodeId => this._focusNodeFn?.(nodeId));
    registerHighlightHoverHandler(panel);
    this._applyCodeHighlights();
  }

  dispose(): void {
    unregisterHighlightClickHandler(this.nbPanel);
    unregisterHighlightHoverHandler(this.nbPanel);
    clearHighlights(this.nbPanel);
    super.dispose();
  }

  restoreHighlights(): void {
    this._applyCodeHighlights();
  }

  private _applyCodeHighlights(elements?: unknown[]): void {
    const data = elements ? { elements } : this.graphData;
    const mode = this.mode;
    loadHighlightResolvers(mode)
      .then(resolvers => {
        if (this.isDisposed || this.mode !== mode) return;
        const targets = extractHighlights(
          data as { elements: unknown[] },
          mode,
          resolvers
        );
        applyHighlights(this.nbPanel, targets);
      })
      .catch((err: Error) =>
        console.error('Failed to apply code highlights', err.message)
      );
  }

  private _onRelabel = (elements: unknown[]): void => {
    this._applyCodeHighlights(elements);
  };

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
      cellWidget.node.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
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
        labelingConfig={this.labelingConfig}
        nnBackend={this.nnBackend}
        featurePreset={this.featurePreset}
        modelVariant={this.modelVariant}
        explainFeatureImportance={this.explainFeatureImportance}
        backendError={this.backendError}
        onLocateCell={this.handleLocateCell}
        onRelabel={this._onRelabel}
        onFocusNode={this._registerFocusNode}
        onVariantTrained={variantKey => {
          this.commands?.execute('systemx:set-model-variant', {
            variant: variantKey
          });
        }}
        tuples={this._tuples}
        onNextNotebook={notebookPath => {
          if (!this.commands) return;
          const attachOptions = this.nbPanel
            ? { mode: 'tab-after', ref: this.nbPanel.id }
            : { mode: 'split-left', ref: this.id };

          this.commands.execute('docmanager:open', {
            path: notebookPath,
            factory: 'Notebook',
            options: attachOptions
          });
        }}
      />
    );
  }

  updateGraphData(graphData: string): void {
    console.debug('Updating graph data via Lumino state');
    let parsed: { elements?: unknown[]; tuples?: LineageTuple[] };
    try {
      parsed = JSON.parse(graphData);
    } catch (err) {
      console.error('Failed to parse graph data', err);
      this.setBackendError('Backend returned malformed graph data.');
      return;
    }
    this.backendError = null;
    this.graphData = { elements: parsed.elements ?? [] };
    this._tuples = parsed.tuples ?? [];
    this.update();
    this._applyCodeHighlights();
  }

  resetView(): void {
    console.debug('Resetting graph view');
    this.resetTrigger += 1;
    this.update();
  }
}
