import { ReactWidget } from '@jupyterlab/apputils';
import { NotebookPanel } from '@jupyterlab/notebook';
import React from 'react';
import EnvironmentComponent from './EnvironmentComponent';
import {
  clearHighlights,
  unregisterHighlightHoverHandler
} from '../../utils/CodeHighlighter';

export class EnvironmentWidget extends ReactWidget {
  private environmentData: any = null;
  private nbPanel: NotebookPanel | null = null;

  constructor() {
    super();
    this.addClass('jp-react-widget');
    this.id = 'systemx-environment-widget';
    this.title.label = 'Environment';
    this.title.closable = true;
  }

  render(): JSX.Element {
    return <EnvironmentComponent data={this.environmentData} />;
  }

  updateEnvironmentData(data: any): void {
    console.debug('Updating Environment telemetry via Lumino state.');

    this.environmentData = typeof data === 'string' ? JSON.parse(data) : data;

    this.update();
  }

  trackNotebook(panel: NotebookPanel | null): void {
    this.nbPanel = panel;
  }

  dispose(): void {
    clearHighlights(this.nbPanel);
    unregisterHighlightHoverHandler(this.nbPanel);
    super.dispose();
  }
}
