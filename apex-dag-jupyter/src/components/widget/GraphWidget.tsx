import { ReactWidget } from '@jupyterlab/apputils';
import React from 'react';
import GraphComponent from './GraphComponent';
import { GraphMode } from '../../types/GraphTypes';

export class GraphWidget extends ReactWidget {
  private mode: GraphMode;
  private graphData: any = { elements: [] };
  private resetTrigger: number = 0;

  constructor(mode: GraphMode = 'dataflow') {
    super();
    this.mode = mode;
    this.addClass('jp-react-widget');
  }

  render(): JSX.Element {
    return (
      <GraphComponent
        graphData={this.graphData}
        mode={this.mode}
        resetTrigger={this.resetTrigger}
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
