import { ReactWidget } from '@jupyterlab/apputils';
import React from 'react';
import GraphComponent from './GraphComponent';

export class GraphWidget extends ReactWidget {
  private mode: 'dataflow' | 'lineage';
  private graphData: any = { elements: [] };
  private resetTrigger: number = 0;

  constructor(mode: 'dataflow' | 'lineage' = 'dataflow') {
    super();
    this.mode = mode;
    this.addClass('jp-react-widget');
  }

  // Lumino seamlessly bridges to React here
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
