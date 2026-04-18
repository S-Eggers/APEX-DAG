import { ReactWidget } from '@jupyterlab/apputils';
import React from 'react';
import EnvironmentComponent from './EnvironmentComponent';

export class EnvironmentWidget extends ReactWidget {
  private environmentData: any = null;

  constructor() {
    super();
    this.addClass('jp-react-widget');
    this.id = 'apex-environment-widget';
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
}
