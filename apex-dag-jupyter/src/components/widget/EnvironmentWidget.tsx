import { ReactWidget } from '@jupyterlab/apputils';
import React from 'react';
import { EnvironmentComponent, EnvironmentData } from './EnvironmentComponent';
import ApexIcon from '../../utils/ApexIcon';

export class EnvironmentWidget extends ReactWidget {
  private _data: EnvironmentData | null = null;
  private _isLoading: boolean = false;
  private _error: string | null = null;

  constructor() {
    super();
    this.addClass('apex-environment-widget');
    this.title.label = 'Environment';
    this.title.icon = ApexIcon;
    this.title.closable = true;
  }

  public setData(data: EnvironmentData): void {
    this._data = data;
    this._isLoading = false;
    this._error = null;
    this.update();
  }

  public setLoading(isLoading: boolean): void {
    this._isLoading = isLoading;
    this.update();
  }

  public setError(errorMsg: string): void {
    this._error = errorMsg;
    this._isLoading = false;
    this.update();
  }

  /**
   * Lumino -> React injection point.
   */
  protected render(): JSX.Element {
    return (
      <EnvironmentComponent
        data={this._data}
        isLoading={this._isLoading}
        error={this._error}
      />
    );
  }
}
