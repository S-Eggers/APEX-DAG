import { ReactWidget } from '@jupyterlab/apputils';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import React from 'react';
import SystemXSettingsComponent from './SystemXSettingsComponent';

export class SystemXSettingsWidget extends ReactWidget {
  private settings: ISettingRegistry.ISettings;

  constructor(settings: ISettingRegistry.ISettings) {
    super();
    this.settings = settings;
    this.addClass('jp-react-widget');
    this.id = 'systemx-settings-widget';
    this.title.label = 'SystemX Settings';
    this.title.closable = true;
  }

  render(): JSX.Element {
    return <SystemXSettingsComponent settings={this.settings} />;
  }
}
