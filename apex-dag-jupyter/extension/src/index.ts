import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { MainAreaWidget } from '@jupyterlab/apputils';
import { ILauncher } from '@jupyterlab/launcher';
import { reactIcon } from '@jupyterlab/ui-components';
import { GraphWidget } from './widget';

namespace CommandIDs {
  export const create = 'apex-dag-widget';
}

/**
 * Initialization data for the apex-dag extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'apex-dag:plugin',
  description: 'APEX-DAG Jupyter Lab Extension',
  autoStart: true,
  optional: [ILauncher],
  activate: (app: JupyterFrontEnd, launcher: ILauncher) => {
    const { commands } = app;

    const command = CommandIDs.create;
    commands.addCommand(command, {
      caption: 'APEX-DAG',
      label: 'APEX-DAG Widget',
      icon: args => (args['isPalette'] ? undefined : reactIcon),
      execute: () => {
        const content = new GraphWidget();
        const widget = new MainAreaWidget<GraphWidget>({ content });
        widget.title.label = 'APEX-DAG';
        widget.title.icon = reactIcon;
        app.shell.add(widget, 'main');
      }
    });

    if (launcher) {
      launcher.add({
        command
      });
    }
  }
};

export default plugin;
