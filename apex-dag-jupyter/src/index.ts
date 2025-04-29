import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { ICommandPalette, MainAreaWidget } from '@jupyterlab/apputils';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { ILauncher } from '@jupyterlab/launcher';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { ICodeCellModel } from '@jupyterlab/cells';

import { GraphWidget } from './widget';
import apexDagLogo from './apex_icon'
import { callBackend } from './call_backend';

namespace CommandIDs {
  export const fullscreen = 'apex:open-fullscreen';
}

/**
 * Initialization data for the apex-dag-jupyter extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'apex-dag-jupyter:plugin',
  description: 'APEX-DAG Jupyter Frontend Extension.',
  autoStart: true,
  optional: [ISettingRegistry],
  requires: [ICommandPalette, INotebookTracker, ILauncher],
  activate: (app: JupyterFrontEnd, palette: ICommandPalette, tracker: INotebookTracker, launcher: ILauncher) => {
    console.log('JupyterLab extension apex-dag-jupyter is activated!');
    console.log('ICommandPalette:', palette);
    console.log('JupyterLab extension jupyterlab_apod is activated!');

    const fullscreen: string = CommandIDs.fullscreen;
    let graphWidget: GraphWidget | null = null;
    app.commands.addCommand(fullscreen, {
      caption: 'APEX-DAG: Fullscreen Widget',
      label: 'APEX-DAG: Fullscreen Widget',
      icon: args => (args['isPalette'] ? undefined : apexDagLogo),
      execute: () => {
        const content = new GraphWidget();
        const widget = new MainAreaWidget<GraphWidget>({ content });
        widget.title.label = 'DAG: Fullscreen Widget';
        widget.title.icon = apexDagLogo;
        app.shell.add(widget, 'main');
        graphWidget = content;
      }
    });

    const fullscreenCommandItem = {
      command: fullscreen,
      category: 'APEX-DAG',
      rank: 1,
    }
    
    palette.addItem(fullscreenCommandItem);

    if (launcher) {
      launcher.add(fullscreenCommandItem);
    }

    tracker.currentChanged.connect(async (sender, notebookPanel: NotebookPanel | null) => {
      console.log('Current notebook changed', notebookPanel);

      if (!notebookPanel) {
        console.warn('No active notebook.');
        return;
      }

      await notebookPanel.revealed;
      console.log('Notebook revealed');

      const content = notebookPanel.content;
      if (!content) {
        console.error('NotebookPanel has no content.');
        return;
      }

      const model = content.model;
      if (!model) {
        console.error('Notebook content has no model.');
        return;
      }

      console.log('Setting up listener for model changes');

      model.contentChanged.connect(() => {
        console.log('Notebook model changed!');
        // Your widget update logic here
        const cells = notebookPanel.content.model?.cells;
        if (cells) {
          let content: string = ""
            for (let i = 0; i < cells.length; i++) {
                const cell = cells.get(i);
                if (cell.type === 'code') {
                    const codeCell = cell as ICodeCellModel
                    content += codeCell.toJSON().source + "\n"
                }
            }
            console.log("Notebook content\n" + content)
            callBackend('dataflow', { code: content })
            .then(response => {
                console.log('Received from server (dataflow):', response);
                if (graphWidget && response.success) {
                  graphWidget.updateGraphData(response.dataflow);
                }
            })
            .catch(error => {
                console.error('Error sending dataflow:', error);
            });
        }
      });
    });

  }
};

export default plugin;
