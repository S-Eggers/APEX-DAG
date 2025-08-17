import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette, MainAreaWidget } from '@jupyterlab/apputils';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { ILauncher } from '@jupyterlab/launcher';
import {
  INotebookTracker,
  NotebookPanel,
  NotebookActions
} from '@jupyterlab/notebook';
import { IMainMenu } from '@jupyterlab/mainmenu';

import { GraphWidget } from './components/widget/GraphWidget';
import ApexIcon from './utils/ApexIcon';
import updateWidget from './utils/updateWidget';
import updateLineageWidget from './utils/updateLineageWidget';
import CommandIDs from './types/CommandIDs';

/**
 * Initialization data for the apex-dag-jupyter extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: CommandIDs.plugin,
  description: 'APEX-DAG Jupyter Frontend Extension.',
  autoStart: true,
  optional: [ISettingRegistry],
  requires: [ICommandPalette, INotebookTracker, ILauncher, IMainMenu],
  activate: async (
    app: JupyterFrontEnd,
    palette: ICommandPalette,
    tracker: INotebookTracker,
    launcher: ILauncher,
    mainMenu: IMainMenu,
    settingRegistry: ISettingRegistry | null
  ) => {
    let debounceDelay: number = -1;
    let replaceDataflowInUDFs: boolean = false;
    let highlightRelevantSubgraphs: boolean = false;
    let greedyNotebookExtraction: boolean = true;

    if (settingRegistry) {
      const settings = await settingRegistry.load(plugin.id);

      const onSettingsChanged = (
        newSettings: ISettingRegistry.ISettings
      ): void => {
        debounceDelay =
          (newSettings.get('debounceDelay').composite as number) ?? 1000;
        replaceDataflowInUDFs =
          (newSettings.get('replaceDataflowInUDFs').composite as boolean) ??
          false;
        highlightRelevantSubgraphs =
          (newSettings.get('highlightRelevantSubgraphs')
            .composite as boolean) ?? false;
        greedyNotebookExtraction =
          (newSettings.get('greedyNotebookExtraction').composite as boolean) ??
          true;
        console.debug('APEX-DAG settings updated:', {
          debounceDelay,
          replaceDataflowInUDFs,
          highlightRelevantSubgraphs
        });
      }; // Load initial settings

      onSettingsChanged(settings); // Listen for future changes

      settings.changed.connect(onSettingsChanged);
    }

    console.debug('JupyterLab extension apex-dag-jupyter is activated!');
    console.debug('ICommandPalette:', palette);

    const dataflowWidget: string = CommandIDs.dataflow;
    let dataflowGraphWidget: GraphWidget;
    app.commands.addCommand(dataflowWidget, {
      caption: 'Dataflow Widget',
      label: 'Dataflow Widget',
      icon: args => (args['isPalette'] ? undefined : ApexIcon),
      execute: () => {
        const content = new GraphWidget();
        const widget = new MainAreaWidget<GraphWidget>({ content });
        widget.title.label = 'Dataflow Widget';
        widget.title.icon = ApexIcon;
        widget.title.closable = true;
        app.shell.add(widget, 'main');
        dataflowGraphWidget = content;
      }
    });

    const dataflowCommandItem = {
      command: dataflowWidget,
      category: 'APEX-DAG',
      rank: 1
    };

    const lineageWidget: string = CommandIDs.lineage;
    let lineageGraphWidget: GraphWidget;
    app.commands.addCommand(lineageWidget, {
      caption: 'Lineage Widget',
      label: 'Lineage Widget',
      icon: args => (args['isPalette'] ? undefined : ApexIcon),
      execute: () => {
        const content = new GraphWidget('lineage');
        const widget = new MainAreaWidget<GraphWidget>({ content });
        widget.title.label = 'Lineage Widget';
        widget.title.icon = ApexIcon;
        widget.title.closable = true;
        app.shell.add(widget, 'main');
        lineageGraphWidget = content;
      }
    });

    const lineageCommandItem = {
      command: lineageWidget,
      category: 'APEX-DAG',
      rank: 2
    };

    palette.addItem(dataflowCommandItem);
    palette.addItem(lineageCommandItem);
    mainMenu.fileMenu.newMenu.addGroup([
      dataflowCommandItem,
      lineageCommandItem
    ]);
    launcher.add(dataflowCommandItem);
    launcher.add(lineageCommandItem);

    let interval: NodeJS.Timeout;
    tracker.currentChanged.connect(
      async (sender, notebookPanel: NotebookPanel | null) => {
        console.debug('Current notebook changed', notebookPanel);
        if (!notebookPanel) {
          if (interval) {
            clearTimeout(interval);
          }
          console.info('No active notebook or dataflow widget.');
          return;
        }

        await notebookPanel.revealed;
        console.debug('Notebook revealed');
        const content = notebookPanel.content;
        const model = content.model;
        if (!content || !model) {
          console.error('NotebookPanel has no content or model.');
          return;
        }

        console.debug('Initial call to updateWidget on notebook open');
        updateWidget(
          dataflowGraphWidget,
          replaceDataflowInUDFs,
          greedyNotebookExtraction,
          highlightRelevantSubgraphs,
          notebookPanel
        );
        updateLineageWidget(
          lineageGraphWidget,
          replaceDataflowInUDFs,
          highlightRelevantSubgraphs,
          greedyNotebookExtraction,
          notebookPanel
        );

        model.contentChanged.connect(() => {
          console.debug('Notebook model changed!');
          updateWidget(
            dataflowGraphWidget,
            replaceDataflowInUDFs,
            greedyNotebookExtraction,
            highlightRelevantSubgraphs,
            notebookPanel
          );

          if (debounceDelay >= 0) {
            if (interval) {
              clearTimeout(interval);
            }

            interval = setTimeout(() => {
              console.debug('Executing debounced lineage update.');
              updateLineageWidget(
                lineageGraphWidget,
                replaceDataflowInUDFs,
                highlightRelevantSubgraphs,
                greedyNotebookExtraction,
                notebookPanel
              );
            }, debounceDelay);
          }
        });

        if (debounceDelay < 0) {
          NotebookActions.executed.connect((sender, { notebook, cell }) => {
            if (notebookPanel && notebookPanel.content === notebook) {
              updateLineageWidget(
                lineageGraphWidget,
                replaceDataflowInUDFs,
                highlightRelevantSubgraphs,
                greedyNotebookExtraction,
                notebookPanel
              );
            }
          });
        }
      }
    );
  }
};

export default plugin;
