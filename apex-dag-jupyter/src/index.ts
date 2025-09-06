import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette, MainAreaWidget } from '@jupyterlab/apputils';
import { ILauncher } from '@jupyterlab/launcher';
import { IMainMenu } from '@jupyterlab/mainmenu';
import {
  INotebookTracker,
  NotebookActions,
  NotebookPanel
} from '@jupyterlab/notebook';
import { ISettingRegistry } from '@jupyterlab/settingregistry';

import { GraphWidget } from './components/widget/GraphWidget';
import CommandIDs from './types/CommandIDs';
import ApexIcon from './utils/ApexIcon';
import updateLineageWidget from './utils/updateLineageWidget';
import updateWidget from './utils/updateWidget';

class AppSettings {
  debounceDelay = -1;
  replaceDataflowInUDFs = false;
  highlightRelevantSubgraphs = false;
  greedyNotebookExtraction = true;
  llmClassification = false;

  update(settings: ISettingRegistry.ISettings) {
    this.debounceDelay =
      (settings.get('debounceDelay').composite as number) ?? 1000;
    this.replaceDataflowInUDFs =
      (settings.get('replaceDataflowInUDFs').composite as boolean) ?? false;
    this.highlightRelevantSubgraphs =
      (settings.get('highlightRelevantSubgraphs').composite as boolean) ??
      false;
    this.greedyNotebookExtraction =
      (settings.get('greedyNotebookExtraction').composite as boolean) ?? true;
    this.llmClassification =
      (settings.get('llmClassification').composite as boolean) ?? false;
    console.debug('APEX-DAG settings updated:', this);
  }
}

function createAndShowWidget(
  app: JupyterFrontEnd,
  label: string,
  type?: 'lineage'
): GraphWidget {
  const content = new GraphWidget(type);
  const widget = new MainAreaWidget<GraphWidget>({ content });
  widget.title.label = label;
  widget.title.icon = ApexIcon;
  widget.title.closable = true;
  app.shell.add(widget, 'main');
  return content;
}

function addMenuWidget(
  app: JupyterFrontEnd,
  palette: ICommandPalette,
  launcher: ILauncher,
  commandId: string,
  label: string,
  category: string,
  rank: number,
  onExecute: () => void
): { command: string; category: string; rank: number } {
  app.commands.addCommand(commandId, {
    caption: label,
    label: label,
    icon: args => (args['isPalette'] ? undefined : ApexIcon),
    execute: onExecute
  });

  const commandItem = { command: commandId, category, rank };
  palette.addItem(commandItem);
  launcher.add(commandItem);
  return commandItem;
}

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
    console.debug('JupyterLab extension apex-dag-jupyter is activated!');
    const appSettings = new AppSettings();

    let dataflowGraphWidget: GraphWidget;
    let lineageGraphWidget: GraphWidget;

    const dataflowCmd = addMenuWidget(
      app,
      palette,
      launcher,
      CommandIDs.dataflow,
      'Dataflow Widget',
      'APEX-DAG',
      1,
      () => {
        dataflowGraphWidget = createAndShowWidget(app, 'Dataflow Widget');
        const notebookPanel = tracker.currentWidget;
        if (notebookPanel) {
          updateDataflow(notebookPanel);
        }
      }
    );

    const lineageCmd = addMenuWidget(
      app,
      palette,
      launcher,
      CommandIDs.lineage,
      'Lineage Widget',
      'APEX-DAG',
      2,
      () => {
        lineageGraphWidget = createAndShowWidget(
          app,
          'Lineage Widget',
          'lineage'
        );
        const notebookPanel = tracker.currentWidget;
        if (notebookPanel) {
          updateLineage(notebookPanel);
        }
      }
    );

    mainMenu.fileMenu.newMenu.addGroup([dataflowCmd, lineageCmd]);

    const updateDataflow = (notebookPanel: NotebookPanel) => {
      if (dataflowGraphWidget) {
        updateWidget(
          dataflowGraphWidget,
          appSettings.replaceDataflowInUDFs,
          appSettings.greedyNotebookExtraction,
          appSettings.highlightRelevantSubgraphs,
          notebookPanel
        );
      }
    };

    const updateLineage = (notebookPanel: NotebookPanel) => {
      if (lineageGraphWidget) {
        updateLineageWidget(
          lineageGraphWidget,
          appSettings.replaceDataflowInUDFs,
          appSettings.highlightRelevantSubgraphs,
          appSettings.greedyNotebookExtraction,
          appSettings.llmClassification,
          notebookPanel
        );
      }
    };

    if (settingRegistry) {
      const settings = await settingRegistry.load(plugin.id);
      const onSettingsChanged = (
        newSettings: ISettingRegistry.ISettings
      ): void => {
        appSettings.update(newSettings);
        const notebookPanel = tracker.currentWidget;
        if (notebookPanel) {
          updateDataflow(notebookPanel);
          updateLineage(notebookPanel);
        }
      };
      onSettingsChanged(settings);
      settings.changed.connect(onSettingsChanged);
    }

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

        console.debug('Initial call to update widgets on notebook open');
        updateDataflow(notebookPanel);
        updateLineage(notebookPanel);

        model.contentChanged.connect(() => {
          console.debug('Notebook model changed!');
          updateDataflow(notebookPanel);

          if (appSettings.debounceDelay >= 0) {
            if (interval) {
              clearTimeout(interval);
            }

            interval = setTimeout(() => {
              console.debug('Executing debounced lineage update.');
              updateLineage(notebookPanel);
            }, appSettings.debounceDelay);
          }
        });

        if (appSettings.debounceDelay < 0) {
          NotebookActions.executed.connect((sender, { notebook }) => {
            if (notebookPanel && notebookPanel.content === notebook) {
              updateLineage(notebookPanel);
            }
          });
        }
      }
    );
  }
};

export default plugin;