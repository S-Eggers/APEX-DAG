import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import {
  ICommandPalette,
  MainAreaWidget,
  CommandToolbarButton
} from '@jupyterlab/apputils';
import { ILauncher } from '@jupyterlab/launcher';
import { IMainMenu } from '@jupyterlab/mainmenu';
import {
  INotebookTracker,
  NotebookActions,
  NotebookPanel
} from '@jupyterlab/notebook';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { toArray } from '@lumino/algorithm';
import { Widget } from '@lumino/widgets';

import { GraphWidget } from './components/widget/GraphWidget';
import { EnvironmentWidget } from './components/widget/EnvironmentWidget';
import CommandIDs from './types/CommandIDs';
import ApexIcon from './utils/ApexIcon';
import updateLineageWidget from './utils/updateLineageWidget';
import updateWidget from './utils/updateWidget';
import updateEnvironmentWidget from './utils/updateEnvironmentWidget';

class AppSettings {
  debounceDelay = 1000;
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
  }
}

type WidgetType = 'dataflow' | 'lineage' | 'environment';

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
    const appSettings = new AppSettings();

    // Centralized wrapper tracking
    const wrappers: Record<WidgetType, MainAreaWidget<Widget> | null> = {
      dataflow: null,
      lineage: null,
      environment: null
    };

    const getOrCreateWidget = (
      label: string,
      type: WidgetType
    ): MainAreaWidget<Widget> => {
      let wrapper = wrappers[type];

      if (wrapper && !wrapper.isDisposed) {
        return wrapper;
      }

      let content: Widget;
      if (type === 'environment') {
        content = new EnvironmentWidget();
      } else {
        content = new GraphWidget(type === 'lineage' ? 'lineage' : undefined);
      }

      wrapper = new MainAreaWidget({ content });
      wrapper.title.label = label;
      wrapper.title.icon = ApexIcon;
      wrapper.title.closable = true;

      wrappers[type] = wrapper;
      return wrapper;
    };

    const updateUI = (notebookPanel: NotebookPanel) => {
      if (wrappers.dataflow && !wrappers.dataflow.isDisposed) {
        updateWidget(
          wrappers.dataflow.content as GraphWidget,
          appSettings.replaceDataflowInUDFs,
          appSettings.greedyNotebookExtraction,
          appSettings.highlightRelevantSubgraphs,
          notebookPanel
        );
      }
      if (wrappers.environment && !wrappers.environment.isDisposed) {
        updateEnvironmentWidget(
          //wrappers.environment.content as EnvironmentWidget,
          notebookPanel
        );
      }
    };

    const updateLineage = (notebookPanel: NotebookPanel) => {
      if (wrappers.lineage && !wrappers.lineage.isDisposed) {
        updateLineageWidget(
          wrappers.lineage.content as GraphWidget,
          appSettings.replaceDataflowInUDFs,
          appSettings.highlightRelevantSubgraphs,
          appSettings.greedyNotebookExtraction,
          appSettings.llmClassification,
          notebookPanel
        );
      }
    };

    const registerCommand = (
      commandId: string,
      label: string,
      type: WidgetType,
      rank: number
    ) => {
      app.commands.addCommand(commandId, {
        caption: label,
        label: label,
        icon: args => (args['isPalette'] ? undefined : ApexIcon),
        execute: () => {
          const wrapper = getOrCreateWidget(label, type);
          const currentNb = tracker.currentWidget;

          if (!wrapper.isAttached) {
            app.shell.add(wrapper, 'main', {
              mode: 'split-right',
              ref: currentNb?.id
            });
          }
          app.shell.activateById(wrapper.id);

          if (currentNb) {
            type === 'lineage' ? updateLineage(currentNb) : updateUI(currentNb);
          }
        }
      });

      const item = { command: commandId, category: 'APEX-DAG', rank };
      palette.addItem(item);
      launcher.add(item);
      return commandId;
    };

    const dataflowCmd = registerCommand(
      CommandIDs.dataflow,
      'Dataflow',
      'dataflow',
      1
    );
    const lineageCmd = registerCommand(
      CommandIDs.lineage,
      'Lineage',
      'lineage',
      2
    );
    const envCmd = registerCommand(
      CommandIDs.environment,
      'Environment',
      'environment',
      3
    );

    mainMenu.fileMenu.newMenu.addGroup([
      { command: dataflowCmd },
      { command: lineageCmd },
      { command: envCmd }
    ]);

    tracker.widgetAdded.connect((sender, notebookPanel) => {
      const dfButton = new CommandToolbarButton({
        commands: app.commands,
        id: CommandIDs.dataflow,
        label: 'Dataflow'
      });
      const linButton = new CommandToolbarButton({
        commands: app.commands,
        id: CommandIDs.lineage,
        label: 'Lineage'
      });
      const envButton = new CommandToolbarButton({
        commands: app.commands,
        id: CommandIDs.environment,
        label: 'Environment'
      });

      const names = toArray(notebookPanel.toolbar.names());
      const cellTypeIdx = names.indexOf('cellType');
      const insertIdx = cellTypeIdx !== -1 ? cellTypeIdx + 1 : 10;

      notebookPanel.toolbar.insertItem(
        insertIdx,
        'apex-dataflow-btn',
        dfButton
      );
      notebookPanel.toolbar.insertItem(
        insertIdx + 1,
        'apex-lineage-btn',
        linButton
      );
      notebookPanel.toolbar.insertItem(
        insertIdx + 2,
        'apex-environment-btn',
        envButton
      );

      let debounceTimer: ReturnType<typeof setTimeout>;

      notebookPanel.context.ready.then(() => {
        const model = notebookPanel.content.model;
        if (!model) return;

        model.contentChanged.connect(() => {
          updateUI(notebookPanel);

          if (appSettings.debounceDelay >= 0) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
              updateLineage(notebookPanel);
            }, appSettings.debounceDelay);
          }
        });
      });
    });

    NotebookActions.executed.connect((sender, { notebook }) => {
      if (appSettings.debounceDelay < 0) {
        const panel = tracker.find(p => p.content === notebook);
        if (panel) updateLineage(panel);
      }
    });

    tracker.currentChanged.connect(async (sender, notebookPanel) => {
      if (!notebookPanel) return;
      await notebookPanel.revealed;
      updateUI(notebookPanel);
      updateLineage(notebookPanel);
    });

    if (settingRegistry) {
      const settings = await settingRegistry.load(plugin.id);
      const onSettingsChanged = (
        newSettings: ISettingRegistry.ISettings
      ): void => {
        appSettings.update(newSettings);
        const notebookPanel = tracker.currentWidget;
        if (notebookPanel) {
          updateUI(notebookPanel);
          updateLineage(notebookPanel);
        }
      };
      onSettingsChanged(settings);
      settings.changed.connect(onSettingsChanged);
    }
  }
};

export default plugin;
