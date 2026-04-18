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
import { updateGraphWidget } from './utils/GraphService';
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

type WidgetType = 'dataflow' | 'lineage' | 'environment' | 'ast' | 'vamsa';

interface WidgetConfig {
  type: WidgetType;
  commandId: string;
  label: string;
  rank: number;
  debouncedUpdate: boolean;
  factory: () => Widget;
  update: (
    content: Widget,
    nbPanel: NotebookPanel,
    settings: AppSettings
  ) => void;
}

const WIDGET_REGISTRY: WidgetConfig[] = [
  {
    type: 'ast',
    commandId: CommandIDs.ast,
    label: 'AST',
    rank: 4,
    debouncedUpdate: false,
    factory: () => new GraphWidget('ast'),
    update: (content, nbPanel, settings) => {
      updateGraphWidget(content as GraphWidget, nbPanel, 'ast', settings);
    }
  },
  {
    type: 'dataflow',
    commandId: CommandIDs.dataflow,
    label: 'Dataflow',
    rank: 1,
    debouncedUpdate: false,
    factory: () => new GraphWidget('dataflow'),
    update: (content, nbPanel, settings) => {
      updateGraphWidget(content as GraphWidget, nbPanel, 'dataflow', settings);
    }
  },
  {
    type: 'lineage',
    commandId: CommandIDs.lineage,
    label: 'Lineage',
    rank: 2,
    debouncedUpdate: true,
    factory: () => new GraphWidget('lineage'),
    update: (content, nbPanel, settings) => {
      updateGraphWidget(content as GraphWidget, nbPanel, 'lineage', settings);
    }
  },
  {
    type: 'vamsa',
    commandId: CommandIDs.vamsa,
    label: 'Lineage (Vamsa)',
    rank: 4,
    debouncedUpdate: true,
    factory: () => new EnvironmentWidget(), // Placeholder
    update: (content, nbPanel, settings) => {
      // updateVamsaWidget(content, nbPanel);
    }
  },
  {
    type: 'environment',
    commandId: CommandIDs.environment,
    label: 'Environment',
    rank: 5,
    debouncedUpdate: false,
    factory: () => new EnvironmentWidget(),
    update: (content, nbPanel, settings) => {
      updateEnvironmentWidget(nbPanel);
    }
  }
];

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
    const activeWrappers = new Map<WidgetType, MainAreaWidget<Widget>>();

    const getOrCreateWidget = (
      config: WidgetConfig
    ): MainAreaWidget<Widget> => {
      let wrapper = activeWrappers.get(config.type);
      if (wrapper && !wrapper.isDisposed) {
        return wrapper;
      }

      wrapper = new MainAreaWidget({ content: config.factory() });
      wrapper.title.label = config.label;
      wrapper.title.icon = ApexIcon;
      wrapper.title.closable = true;

      activeWrappers.set(config.type, wrapper);
      return wrapper;
    };

    const triggerUpdate = (type: WidgetType, nbPanel: NotebookPanel) => {
      const wrapper = activeWrappers.get(type);
      if (wrapper && !wrapper.isDisposed) {
        const config = WIDGET_REGISTRY.find(c => c.type === type);
        if (config) config.update(wrapper.content, nbPanel, appSettings);
      }
    };

    const triggerAllUpdates = (nbPanel: NotebookPanel) => {
      WIDGET_REGISTRY.forEach(config => triggerUpdate(config.type, nbPanel));
    };

    const menuGroup: { command: string }[] = [];

    WIDGET_REGISTRY.forEach(config => {
      app.commands.addCommand(config.commandId, {
        caption: config.label,
        label: config.label,
        icon: args =>
          args['isPalette'] || args['isToolbar'] ? undefined : ApexIcon,
        execute: () => {
          const wrapper = getOrCreateWidget(config);
          const currentNb = tracker.currentWidget;

          if (!wrapper.isAttached) {
            app.shell.add(wrapper, 'main', {
              mode: 'split-right',
              ref: currentNb?.id
            });
          }
          app.shell.activateById(wrapper.id);

          if (currentNb) triggerUpdate(config.type, currentNb);
        }
      });

      const item = {
        command: config.commandId,
        category: 'APEX-DAG',
        rank: config.rank
      };
      palette.addItem(item);
      launcher.add(item);
      menuGroup.push({ command: config.commandId });
    });

    mainMenu.fileMenu.newMenu.addGroup(menuGroup);

    tracker.widgetAdded.connect((sender, notebookPanel) => {
      const names = toArray(notebookPanel.toolbar.names());
      const cellTypeIdx = names.indexOf('cellType');
      let insertIdx = cellTypeIdx !== -1 ? cellTypeIdx + 1 : 10;

      WIDGET_REGISTRY.forEach(config => {
        const button = new CommandToolbarButton({
          commands: app.commands,
          id: config.commandId,
          label: config.label,
          args: { isToolbar: true }
        });
        notebookPanel.toolbar.insertItem(
          insertIdx++,
          `apex-${config.type}-btn`,
          button
        );
      });

      let debounceTimer: ReturnType<typeof setTimeout>;

      notebookPanel.context.ready.then(() => {
        const model = notebookPanel.content.model;
        if (!model) return;

        model.contentChanged.connect(() => {
          WIDGET_REGISTRY.filter(w => !w.debouncedUpdate).forEach(w =>
            triggerUpdate(w.type, notebookPanel)
          );

          if (appSettings.debounceDelay >= 0) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
              WIDGET_REGISTRY.filter(w => w.debouncedUpdate).forEach(w =>
                triggerUpdate(w.type, notebookPanel)
              );
            }, appSettings.debounceDelay);
          }
        });
      });
    });

    NotebookActions.executed.connect((sender, { notebook }) => {
      if (appSettings.debounceDelay < 0) {
        const panel = tracker.find(p => p.content === notebook);
        if (panel) {
          WIDGET_REGISTRY.filter(w => w.debouncedUpdate).forEach(w =>
            triggerUpdate(w.type, panel)
          );
        }
      }
    });

    tracker.currentChanged.connect(async (sender, notebookPanel) => {
      if (!notebookPanel) return;
      await notebookPanel.revealed;
      triggerAllUpdates(notebookPanel);
    });

    if (settingRegistry) {
      const settings = await settingRegistry.load(plugin.id);
      const onSettingsChanged = (
        newSettings: ISettingRegistry.ISettings
      ): void => {
        appSettings.update(newSettings);
        const notebookPanel = tracker.currentWidget;
        if (notebookPanel) triggerAllUpdates(notebookPanel);
      };
      onSettingsChanged(settings);
      settings.changed.connect(onSettingsChanged);
    }
  }
};

export default plugin;
