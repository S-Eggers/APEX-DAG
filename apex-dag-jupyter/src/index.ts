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
import { toArray } from '@lumino/algorithm';
import { Widget } from '@lumino/widgets';

import CommandIDs from './types/CommandIDs';
import ApexIcon from './utils/ApexIcon';
import { AppSettings } from './settings/AppSettings';
import {
  WIDGET_REGISTRY,
  WidgetConfig,
  WidgetType
} from './registry/WidgetRegistry';
import { ApexNativeDropdownWidget } from './components/toolbar/ApexNativeDropdownWidget';

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

      wrapper = new MainAreaWidget({ content: config.factory(app.commands) });
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
        execute: args => {
          const wrapper = getOrCreateWidget(config);
          const currentNb = tracker.currentWidget;

          if (!wrapper.isAttached) {
            let attachOptions: any = {};

            const attachedApexWidgets = Array.from(
              activeWrappers.values()
            ).filter(w => w.isAttached && !w.isDisposed);

            if (attachedApexWidgets.length > 0) {
              const lastApexWidget =
                attachedApexWidgets[attachedApexWidgets.length - 1];

              attachOptions = {
                mode: 'tab-after',
                ref: lastApexWidget.id
              };
            } else if (currentNb && args['isToolbar']) {
              attachOptions = {
                mode: 'split-right',
                ref: currentNb.id
              };
            }

            app.shell.add(wrapper, 'main', attachOptions);
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

    const settingsCmdId = 'apex-dag:open-settings';
    app.commands.addCommand(settingsCmdId, {
      label: 'Settings',
      caption: 'Open APEX-DAG Configuration',
      icon: args => (args['isPalette'] ? undefined : ApexIcon),
      execute: () => {
        app.commands.execute('settingeditor:open', { query: 'APEX-DAG' });
      }
    });

    launcher.add({
      command: settingsCmdId,
      category: 'APEX-DAG',
      rank: 100
    });

    tracker.widgetAdded.connect((sender, notebookPanel) => {
      const names = toArray(notebookPanel.toolbar.names());
      const cellTypeIdx = names.indexOf('cellType');
      const insertIdx = cellTypeIdx !== -1 ? cellTypeIdx + 1 : 10;

      const dropdownWidget = new ApexNativeDropdownWidget(app.commands);
      notebookPanel.toolbar.insertItem(
        insertIdx,
        'apex-dag-dropdown',
        dropdownWidget
      );

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
