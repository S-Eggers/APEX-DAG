import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
  ILayoutRestorer,
  ILabShell
} from '@jupyterlab/application';
import {
  ICommandPalette,
  MainAreaWidget,
  WidgetTracker
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

import CommandIDs from './types/CommandIDs';
import { AppSettings } from './settings/AppSettings';
import {
  WIDGET_REGISTRY,
  WidgetConfig,
  WidgetType
} from './registry/WidgetRegistry';
import { GraphWidget } from './components/widget/GraphWidget';
import { SystemXSettingsWidget } from './components/widget/SystemXSettingsWidget';
import { SystemXNativeDropdownWidget } from './components/toolbar/SystemXNativeDropdownWidget';
import { SYSTEMX_GRADIENT_SVG, settingsIcon } from './utils/SystemXIcons';
import callBackend, { getBackend } from './utils/callBackend';
import { recordExecution } from './utils/ExecutionTracker';
import { traceRecorder } from './utils/ExecutionTraceRecorder';
import { NotebookTrace } from './types/ExecutionTraceTypes';

interface IDiscoveredDataset {
  path: string;
  label: string;
  notebooks: number;
  annotations: number;
  has_annotations: boolean;
}

async function injectDiscoveredDatasets(
  settingRegistry: ISettingRegistry
): Promise<void> {
  let datasets: IDiscoveredDataset[] = [];
  try {
    const response = await getBackend<{
      success: boolean;
      datasets: IDiscoveredDataset[];
    }>('datasets');
    datasets = response?.datasets ?? [];
  } catch (err) {
    console.warn(
      '[SystemX] Dataset auto-discovery failed; keeping free-text dataset path.',
      err
    );
    return;
  }

  settingRegistry.transform(CommandIDs.plugin, {
    fetch: loaded => {
      if (datasets.length === 0) {
        return loaded;
      }
      const extraction = loaded.schema.properties?.extraction as
        | { properties?: Record<string, unknown> }
        | undefined;
      const prop = extraction?.properties?.rawDatasetPath as
        | {
            enum?: string[];
            enumDescriptions?: string[];
            description?: string;
          }
        | undefined;
      if (prop) {
        prop.enum = datasets.map(d => d.path);
        prop.enumDescriptions = datasets.map(d => {
          const ann = d.has_annotations ? `, ${d.annotations} annotations` : '';
          return `${d.label} - ${d.notebooks} notebooks${ann}`;
        });
        prop.description = `${prop.description ?? ''} Auto-discovered under ./data - pick a dataset from the list.`;
      }
      return loaded;
    }
  });
}

type LayoutAttachOptions = {
  mode?:
    | 'tab-after'
    | 'tab-before'
    | 'split-top'
    | 'split-bottom'
    | 'split-right'
    | 'split-left';
  ref?: string | null;
};

const plugin: JupyterFrontEndPlugin<void> = {
  id: CommandIDs.plugin,
  description: 'SystemX Jupyter Frontend Extension.',
  autoStart: true,
  optional: [ISettingRegistry],
  requires: [
    ICommandPalette,
    INotebookTracker,
    ILauncher,
    IMainMenu,
    ILayoutRestorer
  ],
  activate: async (
    app: JupyterFrontEnd,
    palette: ICommandPalette,
    tracker: INotebookTracker,
    launcher: ILauncher,
    mainMenu: IMainMenu,
    restorer: ILayoutRestorer,
    settingRegistry: ISettingRegistry | null
  ) => {
    const gradientContainer = document.createElement('div');
    gradientContainer.innerHTML = SYSTEMX_GRADIENT_SVG;
    document.body.appendChild(gradientContainer);

    const appSettings = new AppSettings();
    let systemxSettings: ISettingRegistry.ISettings | null = null;
    let settingsWidget: MainAreaWidget<SystemXSettingsWidget> | null = null;
    const activeWrappers = new Map<WidgetType, MainAreaWidget<Widget>>();

    const widgetTrackers = new Map<
      WidgetType,
      WidgetTracker<MainAreaWidget<Widget>>
    >();

    const dirtyWidgets = new Set<WidgetType>();
    let lastActiveNotebook: NotebookPanel | null = null;

    const getOrCreateWidget = (
      config: WidgetConfig
    ): MainAreaWidget<Widget> => {
      let wrapper = activeWrappers.get(config.type);
      if (wrapper && !wrapper.isDisposed) {
        return wrapper;
      }

      wrapper = new MainAreaWidget({ content: config.factory(app.commands) });

      wrapper.id = `systemx-widget-${config.type}`;
      wrapper.title.label = config.label;
      wrapper.title.icon = config.icon;
      wrapper.title.closable = true;

      wrapper.disposed.connect(() => {
        activeWrappers.delete(config.type);
        dirtyWidgets.delete(config.type);
      });

      activeWrappers.set(config.type, wrapper);
      return wrapper;
    };

    const triggerUpdate = (
      type: WidgetType,
      nbPanel: NotebookPanel
    ): void | Promise<void> => {
      const wrapper = activeWrappers.get(type);
      if (!wrapper || wrapper.isDisposed) return;

      if (wrapper.isVisible) {
        const config = WIDGET_REGISTRY.find(c => c.type === type);
        dirtyWidgets.delete(type);
        return config?.update(wrapper.content, nbPanel, appSettings);
      } else {
        dirtyWidgets.add(type);
      }
    };

    const mlInFlight = new Set<WidgetType>();
    const runThrottledUpdate = (
      type: WidgetType,
      nbPanel: NotebookPanel
    ): void => {
      if (mlInFlight.has(type)) return; // already running
      const result = triggerUpdate(type, nbPanel);
      if (result instanceof Promise) {
        mlInFlight.add(type);
        void result.finally(() => mlInFlight.delete(type));
      }
    };

    const isThrottled = (w: WidgetConfig): boolean =>
      w.debouncedUpdate &&
      !(
        (w.type === 'execution_state' || w.type === 'execution_trace') &&
        appSettings.execOrderBackend === 'heuristic'
      );

    const triggerAllUpdates = (nbPanel: NotebookPanel) => {
      WIDGET_REGISTRY.forEach(config => triggerUpdate(config.type, nbPanel));
    };

    (app.shell as ILabShell).currentChanged.connect((_, args) => {
      const activeWidget = args.newValue;
      if (!activeWidget) return;

      for (const [type, wrapper] of activeWrappers.entries()) {
        if (wrapper === activeWidget) {
          const targetNb = lastActiveNotebook || tracker.currentWidget;
          if (dirtyWidgets.has(type)) {
            if (targetNb) {
              const config = WIDGET_REGISTRY.find(c => c.type === type);
              if (config) config.update(wrapper.content, targetNb, appSettings);
              dirtyWidgets.delete(type);
            }
          } else if (wrapper.content instanceof GraphWidget) {
            wrapper.content.restoreHighlights();
          }
          break;
        }
      }
    });

    const menuGroup: { command: string }[] = [];

    WIDGET_REGISTRY.forEach(config => {
      const safeNamespace = `systemx-tracker-${String(config.type)}`;

      const widgetTracker = new WidgetTracker<MainAreaWidget<Widget>>({
        namespace: safeNamespace
      });
      widgetTrackers.set(config.type, widgetTracker);

      restorer.restore(widgetTracker, {
        command: config.commandId,
        name: () => String(config.type)
      });

      app.commands.addCommand(config.commandId, {
        caption: config.label,
        label: config.label,
        icon: args =>
          args['isPalette'] || args['isToolbar'] ? undefined : config.icon,
        execute: args => {
          const wrapper = getOrCreateWidget(config);
          const currentNb = tracker.currentWidget;

          if (!widgetTracker.has(wrapper)) {
            void widgetTracker.add(wrapper);
          }

          if (!wrapper.isAttached) {
            let attachOptions: LayoutAttachOptions | undefined = undefined;

            const attachedSystemXWidgets = Array.from(
              activeWrappers.values()
            ).filter(w => w.isAttached && !w.isDisposed);

            if (attachedSystemXWidgets.length > 0) {
              const lastSystemXWidget =
                attachedSystemXWidgets[attachedSystemXWidgets.length - 1];
              attachOptions = {
                mode: 'tab-after',
                ref: lastSystemXWidget.id
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

          return wrapper;
        }
      });

      const item = {
        command: config.commandId,
        category: 'SystemX',
        rank: config.rank
      };
      palette.addItem(item);
      launcher.add(item);
      menuGroup.push({ command: config.commandId });
    });

    mainMenu.fileMenu.newMenu.addGroup(menuGroup);

    const settingsCmdId = 'systemx:open-settings';
    const settingsTracker = new WidgetTracker<MainAreaWidget<Widget>>({
      namespace: 'systemx-settings'
    });

    app.commands.addCommand(settingsCmdId, {
      label: 'Settings',
      caption: 'Open SystemX Configuration',
      icon: args => (args['isPalette'] ? undefined : settingsIcon),
      execute: () => {
        if (!systemxSettings) {
          void app.commands.execute('settingeditor:open', {
            query: 'SystemX'
          });
          return;
        }
        if (!settingsWidget || settingsWidget.isDisposed) {
          const content = new SystemXSettingsWidget(systemxSettings);
          settingsWidget = new MainAreaWidget({ content });
          settingsWidget.id = 'systemx-settings';
          settingsWidget.title.label = 'SystemX Settings';
          settingsWidget.title.icon = settingsIcon;
          settingsWidget.title.closable = true;
          settingsWidget.disposed.connect(() => {
            settingsWidget = null;
          });
        }
        if (!settingsTracker.has(settingsWidget)) {
          void settingsTracker.add(settingsWidget);
        }
        if (!settingsWidget.isAttached) {
          const attachedSystemXWidgets = Array.from(
            activeWrappers.values()
          ).filter(w => w.isAttached && !w.isDisposed);
          const attachOptions: LayoutAttachOptions | undefined =
            attachedSystemXWidgets.length > 0
              ? {
                  mode: 'tab-after',
                  ref: attachedSystemXWidgets[attachedSystemXWidgets.length - 1].id
                }
              : undefined;
          app.shell.add(settingsWidget, 'main', attachOptions);
        }
        app.shell.activateById(settingsWidget.id);
      }
    });

    void restorer.restore(settingsTracker, {
      command: settingsCmdId,
      name: () => 'systemx-settings'
    });

    launcher.add({
      command: settingsCmdId,
      category: 'SystemX',
      rank: 100
    });

    tracker.widgetAdded.connect((sender, notebookPanel) => {
      const names = toArray(notebookPanel.toolbar.names());
      const cellTypeIdx = names.indexOf('cellType');
      const insertIdx = cellTypeIdx !== -1 ? cellTypeIdx + 1 : 10;

      const dropdownWidget = new SystemXNativeDropdownWidget(app.commands);
      notebookPanel.toolbar.insertItem(
        insertIdx,
        'systemx-dropdown',
        dropdownWidget
      );

      let debounceTimer: ReturnType<typeof setTimeout>;

      notebookPanel.context.ready.then(() => {
        const model = notebookPanel.content.model;
        if (!model) return;

        if (appSettings.traceEnabled) {
          traceRecorder.attach(notebookPanel, appSettings);
          void callBackend<{ success: boolean; trace: NotebookTrace | null }>(
            'execution-trace/load',
            {
              filename: notebookPanel.context.path,
              base_path: appSettings.traceStoragePath
            }
          )
            .then(response => {
              if (response.success && response.trace) {
                traceRecorder.hydrate(
                  notebookPanel.context.path,
                  response.trace
                );
              }
            })
            .catch(() => {});
        }

        model.contentChanged.connect(() => {
          WIDGET_REGISTRY.filter(w => !isThrottled(w)).forEach(w =>
            triggerUpdate(w.type, notebookPanel)
          );

          if (appSettings.debounceDelay >= 0) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
              WIDGET_REGISTRY.filter(w => isThrottled(w)).forEach(w =>
                runThrottledUpdate(w.type, notebookPanel)
              );
            }, appSettings.debounceDelay);
          }
        });
      });
    });

    NotebookActions.executed.connect((sender, { cell, notebook, success }) => {
      recordExecution(cell);

      const panel = tracker.find(p => p.content === notebook);
      if (panel) {
        traceRecorder.record(cell, panel, success);
        runThrottledUpdate('execution_trace', panel);
      }

      if (appSettings.debounceDelay < 0 && panel) {
        WIDGET_REGISTRY.filter(w => isThrottled(w)).forEach(w =>
          runThrottledUpdate(w.type, panel)
        );
      }
    });

    tracker.currentChanged.connect(async (sender, notebookPanel) => {
      if (!notebookPanel) return;

      lastActiveNotebook = notebookPanel;

      await notebookPanel.revealed;
      triggerAllUpdates(notebookPanel);
    });

    document.body.classList.add('systemx-theme');

    if (settingRegistry) {
      await injectDiscoveredDatasets(settingRegistry);

      const settings = await settingRegistry.load(plugin.id);
      systemxSettings = settings;

      app.commands.addCommand('systemx:set-model-variant', {
        label: 'SystemX: Select Model Variant',
        execute: async args => {
          const variant = (args?.variant as string) ?? '';
          try {
            const ml =
              (settings.get('ml').composite as Record<string, unknown>) || {};
            await settings.set('ml', { ...ml, modelVariant: variant } as never);
          } catch (e) {
            console.error('SystemX: failed to set model variant', e);
          }
        }
      });

      const applyGlobalTheme = (s: ISettingRegistry.ISettings): void => {
        const ui = s.get('ui').composite as
          | { applyGlobalTheme?: boolean }
          | undefined;
        const enabled = ui?.applyGlobalTheme ?? true;
        document.body.classList.toggle('systemx-theme', enabled);
      };

      const onSettingsChanged = (
        newSettings: ISettingRegistry.ISettings
      ): void => {
        appSettings.update(newSettings);
        applyGlobalTheme(newSettings);

        const targetNb = lastActiveNotebook || tracker.currentWidget;
        if (targetNb) {
          triggerAllUpdates(targetNb);
        }
      };

      onSettingsChanged(settings);
      settings.changed.connect(onSettingsChanged);
    }
  }
};

export default plugin;
