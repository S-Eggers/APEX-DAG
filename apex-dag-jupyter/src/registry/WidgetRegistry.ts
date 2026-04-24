import { Widget } from '@lumino/widgets';
import { NotebookPanel } from '@jupyterlab/notebook';
import CommandIDs from '../types/CommandIDs';
import { GraphWidget } from '../components/widget/GraphWidget';
import { EnvironmentWidget } from '../components/widget/EnvironmentWidget';
import { updateGraphWidget } from '../utils/GraphService';
import updateEnvironmentWidget from '../utils/updateEnvironmentWidget';
import { AppSettings } from '../settings/AppSettings';

export type WidgetType =
  | 'dataflow'
  | 'lineage'
  | 'environment'
  | 'ast'
  | 'vamsa'
  | 'labeling';

export interface WidgetConfig {
  type: WidgetType;
  commandId: string;
  label: string;
  rank: number;
  group: number;
  debouncedUpdate: boolean;
  factory: () => Widget;
  update: (
    content: Widget,
    nbPanel: NotebookPanel,
    settings: AppSettings
  ) => void;
}

export const WIDGET_REGISTRY: WidgetConfig[] = [
  {
    type: 'ast',
    commandId: CommandIDs.ast,
    label: 'AST',
    rank: 1,
    group: 1,
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
    rank: 2,
    group: 2,
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
    rank: 3,
    group: 3,
    debouncedUpdate: true,
    factory: () => new GraphWidget('lineage'),
    update: (content, nbPanel, settings) => {
      updateGraphWidget(content as GraphWidget, nbPanel, 'lineage', settings);
    }
  },
  {
    type: 'vamsa',
    commandId: CommandIDs.vamsa,
    label: 'Vamsa',
    rank: 4,
    group: 3,
    debouncedUpdate: false,
    factory: () => new GraphWidget('vamsa'),
    update: (content, nbPanel, settings) => {
      updateGraphWidget(content as GraphWidget, nbPanel, 'vamsa', settings);
    }
  },
  {
    type: 'labeling',
    commandId: CommandIDs.labeling,
    label: 'Annotate',
    rank: 5,
    group: 3,
    debouncedUpdate: true,
    factory: () => new GraphWidget('labeling'),
    update: (content, nbPanel, settings) => {
      updateGraphWidget(content as GraphWidget, nbPanel, 'labeling', settings);
    }
  },
  {
    type: 'environment',
    commandId: CommandIDs.environment,
    label: 'Environment',
    rank: 6,
    group: 4,
    debouncedUpdate: false,
    factory: () => new EnvironmentWidget(),
    update: (content, nbPanel, settings) => {
      updateEnvironmentWidget(content as EnvironmentWidget, nbPanel, settings);
    }
  }
];
