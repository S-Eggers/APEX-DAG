import { Widget } from '@lumino/widgets';
import { NotebookPanel } from '@jupyterlab/notebook';
import { LabIcon } from '@jupyterlab/ui-components';

import CommandIDs from '../types/CommandIDs';
import { GraphWidget } from '../components/widget/GraphWidget';
import { EnvironmentWidget } from '../components/widget/EnvironmentWidget';
import { ExecutionStateWidget } from '../components/widget/ExecutionStateWidget';
import { ExecutionTraceWidget } from '../components/widget/ExecutionTraceWidget';
import { updateGraphWidget } from '../utils/GraphService';
import updateEnvironmentWidget from '../utils/updateEnvironmentWidget';
import updateExecutionStateWidget from '../utils/updateExecutionStateWidget';
import updateExecutionTraceWidget from '../utils/updateExecutionTraceWidget';
import { AppSettings } from '../settings/AppSettings';

import {
  astIcon,
  dataflowIcon,
  lineageIcon,
  vamsaIcon,
  labelingIcon,
  leakageIcon,
  tupleAnnotationIcon,
  environmentIcon,
  executionStateIcon,
  executionTraceIcon
} from '../utils/SystemXIcons';

export type WidgetType =
  | 'dataflow'
  | 'lineage'
  | 'lineage_annotation'
  | 'environment'
  | 'ast'
  | 'vamsa_wir'
  | 'vamsa_lineage'
  | 'labeling'
  | 'leakage'
  | 'execution_state'
  | 'execution_trace';

export interface WidgetConfig {
  type: WidgetType;
  commandId: string;
  label: string;
  rank: number;
  group: number;
  icon: LabIcon;
  debouncedUpdate: boolean;
  factory: (commands?: any) => Widget;
  update: (
    content: Widget,
    nbPanel: NotebookPanel,
    settings: AppSettings
  ) => void | Promise<void>;
}

export const WIDGET_REGISTRY: WidgetConfig[] = [
  {
    type: 'ast',
    commandId: CommandIDs.ast,
    label: 'AST',
    rank: 1,
    group: 1,
    icon: astIcon,
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
    icon: dataflowIcon,
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
    icon: lineageIcon,
    debouncedUpdate: true,
    factory: () => new GraphWidget('lineage'),
    update: (content, nbPanel, settings) =>
      updateGraphWidget(content as GraphWidget, nbPanel, 'lineage', settings)
  },
  {
    type: 'labeling',
    commandId: CommandIDs.labeling,
    label: 'Annotate',
    rank: 4,
    group: 3,
    icon: labelingIcon,
    debouncedUpdate: true,
    factory: commands => new GraphWidget('labeling', commands),
    update: (content, nbPanel, settings) =>
      updateGraphWidget(content as GraphWidget, nbPanel, 'labeling', settings)
  },
  {
    type: 'lineage_annotation',
    commandId: CommandIDs.lineage_annotation,
    label: 'Annotate Tuples',
    rank: 5,
    group: 3,
    icon: tupleAnnotationIcon,
    debouncedUpdate: true,
    factory: commands => new GraphWidget('lineage_annotation', commands),
    update: (content, nbPanel, settings) =>
      updateGraphWidget(
        content as GraphWidget,
        nbPanel,
        'lineage_annotation',
        settings
      )
  },
  {
    type: 'leakage',
    commandId: CommandIDs.leakage,
    label: 'Leakage',
    rank: 6,
    group: 3,
    icon: leakageIcon,
    debouncedUpdate: true,
    factory: commands => new GraphWidget('leakage', commands),
    update: (content, nbPanel, settings) =>
      updateGraphWidget(content as GraphWidget, nbPanel, 'leakage', settings)
  },
  {
    type: 'vamsa_wir',
    commandId: CommandIDs.vamsa_wir,
    label: 'Vamsa (WIR)',
    rank: 7,
    group: 4,
    icon: vamsaIcon,
    debouncedUpdate: false,
    factory: () => new GraphWidget('vamsa_wir'),
    update: (content, nbPanel, settings) => {
      updateGraphWidget(content as GraphWidget, nbPanel, 'vamsa_wir', settings);
    }
  },
  {
    type: 'vamsa_lineage',
    commandId: CommandIDs.vamsa_lineage,
    label: 'Vamsa (Lineage)',
    rank: 8,
    group: 4,
    icon: vamsaIcon,
    debouncedUpdate: false,
    factory: () => new GraphWidget('vamsa_lineage'),
    update: (content, nbPanel, settings) => {
      updateGraphWidget(
        content as GraphWidget,
        nbPanel,
        'vamsa_lineage',
        settings
      );
    }
  },
  {
    type: 'environment',
    commandId: CommandIDs.environment,
    label: 'Environment',
    rank: 9,
    group: 5,
    icon: environmentIcon,
    debouncedUpdate: false,
    factory: () => new EnvironmentWidget(),
    update: (content, nbPanel, settings) => {
      updateEnvironmentWidget(content as EnvironmentWidget, nbPanel, settings);
    }
  },
  {
    type: 'execution_state',
    commandId: CommandIDs.execution_state,
    label: 'Execution State',
    rank: 10,
    group: 5,
    icon: executionStateIcon,
    debouncedUpdate: true,
    factory: () => new ExecutionStateWidget(),
    update: (content, nbPanel, settings) =>
      updateExecutionStateWidget(
        content as ExecutionStateWidget,
        nbPanel,
        settings
      )
  },
  {
    type: 'execution_trace',
    commandId: CommandIDs.execution_trace,
    label: 'Execution Trace',
    rank: 11,
    group: 5,
    icon: executionTraceIcon,
    debouncedUpdate: true,
    factory: () => new ExecutionTraceWidget(),
    update: (content, nbPanel, settings) =>
      updateExecutionTraceWidget(
        content as ExecutionTraceWidget,
        nbPanel,
        settings
      )
  }
];
