import { Widget } from '@lumino/widgets';
import { NotebookPanel } from '@jupyterlab/notebook';
import { LabIcon } from '@jupyterlab/ui-components';

import CommandIDs from '../types/CommandIDs';
import { GraphWidget } from '../components/widget/GraphWidget';
import { EnvironmentWidget } from '../components/widget/EnvironmentWidget';
import { updateGraphWidget } from '../utils/GraphService';
import updateEnvironmentWidget from '../utils/updateEnvironmentWidget';
import { AppSettings } from '../settings/AppSettings';

// Import the pre-built icons
import {
  astIcon,
  dataflowIcon,
  lineageIcon,
  vamsaIcon,
  labelingIcon,
  environmentIcon
} from '../utils/ApexIcons';

export type WidgetType =
  | 'dataflow'
  | 'lineage'
  | 'environment'
  | 'ast'
  | 'vamsa_wir'
  | 'vamsa_lineage'
  | 'labeling';

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
  ) => void;
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
    update: (content, nbPanel, settings) => {
      updateGraphWidget(content as GraphWidget, nbPanel, 'lineage', settings);
    }
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
    update: (content, nbPanel, settings) => {
      updateGraphWidget(content as GraphWidget, nbPanel, 'labeling', settings);
    }
  },
  {
    type: 'vamsa_wir',
    commandId: CommandIDs.vamsa_wir,
    label: 'Vamsa (WIR)',
    rank: 5,
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
    rank: 6,
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
    rank: 7,
    group: 5,
    icon: environmentIcon,
    debouncedUpdate: false,
    factory: () => new EnvironmentWidget(),
    update: (content, nbPanel, settings) => {
      updateEnvironmentWidget(content as EnvironmentWidget, nbPanel, settings);
    }
  }
];
