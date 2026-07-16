import { NotebookPanel } from '@jupyterlab/notebook';
import { EnvironmentWidget } from '../components/widget/EnvironmentWidget';
import callBackend from './callBackend';
import { getNotebookCode } from './getNotebookCode';
import {
  applyHighlights,
  registerHighlightHoverHandler,
  HighlightTarget
} from './CodeHighlighter';

const _UNUSED = '#f59e0b'; // amber
const _STDLIB = '#94a3b8'; // slate/grey
const _LIB = '#6366f1'; // indigo (installed / resolved)

interface EnvImports {
  declared: [string, string | null][];
  counts: Record<string, number>;
  versions?: Record<string, string | null>;
}

export function buildEnvironmentHighlights(
  notebookPanel: NotebookPanel,
  env: { imports?: EnvImports } | null
): HighlightTarget[] {
  const imports = env?.imports;
  if (!imports?.declared?.length) return [];

  const specs = new Map<string, { color: string; label: string }>();
  for (const [modulePath, alias] of imports.declared) {
    const lookupName = alias || modulePath.split('.').pop() || modulePath;
    if (!lookupName || specs.has(lookupName)) continue;
    const displayName = modulePath;

    const top = modulePath.split('.')[0];
    const version = imports.versions?.[top];
    const count = imports.counts?.[lookupName] ?? 0;

    let color: string;
    let label: string;
    if (count === 0) {
      color = _UNUSED;
      label = `${displayName} · unused`;
    } else if (version === 'stdlib') {
      color = _STDLIB;
      label = `${displayName} · stdlib`;
    } else if (version) {
      color = _LIB;
      label = `${displayName} · ${version}`;
    } else {
      color = _LIB;
      label = displayName;
    }
    specs.set(lookupName, { color, label });
  }
  if (specs.size === 0) return [];

  const targets: HighlightTarget[] = [];
  for (const cellWidget of notebookPanel.content.widgets) {
    const cellId = cellWidget.model?.id;
    if (!cellId) continue;
    for (const [codeText, { color, label }] of specs) {
      targets.push({
        cellId,
        codeText,
        color,
        domainLabel: label,
        wholeWord: true
      });
    }
  }
  return targets;
}

export const updateEnvironmentWidget = (
  widget: EnvironmentWidget | null,
  notebookPanel: NotebookPanel,
  settings: any
) => {
  if (!widget) return;

  const cells = notebookPanel.content.model?.cells;
  if (!cells) {
    console.warn('No notebook cells available for environment extraction.');
    return;
  }

  const content = getNotebookCode(cells, settings.greedyNotebookExtraction);
  console.debug('[ENVIRONMENT] Extracted notebook content for telemetry.');

  const payload = {
    code: content
  };

  callBackend('environment', payload)
    .then(response => {
      console.info('[ENVIRONMENT] Raw response received:', response);

      const parsedResponse =
        typeof response === 'string' ? JSON.parse(response) : response;

      if (!parsedResponse.success) {
        console.error(
          '[ENVIRONMENT] Backend returned failure.',
          parsedResponse
        );
        return;
      }

      if (parsedResponse.environment_data) {
        console.debug(
          '[ENVIRONMENT] Successfully parsed telemetry, updating widget.'
        );
        widget.updateEnvironmentData(parsedResponse.environment_data);

        widget.trackNotebook(notebookPanel);
        const targets = buildEnvironmentHighlights(
          notebookPanel,
          parsedResponse.environment_data
        );
        applyHighlights(notebookPanel, targets);
        registerHighlightHoverHandler(notebookPanel);
      } else {
        console.error(
          '[ENVIRONMENT] Payload is missing the "environment_data" object.',
          parsedResponse
        );
      }
    })
    .catch(error => {
      console.error('[ENVIRONMENT] Network or parsing error:', error);
    });
};

export default updateEnvironmentWidget;
