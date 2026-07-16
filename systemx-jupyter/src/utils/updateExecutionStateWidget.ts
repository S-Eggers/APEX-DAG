import { NotebookPanel } from '@jupyterlab/notebook';
import { ExecutionStateWidget } from '../components/widget/ExecutionStateWidget';
import callBackend from './callBackend';
import { getNotebookCode } from './getNotebookCode';
import {
  applyHighlights,
  registerHighlightHoverHandler,
  HighlightTarget
} from './CodeHighlighter';
import {
  ExecutionStateData,
  STATUS_COLORS
} from '../types/ExecutionStateTypes';
import { AppSettings } from '../settings/AppSettings';

export function buildExecutionStateHighlights(
  data: ExecutionStateData
): HighlightTarget[] {
  const targets: HighlightTarget[] = [];
  const seen = new Set<string>();
  const push = (
    cellId: string,
    codeText: string,
    color: string,
    label: string
  ): void => {
    const key = `${cellId}:${codeText}`;
    if (seen.has(key) || !codeText) return;
    seen.add(key);
    targets.push({ cellId, codeText, color, domainLabel: label, wholeWord: true });
  };

  for (const edge of data.cell_graph.edges) {
    if (edge.kind === 'FILE_ARTIFACT' || edge.kind === 'OPAQUE') continue;
    const srcState = data.cell_states[edge.src_cell];
    const dstNode = data.cell_graph.cells.find(
      c => c.cell_id === edge.dst_cell
    );
    const srcNode = data.cell_graph.cells.find(
      c => c.cell_id === edge.src_cell
    );

    if (srcState && ['stale', 'dirty'].includes(srcState.status)) {
      push(
        edge.dst_cell,
        edge.name,
        STATUS_COLORS.stale,
        `\`${edge.name}\` comes from a stale/edited cell - re-run its producer`
      );
    }
    if (
      dstNode &&
      srcNode &&
      srcNode.document_index > dstNode.document_index
    ) {
      push(
        edge.dst_cell,
        edge.name,
        STATUS_COLORS.unknown,
        `\`${edge.name}\` is defined further down the notebook (cell #${srcNode.document_index + 1})`
      );
    }
    if (edge.ambiguous) {
      push(
        edge.dst_cell,
        edge.name,
        STATUS_COLORS.dirty,
        `\`${edge.name}\` is defined in ${edge.candidate_def_cells.length} cells - binding depends on execution order`
      );
    }
  }

  for (const node of data.cell_graph.cells) {
    for (const name of node.undefined_uses) {
      push(
        node.cell_id,
        name,
        STATUS_COLORS['missing-deps'],
        `\`${name}\` is defined in no current cell (deleted cell or magic residue)`
      );
    }
  }

  return targets;
}

export const updateExecutionStateWidget = (
  widget: ExecutionStateWidget | null,
  notebookPanel: NotebookPanel,
  settings: AppSettings
): Promise<void> => {
  if (!widget) return Promise.resolve();

  const cells = notebookPanel.content.model?.cells;
  if (!cells) return Promise.resolve();

  const content = getNotebookCode(cells, true);

  const payload = {
    cells: content,
    filename: notebookPanel.context.path,
    detectDsl: settings.detectDsl,
    execBackend: settings.execOrderBackend
  };

  return callBackend<{ success: boolean; execution_state: ExecutionStateData }>(
    'execution-state',
    payload
  )
    .then(response => {
      if (!response.success || !response.execution_state) {
        console.error('[EXECUTION-STATE] Backend returned failure.', response);
        return;
      }
      widget.updateExecutionState(response.execution_state);
      widget.trackNotebook(notebookPanel);
      applyHighlights(
        notebookPanel,
        buildExecutionStateHighlights(response.execution_state)
      );
      registerHighlightHoverHandler(notebookPanel);
    })
    .catch(error => {
      console.error('[EXECUTION-STATE] Network or parsing error:', error);
    });
};

export default updateExecutionStateWidget;
