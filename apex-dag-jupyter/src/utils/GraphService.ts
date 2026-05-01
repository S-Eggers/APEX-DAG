import { NotebookPanel } from '@jupyterlab/notebook';
import { GraphWidget } from '../components/widget/GraphWidget';
import callBackend from './callBackend';
import { getNotebookCode } from './getNotebookCode';
import { GraphMode } from '../types/GraphTypes';

export const updateGraphWidget = (
  graphWidget: GraphWidget | null,
  notebookPanel: NotebookPanel,
  mode: GraphMode,
  settings: any
) => {
  if (!graphWidget) return;

  if (settings && settings.rawDatasetPath && 'setDatasetPath' in graphWidget) {
    (graphWidget as any).setDatasetPath(settings.rawDatasetPath);
  }

  const cells = notebookPanel.content.model?.cells;
  if (!cells) {
    console.warn('No notebook cells available for extraction.');
    return;
  }
  const fullPath = notebookPanel.context.localPath || 'untitled.ipynb';
  const notebookName =
    fullPath.split('/').pop()?.replace('.ipynb', '') || 'untitled';

  const content = getNotebookCode(cells, settings.greedyNotebookExtraction);
  console.debug(
    `[${mode.toUpperCase()}] Extracted notebook content:\n`,
    content
  );
  const legacyConcatenatedCode = content.map(c => c.source).join('\n');

  const payload: any = {
    code: legacyConcatenatedCode,
    cells: content,
    replaceDataflowInUDFs: settings.replaceDataflowInUDFs,
    highlightRelevantSubgraphs: settings.highlightRelevantSubgraphs
  };

  if (mode === 'lineage' || mode === 'labeling') {
    payload.llmClassification = settings.llmClassification;
  }
  if (mode === 'labeling') {
    payload.filename = notebookPanel.context.path;
    payload.useGraphRefiner = settings.useGraphRefiner;
  }

  if (mode === 'vamsa_wir' || mode === 'vamsa_lineage') {
    payload.mode = mode === 'vamsa_wir' ? 0 : 1;
  }

  callBackend(mode, payload)
    .then(response => {
      console.info(`[${mode.toUpperCase()}] Response received:`, response);

      if (!response.success) {
        console.error(
          `[${mode.toUpperCase()}] Backend returned failure.`,
          response
        );
        return;
      }

      let graphDataString;

      switch (mode) {
        case 'dataflow':
          graphDataString = response.dataflow;
          break;
        case 'lineage':
          graphDataString = response.lineage_predictions;
          break;
        case 'vamsa_wir':
        case 'vamsa_lineage':
          graphDataString = response.vamsa;
          break;
        case 'ast':
          graphDataString = response.ast_graph;
          break;
        case 'labeling':
          graphDataString = response.predictions;
          break;
        default:
          console.error('Unknown graph mode:', mode);
          return;
      }

      if (graphDataString) {
        graphWidget.setNotebookContext(notebookName, content, notebookPanel);

        graphWidget.updateGraphData(
          typeof graphDataString === 'string'
            ? graphDataString
            : JSON.stringify(graphDataString)
        );
      }
    })
    .catch(error => {
      console.error(`[${mode.toUpperCase()}] Network or parsing error:`, error);
    });
};
