import { NotebookPanel } from '@jupyterlab/notebook';
import { GraphWidget } from '../components/widget/GraphWidget';
import callBackend from './callBackend';
import { getNotebookCode } from './getNotebookCode';
import { GraphMode } from '../types/GraphTypes';
import { AppSettings } from '../settings/AppSettings';

interface BackendResponse {
  success: boolean;
  message?: string;
  backend_error?: string;
  dataflow?: string | object;
  lineage_predictions?: string | object;
  vamsa?: string | object;
  ast_graph?: string | object;
  predictions?: string | object;
}

export const updateGraphWidget = (
  graphWidget: GraphWidget | null,
  notebookPanel: NotebookPanel,
  mode: GraphMode,
  settings: AppSettings
): Promise<void> => {
  if (!graphWidget) return Promise.resolve();

  graphWidget.setLabelingConfig({
    rawDatasetPath: settings.rawDatasetPath,
    defaultFetchMode: settings.defaultFetchMode,
    defaultFlagType: settings.defaultFlagType
  });
  graphWidget.setNNBackend(settings.nnBackend);
  graphWidget.setFeaturePreset(settings.featurePreset);
  graphWidget.setModelVariant(settings.modelVariant);
  graphWidget.setExplainFeatureImportance(settings.explainFeatureImportance);

  const cells = notebookPanel.content.model?.cells;
  if (!cells) {
    console.warn('No notebook cells available for extraction.');
    return Promise.resolve();
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

  const payload: Record<string, unknown> = {
    code: legacyConcatenatedCode,
    cells: content,
    replaceDataflowInUDFs: settings.replaceDataflowInUDFs,
    highlightRelevantSubgraphs: settings.highlightRelevantSubgraphs,
    detectDsl: settings.detectDsl
  };

  const isLabelingLike = mode === 'labeling' || mode === 'leakage';

  const isLineageLike = mode === 'lineage' || mode === 'lineage_annotation';

  if (isLineageLike || isLabelingLike) {
    payload.nnBackend = settings.nnBackend;
    payload.featurePreset = settings.featurePreset;
    payload.modelVariant = settings.modelVariant;
    console.info(
      `[SystemX][${mode.toUpperCase()}] nnBackend = ${settings.nnBackend}, featurePreset = ${settings.featurePreset}, modelVariant = ${settings.modelVariant || '(base)'}`
    );
  }

  if (isLabelingLike) {
    payload.filename = notebookPanel.context.path;
    payload.useGraphRefiner = settings.useGraphRefiner;
    payload.base_path = settings.rawDatasetPath;
    payload.usePredictionForAnnotation = settings.usePredictionForAnnotation;
    payload.explainFeatureImportance = settings.explainFeatureImportance;
  }

  if (mode === 'vamsa_wir' || mode === 'vamsa_lineage') {
    payload.mode = mode === 'vamsa_wir' ? 0 : 1;
  }

  const endpoint = mode === 'lineage_annotation' ? 'lineage' : mode;

  return callBackend<BackendResponse>(endpoint, payload)
    .then(response => {
      console.info(`[${mode.toUpperCase()}] Response received:`, response);

      graphWidget.setBackendError(null);

      if (!response.success) {
        const msg =
          response.backend_error ||
          response.message ||
          'Backend returned an error.';
        console.error(
          `[${mode.toUpperCase()}] Backend error: ${msg}`,
          response
        );
        graphWidget.setBackendError(msg);
        return;
      }

      let graphDataString: string | object | undefined;

      switch (mode) {
        case 'dataflow':
          graphDataString = response.dataflow;
          break;
        case 'lineage':
        case 'lineage_annotation':
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
        case 'leakage':
          graphDataString = response.predictions || response.dataflow;
          break;
        default:
          console.error(`Unknown graph mode: ${mode}`);
          return;
      }

      if (graphDataString) {
        graphWidget.setNotebookContext(notebookName, content, notebookPanel);

        graphWidget.updateGraphData(
          typeof graphDataString === 'string'
            ? graphDataString
            : JSON.stringify(graphDataString)
        );
      } else {
        const msg = `Backend returned no ${mode} data.`;
        console.error(`[${mode.toUpperCase()}] ${msg}`, response);
        graphWidget.setBackendError(msg);
      }
    })
    .catch(error => {
      console.error(`[${mode.toUpperCase()}] Network or parsing error:`, error);
      const msg =
        error instanceof Error
          ? error.message
          : 'Failed to reach the SystemX backend.';
      graphWidget.setBackendError(msg);
    });
};
