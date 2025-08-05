import { NotebookPanel } from '@jupyterlab/notebook';
import { GraphWidget } from '../components/widget/GraphWidget';
import callBackend from './callBackend';
import { getNotebookCode } from './getNotebookCode';

const updateWidget = (
  graphWidget: GraphWidget | null,
  replaceDataflowInUDFs: boolean,
  highlightRelevantSubgraphs: boolean,
  greedyNotebookExtraction: boolean,
  notebookPanel: NotebookPanel
) => {
  if (!graphWidget) {
    return;
  }
  const cells = notebookPanel.content.model?.cells;
  if (cells) {
    let content = getNotebookCode(cells, greedyNotebookExtraction)
    console.debug('Notebook content\n' + content);
    callBackend('lineage', {
      code: content,
      replaceDataflowInUDFs,
      highlightRelevantSubgraphs
    })
      .then(response => {
        console.info('Dataflow received from server:', response);
        if (graphWidget && response.success) {
          graphWidget.updateGraphData(response.lineage_predictions);
        }
      })
      .catch(error => {
        console.error('Error sending dataflow:', error);
      });
  }
};

export default updateWidget;
