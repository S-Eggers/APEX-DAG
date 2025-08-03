import { NotebookPanel } from '@jupyterlab/notebook';
import { ICodeCellModel } from '@jupyterlab/cells';

import { GraphWidget } from '../components/widget/GraphWidget';
import callBackend from './callBackend';

const updateWidget = (
  graphWidget: GraphWidget | null,
  replaceDataflowInUDFs: boolean,
  notebookPanel: NotebookPanel
) => {
  if (!graphWidget) {
    return;
  }
  const cells = notebookPanel.content.model?.cells;
  if (cells) {
    let content: string = '';
    for (let i = 0; i < cells.length; i++) {
      const cell = cells.get(i);
      if (cell.type === 'code') {
        const codeCell = cell as ICodeCellModel;
        content += codeCell.toJSON().source + '\n';
      }
    }
    console.debug('Notebook content\n' + content);
    callBackend('lineage', { code: content, replaceDataflowInUDFs })
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
