import { ICodeCellModel } from '@jupyterlab/cells';
import { CellList } from '@jupyterlab/notebook';
import { ExtractedCell } from '../types/NotebookTypes';
import { getDirtyState } from './ExecutionTracker';

const toExtractedCell = (
  codeModel: ICodeCellModel,
  documentIndex: number
): ExtractedCell => ({
  cell_id: codeModel.id,
  source: codeModel.sharedModel.getSource(),
  execution_count: codeModel.executionCount ?? null,
  document_index: documentIndex,
  is_dirty: getDirtyState(codeModel)
});

export const getNotebookCode = (
  cells: CellList,
  greedy: boolean = true
): ExtractedCell[] => {
  const extractedCells: ExtractedCell[] = [];

  if (greedy) {
    for (let i = 0; i < cells.length; i++) {
      const cell = cells.get(i);

      if (cell.type === 'code') {
        const codeModel = cell as ICodeCellModel;

        if (codeModel.sharedModel.getSource().trim().length > 0) {
          extractedCells.push(toExtractedCell(codeModel, i));
        }
      }
    }
  } else {
    const executedCodeCells: { model: ICodeCellModel; index: number }[] = [];

    for (let i = 0; i < cells.length; i++) {
      const cell = cells.get(i);
      if (cell.type === 'code') {
        const codeModel = cell as ICodeCellModel;
        if (codeModel.executionCount && codeModel.executionCount > 0) {
          executedCodeCells.push({ model: codeModel, index: i });
        }
      }
    }

    executedCodeCells.sort(
      (a, b) => a.model.executionCount! - b.model.executionCount!
    );

    for (const { model, index } of executedCodeCells) {
      if (model.sharedModel.getSource().trim().length > 0) {
        extractedCells.push(toExtractedCell(model, index));
      }
    }
  }

  return extractedCells;
};
