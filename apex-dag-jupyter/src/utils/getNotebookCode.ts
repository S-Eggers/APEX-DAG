import { ICodeCellModel } from '@jupyterlab/cells';
import { CellList } from '@jupyterlab/notebook';
import { ExtractedCell } from '../types/NotebookTypes';

export const getNotebookCode = (
  cells: CellList,
  greedy: boolean = true
): ExtractedCell[] => {
  const extractedCells: ExtractedCell[] = [];

  if (greedy) {
    for (let i = 0; i < cells.length; i++) {
      const cell = cells.get(i);

      if (cell.type === 'code') {
        const sourceCode = cell.sharedModel.getSource();

        if (sourceCode.trim().length > 0) {
          extractedCells.push({
            cell_id: cell.id,
            source: sourceCode
          });
        }
      }
    }
  } else {
    const executedCodeCells: ICodeCellModel[] = [];

    for (let i = 0; i < cells.length; i++) {
      const cell = cells.get(i);
      if (cell.type === 'code') {
        const codeModel = cell as ICodeCellModel;
        if (codeModel.executionCount && codeModel.executionCount > 0) {
          executedCodeCells.push(codeModel);
        }
      }
    }

    executedCodeCells.sort((a, b) => a.executionCount! - b.executionCount!);

    for (const codeModel of executedCodeCells) {
      const sourceCode = codeModel.sharedModel.getSource();

      if (sourceCode.trim().length > 0) {
        extractedCells.push({
          cell_id: codeModel.id,
          source: sourceCode
        });
      }
    }
  }

  return extractedCells;
};
