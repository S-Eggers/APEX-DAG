import { ICodeCellModel } from '@jupyterlab/cells';
import { CellList } from '@jupyterlab/notebook';

export const getNotebookCode = (cells: CellList, greedy: boolean = true): string => {
  if (greedy) {
    let content: string = "";
    for (let i = 0; i < cells.length; i++) {
      const cell = cells.get(i);
      if (cell.type === "code") {
        const codeCell = cell as ICodeCellModel;
        content += codeCell.toJSON().source + '\n';
      }
    }
    return content;
  } else {
    const executedCodeCells = [];
    for (let i = 0; i < cells.length; i++) {
      const cell = cells.get(i);
      if (cell.type === "code") {
        const codeModel = cell as ICodeCellModel;
        if (codeModel.executionCount && codeModel.executionCount > 0) {
          executedCodeCells.push(codeModel);
        }
      }
    }
    executedCodeCells.sort((a, b) => {
      return a.executionCount! - b.executionCount!;
    });
    const content = executedCodeCells.map(model => model.toJSON().source  + "\n").join("");

    return content;
  }
};