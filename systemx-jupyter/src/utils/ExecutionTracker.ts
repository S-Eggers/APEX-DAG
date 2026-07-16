import { Cell, ICodeCellModel } from '@jupyterlab/cells';

const sourceHashAtExecution = new Map<string, string>();

export const hashSource = (source: string): string => {
  let hash = 5381;
  for (let i = 0; i < source.length; i++) {
    hash = ((hash << 5) + hash + source.charCodeAt(i)) | 0;
  }
  return hash.toString(36);
};

export const recordExecution = (cell: Cell | undefined): void => {
  const model = cell?.model;
  if (!model || model.type !== 'code') return;
  sourceHashAtExecution.set(model.id, hashSource(model.sharedModel.getSource()));
};

export const getDirtyState = (model: ICodeCellModel): boolean | null => {
  const recorded = sourceHashAtExecution.get(model.id);
  if (recorded === undefined) return null;
  return recorded !== hashSource(model.sharedModel.getSource());
};
