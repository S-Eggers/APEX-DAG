import React from 'react';
import Graph from '../graph/Graph';
import { GraphComponentProps } from '../../types/GraphTypes';
import { useGraphTaxonomy } from '../../hooks/useGraphTaxonomy';

const GraphComponent: React.FC<GraphComponentProps> = ({
  graphData,
  mode,
  resetTrigger,
  notebookName,
  notebookCode,
  onLocateCell
}) => {
  const taxonomy = useGraphTaxonomy(mode);

  if (!taxonomy.isLoaded) {
    return (
      <div className="flex items-center justify-center h-full w-full bg-[#171717] text-gray-400 font-mono text-sm">
        Loading...
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full max-h-full w-full max-w-full bg-[#171717]">
      <Graph
        graphData={graphData}
        mode={mode}
        resetTrigger={resetTrigger}
        taxonomy={taxonomy}
        notebookName={notebookName}
        notebookCode={notebookCode}
        onLocateCell={onLocateCell}
      />
    </div>
  );
};

export default GraphComponent;
