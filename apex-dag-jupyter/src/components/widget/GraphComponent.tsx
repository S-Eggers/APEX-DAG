import React from 'react';
import Graph from '../graph/Graph';
import { GraphMode } from '../../types/GraphTypes';
import { useGraphTaxonomy } from '../../hooks/useGraphTaxonomy';

interface GraphComponentProps {
  graphData: { elements: any[] };
  mode: GraphMode;
  resetTrigger: number;
  notebookName: string;
  notebookCode: string;
}

const GraphComponent: React.FC<GraphComponentProps> = ({
  graphData,
  mode,
  resetTrigger,
  notebookName,
  notebookCode
}) => {
  const taxonomy = useGraphTaxonomy(mode);

  if (!taxonomy.isLoaded) {
    return (
      <div className="flex items-center justify-center h-full w-full bg-[#171717] text-gray-400 font-mono text-sm">
        Loading domain taxonomy from backend...
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
      />
    </div>
  );
};

export default GraphComponent;
