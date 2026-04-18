import React from 'react';
import Graph from '../graph/Graph';

interface GraphComponentProps {
  graphData: { elements: any[] };
  mode: 'dataflow' | 'lineage';
  resetTrigger: number;
}

const GraphComponent: React.FC<GraphComponentProps> = ({
  graphData,
  mode,
  resetTrigger
}) => {
  return (
    // Replaced .apexDagPage with Tailwind.
    // Notice h-full instead of h-screen (100vh) to prevent Jupyter scrollbar bugs.
    <div className="flex flex-col h-full max-h-full w-full max-w-full bg-[#171717]">
      <Graph graphData={graphData} mode={mode} resetTrigger={resetTrigger} />
    </div>
  );
};

export default GraphComponent;
