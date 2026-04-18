import React from 'react';
import Graph from '../graph/Graph';
import { GraphMode } from '../../types/GraphTypes';

interface GraphComponentProps {
  graphData: { elements: any[] };
  mode: GraphMode;
  resetTrigger: number;
}

const GraphComponent: React.FC<GraphComponentProps> = ({
  graphData,
  mode,
  resetTrigger
}) => {
  return (
    <div className="flex flex-col h-full max-h-full w-full max-w-full bg-[#171717]">
      <Graph graphData={graphData} mode={mode} resetTrigger={resetTrigger} />
    </div>
  );
};

export default GraphComponent;
