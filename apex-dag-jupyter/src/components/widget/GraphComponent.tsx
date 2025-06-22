import React, { useEffect, useState } from "react";
import Graph from "../graph/Graph";
import GraphComponentProps from "../../types/GraphComponentProps";

/**
 * React component for a dataflow graph.
 *
 * @returns The React component
 */
const GraphComponent = ({ eventTarget }: GraphComponentProps): JSX.Element => {
  const [graphData, setGraphData] = useState({ elements: [] });

  useEffect(() => {
    const handler = (event: Event) => {
      const customEvent = event as CustomEvent;
      const newData = customEvent.detail;
      console.log("Received update from parent:", newData);
      setGraphData(newData);
    };

    eventTarget.addEventListener("graph-update", handler);

    return () => {
      eventTarget.removeEventListener("graph-update", handler);
    };
  }, [eventTarget]);

  return (
    <div className={"apexDagPage"}>
      <Graph graphData={graphData} />
    </div>
  );};

  export default GraphComponent;