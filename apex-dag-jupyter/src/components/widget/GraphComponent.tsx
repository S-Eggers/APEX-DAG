import React, { useEffect, useState } from "react";
import Graph from "../graph/Graph";
import GraphComponentProps from "../../types/GraphComponentProps";

/**
 * React component for a dataflow graph.
 *
 * @returns The React component
 */
const GraphComponent = ({ eventTarget, mode = "dataflow" }: GraphComponentProps): JSX.Element => {
  const [graphData, setGraphData] = useState({ elements: [] });

  useEffect(() => {
    const handler = (event: Event) => {
      const customEvent = event as CustomEvent;
      const newData = customEvent.detail;
      console.log("Received update from parent:", newData);
      setGraphData(newData);
    };

    eventTarget.addEventListener("graph-update", handler);
    // The reset-view event is now handled directly by Graph.tsx via eventTarget

    return () => {
      eventTarget.removeEventListener("graph-update", handler);
    };
  }, [eventTarget]);

  return (
    <div className={"apexDagPage"}>
      <Graph graphData={graphData} mode={mode} eventTarget={eventTarget} />
    </div>
  );};

  export default GraphComponent;