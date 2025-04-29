import { ReactWidget } from '@jupyterlab/ui-components';

import React, { useEffect, useState } from 'react';

import Graph from './graph';



/**
 * React component for a counter.
 *
 * @returns The React component
 */
const GraphComponent = (): JSX.Element => {
  const [graphData, setGraphData] = useState({ elements: [] });
  //setGraphData(require("./data_flow_graph.json"));

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8081");

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "update") {
        const newGraphData = message.data;
        console.log(newGraphData);
        console.log(typeof newGraphData);
        setGraphData(newGraphData);
      }
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    return () => {
      socket.close();
    };
    
  }, []);

  return (
    <>
      <div className={"apex-dag-page"}>
        <Graph graphData={graphData} />
      </div>
    </>
  );};

/**
 * A Counter Lumino Widget that wraps a CounterComponent.
 */
export class GraphWidget extends ReactWidget {
  /**
   * Constructs a new CounterWidget.
   */
  constructor() {
    super();
    this.addClass('jp-react-widget');
  }

  render(): JSX.Element {
    return <GraphComponent />;
  }
}