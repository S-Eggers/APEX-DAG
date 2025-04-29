import { ReactWidget } from '@jupyterlab/ui-components';

import React, { useEffect, useState } from 'react';

import Graph from './graph';

interface GraphComponentProps {
  eventTarget: EventTarget;
}

/**
 * React component for a counter.
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
  private eventTarget: EventTarget;
  /**
   * Constructs a new CounterWidget.
   */
  constructor() {
    super();
    this.addClass('jp-react-widget');
    this.eventTarget = new EventTarget();
  }

  render(): JSX.Element {
    return <GraphComponent eventTarget={this.eventTarget} />;
  }

  updateGraphData(graphData: any): void {
    console.log("Updating graph")
    const event = new CustomEvent("graph-update", { detail: JSON.parse(graphData) });
    this.eventTarget.dispatchEvent(event);
  }
}