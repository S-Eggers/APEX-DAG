import { ReactWidget } from "@jupyterlab/ui-components";
import React from "react";
import GraphComponent from "./GraphComponent"



/**
 * A Counter Lumino Widget that wraps a CounterComponent.
 */
export class GraphWidget extends ReactWidget {
  private eventTarget: EventTarget;
  /**
   * Constructs a new DFG widget.
   */
  constructor() {
    super();
    this.addClass("jp-react-widget");
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