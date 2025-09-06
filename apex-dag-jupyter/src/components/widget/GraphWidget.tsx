import { ReactWidget } from "@jupyterlab/ui-components";
import React from "react";
import GraphComponent from "./GraphComponent"



/**
 * A Counter Lumino Widget that wraps a CounterComponent.
 */
export class GraphWidget extends ReactWidget {
  private eventTarget: EventTarget;
  private mode: "dataflow" | "lineage";

  /**
   * Constructs a new DFG widget.
   */
  constructor(mode: "dataflow" | "lineage" = "dataflow") {
    super();
    this.mode = mode;
    this.addClass("jp-react-widget");
    this.eventTarget = new EventTarget();
  }

  render(): JSX.Element {
    return <GraphComponent eventTarget={this.eventTarget} mode={this.mode} />;
  }

  updateGraphData(graphData: any): void {
    console.log("Updating graph")
    const event = new CustomEvent("graph-update", { detail: JSON.parse(graphData) });
    this.eventTarget.dispatchEvent(event);
  }

  resetView(): void {
    console.log("Resetting graph view");
    const event = new CustomEvent("reset-view");
    this.eventTarget.dispatchEvent(event);
  }
}