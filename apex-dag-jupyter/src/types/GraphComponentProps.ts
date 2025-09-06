interface GraphComponentProps {
    eventTarget: EventTarget;
    mode?: "dataflow" | "lineage";
    resetTrigger?: boolean;
}

export default GraphComponentProps;