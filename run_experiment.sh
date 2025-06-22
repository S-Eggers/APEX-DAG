#!/bin/bash
NOTEBOOK="./data/raw/demo.ipynb"
WINDOW=1
GREEDY=false
SAVE_PREV=false
EXPERIMENT="data_flow_graph_test"
DRAW=false
CHECKPOINT_PATH=""

usage() {
    echo "Usage: $0 -e <experiment> [options]"
    echo ""
    echo "Available experiments:"
    echo "  - ast_graph_test"
    echo "  - analyze_kaggle_dataset"
    echo "  - data_flow_graph_test"
    echo "  - mine_dataflows_on_kaggle_dataset"
    echo "  - watch"
    echo "  - pretrain"
    echo ""
    echo "Options:"
    echo "  -n  Notebook filename (default: $NOTEBOOK)"
    echo "  -w  Cell window size (default: $WINDOW)"
    echo "  -g  Enable greedy mode"
    echo "  -s  Save previous graph for debugging"
    echo "  -e  Experiment to run (default: $EXPERIMENT)"
    echo "  -d  Draw AST/Dataflow Graphs"
    echo "  -c  Checkpoint path"
    exit 1
}

while getopts "n:w:gse:dc:" opt; do
    case ${opt} in
        n ) NOTEBOOK=$OPTARG ;;
        w ) WINDOW=$OPTARG ;;
        g ) GREEDY=true ;;
        s ) SAVE_PREV=true ;;
        e ) EXPERIMENT=$OPTARG ;;
        d ) DRAW=true ;;
        c ) CHECKPOINT_PATH=$OPTARG ;;
        * ) usage ;;
    esac
done

case $EXPERIMENT in
    ast_graph_test|analyze_kaggle_dataset|data_flow_graph_test|mine_dataflows_on_kaggle_dataset|watch|pretrain)
        ;; # Valid
    *)
        echo "Invalid experiment: $EXPERIMENT"
        usage
        ;;
esac

CMD="python main.py -e $EXPERIMENT"

CMD+=" -n $NOTEBOOK"
CMD+=" -w $WINDOW"
[ "$GREEDY" = true ] && CMD+=" -g"
[ "$SAVE_PREV" = true ] && CMD+=" -s"
[ "$DRAW" = true ] && CMD+=" -d"
[ -n "$CHECKPOINT_PATH" ] && CMD+=" -c $CHECKPOINT_PATH"

echo "Running: $CMD"
eval $CMD
