#!/bin/bash

usage() {
    echo "Usage: $0 -n <notebook path> -e <single_dataflow|mine_dataflows|watch>"
    exit 1
}

while getopts ":n:e:" opt; do
    case ${opt} in
        n)
            n_value=$OPTARG
            ;;
        e)
            e_value=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

if [ -z "$e_value" ]; then
    echo "Error: The -e argument is required."
    usage
fi

case $e_value in
    single_dataflow)
        if [ -z "$n_value" ]; then
            echo "Error: The -n argument is required when e=single_dataflow."
            usage
        fi
        python main.py -g -e "data_flow_graph_test" -n "$n_value"
        ;;
    watch)
        if [ -z "$n_value" ]; then
            echo "Error: The -n argument is required when e=watch."
            usage
        fi
        python main.py -g -e "watch" -n "$n_value"
        ;;
    mine_dataflows)
        python main.py -g -e "mine_dataflows_on_kaggle_dataset"
        ;;
    *)
        echo "Error: Invalid value for -e. Valid options are 'single_dataflow' or 'mine_dataflows'."
        usage
        ;;
esac