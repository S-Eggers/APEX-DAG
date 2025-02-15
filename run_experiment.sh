#!/bin/bash

usage() {
    echo "Usage: $0 -n <notebook path> -e <single_dataflow|mine_dataflows|watch|pretrain> [-d]"
    exit 1
}

while getopts ":n:e:c:d" opt; do
    case ${opt} in
        n)
            n_value=$OPTARG
            ;;
        e)
            e_value=$OPTARG
            ;;
        c)
            c_value=$OPTARG
            ;;
        d)
            d_flag=true
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
        if [ "$d_flag" = true ]; then
            python main.py -g -e "data_flow_graph_test" -n "$n_value" -d
        else
            python main.py -g -e "data_flow_graph_test" -n "$n_value"
        fi
        ;;
    watch)
        if [ -z "$n_value" ]; then
            echo "Error: The -n argument is required when e=watch."
            usage
        fi
        if [ "$d_flag" = true ]; then
            python main.py -g -e "watch" -n "$n_value" -d
        else
            python main.py -g -e "watch" -n "$n_value"
        fi
        ;;
    pretrain)
        export TF_USE_LEGACY_KERAS=1
        if [ "$d_flag" = true ]; then
            python main.py -g -e "pretrain" -n "$n_value" -c "$c_value" -d
        else
            python main.py -g -e "pretrain" -n "$n_value" -c "$c_value"
        fi
        ;;
    mine_dataflows)
        if [ "$d_flag" = true ]; then
            python main.py -g -e "mine_dataflows_on_kaggle_dataset" -d
        else
            python main.py -g -e "mine_dataflows_on_kaggle_dataset"
        fi
        ;;
    *)
        echo "Error: Invalid value for -e. Valid options are 'single_dataflow', 'mine_dataflows', or 'watch'."
        usage
        ;;
esac
