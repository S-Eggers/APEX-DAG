node ./apex-dag-jupyter/src/app/module/file_watcher.js ./apex-dag-jupyter/src/app/data_flow_graph.json &
python main.py -e watch -n "$1" &
cd ./apex-dag-jupyter/src/app && npm run dev &

wait