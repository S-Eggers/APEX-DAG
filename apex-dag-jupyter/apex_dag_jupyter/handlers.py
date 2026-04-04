from jupyter_server.utils import url_path_join

from .LineageHandler import LineageHandler
from .DataflowHandler import DataflowHandler


def setup_handlers(web_app, model_instance, jupyter_server_app_config=None):
    host_pattern = ".*$"

    base_url = web_app.settings["base_url"]

    dataflow_pattern = url_path_join(base_url, "apex-dag", "dataflow")
    lineage_pattern = url_path_join(base_url, "apex-dag", "lineage")
    handlers = [
        (
            dataflow_pattern,
            DataflowHandler,
            dict(
                model=model_instance,
                jupyter_server_app_config=jupyter_server_app_config,
            ),
        ),
        (
            lineage_pattern,
            LineageHandler,
            dict(
                model=model_instance,
                jupyter_server_app_config=jupyter_server_app_config,
            ),
        ),
    ]

    web_app.add_handlers(host_pattern, handlers)
