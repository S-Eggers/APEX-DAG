from jupyter_server.utils import url_path_join

from .LineageHandler import LineageHandler
from .DataflowHandler import DataflowHandler
from .ASTHandler import ASTHandler
from .EnvironmentHandler import EnvironmentHandler

def setup_handlers(web_app, model_instance, jupyter_server_app_config=None):
    host_pattern = ".*$"
    base_url = web_app.settings["base_url"]

    handlers = [
        (
            url_path_join(base_url, "apex-dag", "dataflow"),
            DataflowHandler,
            dict(jupyter_server_app_config=jupyter_server_app_config),
        ),
        (
            url_path_join(base_url, "apex-dag", "lineage"),
            LineageHandler,
            dict(model=model_instance, jupyter_server_app_config=jupyter_server_app_config),
        ),
        (
            url_path_join(base_url, "apex-dag", "ast"),
            ASTHandler,
            dict(jupyter_server_app_config=jupyter_server_app_config), 
        ),
        (
            url_path_join(base_url, "apex-dag", "environment"),
            EnvironmentHandler,
            dict(jupyter_server_app_config=jupyter_server_app_config), 
        )
    ]

    web_app.add_handlers(host_pattern, handlers)