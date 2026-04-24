from jupyter_server.utils import url_path_join

from .handler.LineageHandler import LineageHandler
from .handler.DataflowHandler import DataflowHandler
from .handler.ASTHandler import ASTHandler
from .handler.EnvironmentHandler import EnvironmentHandler
from .handler.LabelingSaveHandler import LabelingSaveHandler#
from .handler.LabelingPredictHandler import LabelingPredictHandler
from .handler.LabelingGenerateHandler import LabelingGenerateHandler
from .handler.ConstantsHandler import ConstantsHandler
from .handler.VamsaHandler import VamsaHandler


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
        ),
        (
            url_path_join(base_url, "apex-dag", "labeling", "save"),
            LabelingSaveHandler
        ),
        (
            url_path_join(base_url, "apex-dag", "labeling", "predict"),
            LabelingPredictHandler,
            dict(model=model_instance),
        ),
        (
            url_path_join(base_url, "apex-dag", "labeling", "generate"),
            LabelingGenerateHandler,
            {}
        ),
        (
            url_path_join(base_url, "apex-dag", "constants"),
            ConstantsHandler,
            {}
        ),
        (
            url_path_join(base_url, "apex-dag", "vamsa"),
            VamsaHandler,
            {}
        )
    ]

    web_app.add_handlers(host_pattern, handlers)