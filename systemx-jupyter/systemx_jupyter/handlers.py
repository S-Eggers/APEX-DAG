import tornado
from jupyter_server.utils import url_path_join

from .handler.ASTHandler import ASTHandler
from .handler.ConstantsHandler import ConstantsHandler
from .handler.DataflowHandler import DataflowHandler
from .handler.DatasetsHandler import DatasetsHandler
from .handler.EnvironmentHandler import EnvironmentHandler
from .handler.ExecutionStateHandler import ExecutionStateHandler
from .handler.ExecutionTraceHandler import ExecutionTraceHandler
from .handler.ExecutionTraceLoadHandler import ExecutionTraceLoadHandler
from .handler.ExecutionTraceSaveHandler import ExecutionTraceSaveHandler
from .handler.LabelingFlagHandler import LabelingFlagHandler
from .handler.LabelingGenerateHandler import LabelingGenerateHandler
from .handler.LabelingNextHandler import LabelingNextHandler
from .handler.LabelingPredictHandler import LabelingPredictHandler
from .handler.LabelingRefineHandler import LabelingRefineHandler
from .handler.LabelingSaveHandler import LabelingSaveHandler
from .handler.LabelingTrainHandler import LabelingTrainHandler
from .handler.LeakageHandler import LeakageHandler
from .handler.LeakageSaveHandler import LeakageSaveHandler
from .handler.LineageHandler import LineageHandler
from .handler.LineageTupleSaveHandler import LineageTupleSaveHandler
from .handler.ModelsHandler import ModelsHandler
from .handler.VamsaHandler import VamsaHandler


def setup_handlers(
    web_app: tornado.web.Application,
    models: dict,
    jupyter_server_app_config: dict | None = None,
    registry: object | None = None,
) -> None:
    host_pattern = ".*$"
    base_url = web_app.settings["base_url"]

    handlers = [
        (
            url_path_join(base_url, "systemx", "dataflow"),
            DataflowHandler,
            dict(jupyter_server_app_config=jupyter_server_app_config),
        ),
        (
            url_path_join(base_url, "systemx", "lineage"),
            LineageHandler,
            dict(
                models=models,
                jupyter_server_app_config=jupyter_server_app_config,
            ),
        ),
        (
            url_path_join(base_url, "systemx", "ast"),
            ASTHandler,
            dict(jupyter_server_app_config=jupyter_server_app_config),
        ),
        (
            url_path_join(base_url, "systemx", "environment"),
            EnvironmentHandler,
            dict(jupyter_server_app_config=jupyter_server_app_config),
        ),
        (
            url_path_join(base_url, "systemx", "execution-state"),
            ExecutionStateHandler,
            dict(
                models=models,
                jupyter_server_app_config=jupyter_server_app_config,
            ),
        ),
        (
            url_path_join(base_url, "systemx", "execution-trace", "analyze"),
            ExecutionTraceHandler,
            dict(
                models=models,
                jupyter_server_app_config=jupyter_server_app_config,
            ),
        ),
        (url_path_join(base_url, "systemx", "execution-trace", "save"), ExecutionTraceSaveHandler),
        (url_path_join(base_url, "systemx", "execution-trace", "load"), ExecutionTraceLoadHandler),
        (url_path_join(base_url, "systemx", "labeling", "save"), LabelingSaveHandler),
        (url_path_join(base_url, "systemx", "labeling", "refine"), LabelingRefineHandler),
        (
            url_path_join(base_url, "systemx", "labeling", "predict"),
            LabelingPredictHandler,
            dict(
                models=models,
                jupyter_server_app_config=jupyter_server_app_config,
            ),
        ),
        (
            url_path_join(base_url, "systemx", "leakage"),
            LeakageHandler,
            dict(
                models=models,
                jupyter_server_app_config=jupyter_server_app_config,
            ),
        ),
        (url_path_join(base_url, "systemx", "leakage", "save"), LeakageSaveHandler),
        (url_path_join(base_url, "systemx", "lineage", "save"), LineageTupleSaveHandler),
        (
            url_path_join(base_url, "systemx", "labeling", "generate"),
            LabelingGenerateHandler,
            {},
        ),
        (
            url_path_join(base_url, "systemx", "labeling", "next"),
            LabelingNextHandler,
            {},
        ),
        (
            url_path_join(base_url, "systemx", "labeling", "flag"),
            LabelingFlagHandler,
            {},
        ),
        (
            url_path_join(base_url, "systemx", "labeling", "train"),
            LabelingTrainHandler,
            dict(registry=registry, jupyter_server_app_config=jupyter_server_app_config),
        ),
        (
            url_path_join(base_url, "systemx", "models"),
            ModelsHandler,
            dict(registry=registry, jupyter_server_app_config=jupyter_server_app_config),
        ),
        (url_path_join(base_url, "systemx", "constants"), ConstantsHandler, {}),
        (url_path_join(base_url, "systemx", "datasets"), DatasetsHandler, {}),
        (url_path_join(base_url, "systemx", "vamsa/wir"), VamsaHandler, {}),
        (url_path_join(base_url, "systemx", "vamsa/lineage"), VamsaHandler, {}),
    ]

    web_app.add_handlers(host_pattern, handlers)
