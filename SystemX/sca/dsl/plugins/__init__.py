from SystemX.sca.dsl.plugins.airflow import AirflowPlugin
from SystemX.sca.dsl.plugins.beam import BeamPlugin
from SystemX.sca.dsl.plugins.pipes import PipesPlugin

DEFAULT_PLUGINS = [BeamPlugin, AirflowPlugin, PipesPlugin]

__all__ = ["DEFAULT_PLUGINS", "AirflowPlugin", "BeamPlugin", "PipesPlugin"]
