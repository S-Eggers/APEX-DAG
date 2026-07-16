from __future__ import annotations

import json

import tornado
from jupyter_server.base.handlers import APIHandler


class ModelsHandler(APIHandler):
    def initialize(self, registry: object = None, jupyter_server_app_config: dict | None = None) -> None:
        self.registry = registry
        self.jupyter_server_app_config = jupyter_server_app_config

    @tornado.web.authenticated
    def get(self) -> None:
        variants = self.registry.list_variants() if self.registry is not None else []
        self.finish(json.dumps({"success": True, "variants": variants}))

    def data_received(self, chunk: bytes) -> None:
        pass
