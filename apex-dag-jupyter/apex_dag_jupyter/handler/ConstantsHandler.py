import json

import tornado.web
from ApexDAG.sca.constants import AST_EDGES, AST_NODES, DATAFLOW_EDGES, DATAFLOW_NODES, DOMAIN_EDGES, DOMAIN_NODES
from jupyter_server.base.handlers import APIHandler


class ConstantsHandler(APIHandler):
    @tornado.web.authenticated
    def get(self) -> None:
        response_payload = {
            "success": True,
            "taxonomy": {
                "ast": {
                    "nodes": AST_NODES,
                    "edges": AST_EDGES,
                },
                "dataflow": {
                    "nodes": DATAFLOW_NODES,
                    "edges": DATAFLOW_EDGES,
                },
                "lineage": {
                    "nodes": DOMAIN_NODES,
                    "edges": DOMAIN_EDGES,
                },
                "labeling": {
                    "nodes": DOMAIN_NODES,
                    "edges": DOMAIN_EDGES,
                },
                "vamsa": {
                    "nodes": DOMAIN_NODES,
                    "edges": DOMAIN_EDGES,
                },
            },
        }

        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps(response_payload))
