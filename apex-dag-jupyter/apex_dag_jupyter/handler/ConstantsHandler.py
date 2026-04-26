import json

import tornado
from ApexDAG.sca import (
    REVERSE_AST_EDGE_TYPES,
    REVERSE_AST_NODE_TYPES,
    REVERSE_DOMAIN_EDGE_TYPES,
    REVERSE_DOMAIN_NODE_TYPES,
    REVERSE_EDGE_TYPES,
    REVERSE_NODE_TYPES,
)
from jupyter_server.base.handlers import APIHandler


class ConstantsHandler(APIHandler):
    @tornado.web.authenticated
    def get(self) -> None:
        self.finish(
            json.dumps(
                {
                    "success": True,
                    "taxonomy": {
                        "ast": {
                            "nodes": REVERSE_AST_NODE_TYPES,
                            "edges": REVERSE_AST_EDGE_TYPES,
                        },
                        "dataflow": {
                            "nodes": REVERSE_NODE_TYPES,
                            "edges": REVERSE_EDGE_TYPES,
                        },
                        "lineage": {
                            "nodes": REVERSE_DOMAIN_NODE_TYPES,
                            "edges": REVERSE_DOMAIN_EDGE_TYPES,
                        },
                        "labeling": {
                            "nodes": REVERSE_DOMAIN_NODE_TYPES,
                            "edges": REVERSE_DOMAIN_EDGE_TYPES,
                        },
                        "vamsa": {
                            "nodes": REVERSE_DOMAIN_NODE_TYPES,
                            "edges": REVERSE_DOMAIN_EDGE_TYPES,
                        },
                    },
                }
            )
        )
