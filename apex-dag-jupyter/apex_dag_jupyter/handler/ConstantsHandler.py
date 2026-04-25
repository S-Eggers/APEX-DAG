import json
import tornado
from jupyter_server.base.handlers import APIHandler
from ApexDAG.sca import (
    REVERSE_AST_NODE_TYPES, 
    REVERSE_AST_EDGE_TYPES, 
    REVERSE_NODE_TYPES,
    REVERSE_EDGE_TYPES,
    REVERSE_DOMAIN_NODE_TYPES,
    REVERSE_DOMAIN_EDGE_TYPES
)

class ConstantsHandler(APIHandler):
    @tornado.web.authenticated
    def get(self):
        self.finish(json.dumps({
            "success": True,
            "taxonomy": {
                "ast": {
                    "nodes": REVERSE_AST_NODE_TYPES,
                    "edges": REVERSE_AST_EDGE_TYPES
                },
                "dataflow": {
                    "nodes": REVERSE_NODE_TYPES,
                    "edges": REVERSE_EDGE_TYPES
                },
                "lineage": {
                    "nodes": REVERSE_DOMAIN_NODE_TYPES,
                    "edges": REVERSE_DOMAIN_EDGE_TYPES
                },
                "labeling":  {
                    "nodes": REVERSE_DOMAIN_NODE_TYPES,
                    "edges": REVERSE_DOMAIN_EDGE_TYPES
                },
                "vamsa": {
                    "nodes": REVERSE_DOMAIN_NODE_TYPES,
                    "edges": REVERSE_DOMAIN_EDGE_TYPES
                }
            }
        }))