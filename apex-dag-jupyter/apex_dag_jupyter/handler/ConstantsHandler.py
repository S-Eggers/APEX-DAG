import json
import tornado
from jupyter_server.base.handlers import APIHandler
from ApexDAG.sca import AST_NODE_TYPES, AST_EDGE_TYPES, NODE_TYPES, EDGE_TYPES, REVERSE_NODE_TYPES, REVERSE_DOMAIN_EDGE_TYPES

class ConstantsHandler(APIHandler):
    @tornado.web.authenticated
    def get(self):
        ast_nodes = {v: k for k, v in AST_NODE_TYPES.items()}
        ast_edges = {v: k for k, v in AST_EDGE_TYPES.items()}

        df_nodes = {v: k for k, v in NODE_TYPES.items()}
        df_edges = {v: k for k, v in EDGE_TYPES.items()}
        
        lin_nodes = {k: v for k, v in REVERSE_NODE_TYPES.items()}
        lin_edges = {k: v for k, v in REVERSE_DOMAIN_EDGE_TYPES.items()} 

        self.finish(json.dumps({
            "success": True,
            "taxonomy": {
                "ast": {
                    "nodes": ast_nodes,
                    "edges": ast_edges
                },
                "dataflow": {
                    "nodes": df_nodes,
                    "edges": df_edges
                },
                "lineage": {
                    "nodes": lin_nodes,
                    "edges": lin_edges
                },
                "labeling":  {
                    "nodes": lin_nodes,
                    "edges": lin_edges
                },
            }
        }))