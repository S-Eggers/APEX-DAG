import json

import tornado.web
from SystemX.sca.constants import AST_EDGES, AST_NODES, COMPUTE_HUBS, DATAFLOW_EDGES, DATAFLOW_NODES, DOMAIN_EDGES, DOMAIN_NODES, WIR_EDGES
from SystemX.sca.leakage import LEAKAGE_GOLD_TAXONOMY
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
                "lineage": {"nodes": DOMAIN_NODES, "edges": DOMAIN_EDGES, "domain_nodes": DOMAIN_NODES},
                "labeling": {"nodes": DATAFLOW_NODES, "edges": DATAFLOW_EDGES, "hubs": DOMAIN_EDGES, "hub_types": list(COMPUTE_HUBS), "domain_nodes": DOMAIN_NODES},
                "leakage": {"nodes": DATAFLOW_NODES, "edges": DATAFLOW_EDGES, "hubs": DOMAIN_EDGES, "hub_types": list(COMPUTE_HUBS), "domain_nodes": DOMAIN_NODES, "gold": LEAKAGE_GOLD_TAXONOMY},
                "vamsa_wir": {
                    "nodes": DOMAIN_NODES,
                    "edges": WIR_EDGES,
                },
                "vamsa_lineage": {
                    "nodes": DOMAIN_NODES,
                    "edges": DOMAIN_EDGES,
                    "domain_nodes": DOMAIN_NODES,
                },
            },
        }

        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps(response_payload))
