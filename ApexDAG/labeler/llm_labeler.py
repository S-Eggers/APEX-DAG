from ApexDAG.label_notebooks.online_labeler import OnlineGraphLabeler as GraphLabeler
from ApexDAG.label_notebooks.utils import Config
from ApexDAG.labeler.edge_labeler import EdgeLabeler
from ApexDAG.sca import (
    DOMAIN_EDGE_TYPES,
)
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph


class LLMLabeler(EdgeLabeler):
    def apply_labels(self, graph: PythonDataFlowGraph) -> None:
        config = Config(
            model_name="gemini-1.5-flash",
            max_tokens=0,
            max_depth=4,
            llm_provider="google",
            retry_attempts=2,
            retry_delay=0,
            success_delay=0,
            sleep_interval=0,
            max_workers=16
        )
        labeler = GraphLabeler(config, graph.get_graph(), graph.get_code())
        labeled_graph, _ = labeler.label_graph()
        attrs_to_set = {}
        for u, v, key, data in labeled_graph.edges(data=True, keys=True):
            if "domain_label" in data and data["domain_label"] in DOMAIN_EDGE_TYPES:
                attrs_to_set[(u, v, key)] = DOMAIN_EDGE_TYPES[data["domain_label"].upper()]
            else:
                attrs_to_set[(u, v, key)] = DOMAIN_EDGE_TYPES["NOT_INTERESTING"]


        graph.set_domain_label(attrs_to_set, name="predicted_label")
