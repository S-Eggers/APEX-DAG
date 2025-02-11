from enum import Enum

def generate_message(node_id_source: str, node_id_target: str, code_edge:str, subgraph_context: str) -> list:
    """
    Generates a message for labeling edges in a dataflow graph.

    Args:
        node_id_source (str): The unique identifier of the source node.
        node_id_target (str): The unique identifier of the target node.
        code_edge (str): The code that connects the source and target nodes.
        subgraph_context (str): The context of the subgraph.

    Returns:
        list: A properly formatted message for the LLM.
    """


    domain_labels = """
    MODEL_TRAIN: Model Train - Training machine learning models
    MODEL_EVALUATION: Model Evaluation - Evaluating the performance of models
    HYPERPARAMETER_TUNING: Hyperparameter Tuning - Optimizing model parameters
    DATA_EXPORT: Data Export - Exporting data to external storage, saving the data, exporting the data/models
    DATA_IMPORT_EXTRACTION: Data Import / Extraction - Importing or extracting data from external sources (not variables)
    DATA_TRANSFORM: Data Transform - Transforming data into a suitable format
    EDA: EDA - Exploratory Data Analysis
    ENVIRONMENT: Environment - Setting up or managing the environment: i.e. installing packages/modules, setting up virtual environments
    NOT_INTERESTING: Not Interesting - Not relevant to the analysis
    """

    json_structure = """
    {
      "source_id": "...",  # Unique identifier for the source node
      "target_id": "...",  # Unique identifier for the target node
      "code": "...", #
      "edge_type": "...",  # Type of the edge
      "domain_label": "..."  # Domain-specific label for the edge
    }
    """

    message_content = (
        f"You are an expert in labeling dataflow graphs of Python programs, specifically machine learning. "
        f"Please label the following edge within the provided subgraph context. "
        f"The edge of interest is indicated, and the subgraph provides additional context. "
        f"Use one of the following domain labels:\n{domain_labels}\n\n"
        f"Here is the edge of interest:\n"
        f"  ID: '{node_id_source} ---{code_edge}--->{node_id_target}'\n"
        f"\nHere is the subgraph context:\n{subgraph_context}\n\n"
        f"Please ensure that your response is formatted as valid JSON with the following structure:\n{json_structure}\n\n"
        f"Provide the domain label based on the given node's context."
    )

    return [{"role": "user", "content": message_content}]


class DomainLabel(str, Enum):
    MODEL_TRAIN = "Model Train - Training machine learning models"
    MODEL_EVALUATION = "Model Evaluation - Evaluating the performance of models"
    HYPERPARAMETER_TUNING = "Hyperparameter Tuning - Optimizing model parameters"
    DATA_EXPORT = "Data Export - Exporting data to external storage (not to a local variable)"
    DATA_EXTRACTION = "Data Extraction - Importing or extracting data from external data sources. (like oad from zip, load from disk load from sql, load from url- or any other)"
    DATA_TRANSFORM = "Data Transform - Transforming data into a suitable format"
    EDA = "EDA - Exploratory Data Analysis"
    ENVIRONMENT = "Environment - Setting up or managing the environment"
    NOT_INTERESTING = "Not Interesting - Not relevant to the analysis"
    
    def __str__(self):
        return self.value
    
    @classmethod
    def _full_list_to_str(cls):
        return "\n".join([f"{label.name}: {label.value}" for label in cls])