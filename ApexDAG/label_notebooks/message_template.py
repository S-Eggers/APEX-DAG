from enum import Enum

def generate_message(node_id: str, subgraph_context: str) -> list:
    """
    Generates a message for labeling nodes in a dataflow graph.

    Args:
        node_id (str): The unique identifier of the node of interest.
        subgraph_context (str): The context of the subgraph.

    Returns:
        list: A properly formatted message for the LLM.
    """
    # redundancy, TODO: remove duplication
    domain_labels = """
    MODEL_TRAIN: Model Train - Training machine learning models
    MODEL_EVALUATION: Model Evaluation - Evaluating the performance of models
    HYPERPARAMETER_TUNING: Hyperparameter Tuning - Optimizing model parameters
    DATA_EXPORT: Data Export - Exporting data to external storage
    DATA_IMPORT_EXTRACTION: Data Import / Extraction - Importing or extracting data from sources
    DATA_TRANSFORM: Data Transform - Transforming data into a suitable format
    EDA: EDA - Exploratory Data Analysis
    ENVIRONMENT: Environment - Setting up or managing the environment
    NOT_INTERESTING: Not Interesting - Not relevant to the analysis
    """

    json_structure = """
    {
      "id": "...",  # Unique identifier for the node
      "node_type": "...",  # Type of the node
      "domain_label": "..."  # Domain-specific label for the node
    }
    """

    message_content = (
        f"You are an expert in labeling dataflow graphs of Python programs, specifically machine learning. "
        f"Please label the following node within the provided subgraph context. "
        f"The node of interest is indicated, and the subgraph provides additional context. "
        f"Use one of the following domain labels:\n{domain_labels}\n\n"
        f"Here is the node of interest:\n"
        f"  ID: '{node_id}'\n"
        f"\nHere is the subgraph context:\n{subgraph_context}\n\n"
        f"Please ensure that your response is formatted as valid JSON with the following structure:\n{json_structure}\n\n"
        f"Provide the domain label based on the given node's context."
    )

    return [{"role": "user", "content": message_content}]


class DomainLabel(str, Enum):
    MODEL_TRAIN = "Model Train - Training machine learning models"
    MODEL_EVALUATION = "Model Evaluation - Evaluating the performance of models"
    HYPERPARAMETER_TUNING = "Hyperparameter Tuning - Optimizing model parameters"
    DATA_EXPORT = "Data Export - Exporting data to external storage"
    DATA_IMPORT_EXTRACTION = "Data Import / Extraction - Importing or extracting data from sources"
    DATA_TRANSFORM = "Data Transform - Transforming data into a suitable format"
    EDA = "EDA - Exploratory Data Analysis"
    ENVIRONMENT = "Environment - Setting up or managing the environment"
    NOT_INTERESTING = "Not Interesting - Not relevant to the analysis"
    
    def __str__(self):
        return self.value
    
    @classmethod
    def _full_list_to_str(cls):
        return "\n".join([f"{label.name}: {label.value}" for label in cls])