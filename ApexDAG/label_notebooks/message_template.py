from enum import Enum

def generate_message(node_id_source: str, node_id_target: str, code_edge: str, subgraph_context: str, code: str) -> list:
    """
    Generates a message for labeling edges in a dataflow graph.

    Args:
        node_id_source (str): The unique identifier of the source node.
        node_id_target (str): The unique identifier of the target node.
        code_edge (str): The code that connects the source and target nodes.
        subgraph_context (str): The context of the subgraph.
        code (str): The code snippet from which the graph was created.

    Returns:
        list: A properly formatted message for the LLM.
    """

    domain_labels = """
    - MODEL_TRAIN: Model Train - Training machine learning models
    - MODEL_EVALUATION: Model Evaluation - Evaluating the performance of models
    - HYPERPARAMETER_TUNING: Hyperparameter Tuning - Optimizing model parameters
    - DATA_EXPORT: Data Export - Exporting data to external storage, saving the data, exporting the data/models
    - DATA_IMPORT_EXTRACTION: Data Import / Extraction - Importing or extracting data from external sources (not variables)
    - DATA_TRANSFORM: Data Transform - Transforming data into a suitable format
    - EDA: EDA - Exploratory Data Analysis
    - ENVIRONMENT: Environment - Setting up or managing the environment: i.e. installing packages/modules, setting up virtual environments
    - NOT_INTERESTING: Not Interesting - Not relevant to the analysis
    """

    message_content = (
        "You are an expert in labeling dataflow graphs of Python programs, specifically in the context of machine learning. "
        "Your task is to label a specific edge in the dataflow graph based on the provided context and output the `LabelledEdge` method with the appropriate parameters.\n\n"
        "Follow these steps:\n"
        "1. Examine the edge of interest, which is indicated by the following ID: "
        f"'{node_id_source} ---{code_edge}--->{node_id_target}'.\n"
        "2. Analyze the graph context and the code provided below to understand the relationship between the nodes.\n"
        "3. Choose the most appropriate domain label from the following list:\n"
        f"{domain_labels}\n"
        "4. Explain your reasoning for choosing this label.\n"
        "5. Output the `LabelledEdge` method with the following structure:\n"
        "   LabelledEdge {\n"
        f"       \"source\": \"{node_id_source}\",  # Unique identifier for the source node\n"
        f"       \"target\": \"{node_id_target}\",  # Unique identifier for the target node\n"
        f"       \"code\": \"{code_edge}\",  # Code representing the edge\n"
        "       \"edge_type\": \"...\",  # Type of the edge\n"
        "       \"domain_label\": \"...\"  # Domain-specific label for the edge\n"
        "   }\n\n"
        "Here is the code snippet from which this graph was created:\n"
        f"{code}\n\n"
        "Here is the graph context for your task:\n"
        f"{subgraph_context}\n\n"
        "Please ensure that your response is only the `LabelledEdge` method with the appropriate parameters.\n"
        "If not enough context is provided, output the domain label 'MORE_CONTEXT_NEEDED'."
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