from enum import Enum

def generate_message(node, subgraph_context):
    """
    Generates the labeling prompt for the Groq API.
    """
    return [
        {
            "role": "user",
            "content": (
                "You are an expert in labeling dataflow graphs of Python programs. "
                "Please label the following node within the provided subgraph context. "
                "The node of interest is indicated, and the subgraph provides additional context. "
                "Use one of the following labels:\n"
                f"{DomainLabel._full_list_to_str()}"
                "\n"
                f"Here is the node of interest: '{node.id}'\n"
                f"And here is the subgraph context: {subgraph_context}"
                f"Ensure the following JSON structure:"
                "{\n"
                '  "id": "...",  # Unique identifier for the node\n'
                '  "node_type": "...",  # Type of the node \n'
                '  "domain_label": "...",  # Domain-specific label for the node\n'
                "}\n"
                "Please provide the label in this format based on the node's context."
            ),
        }
    ]


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