import json
from enum import Enum


class DomainLabel(str, Enum):
    """Enumeration for domain-specific labels in the ML dataflow graph."""

    MODEL_TRAIN = "Training machine learning models."
    MODEL_EVALUATION = "Evaluating the performance of models."
    HYPERPARAMETER_TUNING = "Optimizing model parameters."
    DATA_EXPORT = "Exporting data to external storage, saving files."
    DATA_IMPORT_EXTRACTION = "Importing or extracting data from external sources (e.g., files, databases, APIs)."
    DATA_TRANSFORM = "Transforming, cleaning, or preprocessing data (e.g., scaling, one-hot encoding, feature engineering)."
    EDA = "Exploratory Data Analysis (e.g., plotting, printing stats, generating reports)."
    ENVIRONMENT = "Setting up or managing the environment (e.g., installing packages, setting seeds)."
    NOT_INTERESTING = "Not relevant to the core ML workflow (e.g., logging, printing status, comments)."
    MORE_CONTEXT_NEEDED = "Insufficient information to make a confident decision."

    def __str__(self):
        return self.name

    @classmethod
    def get_prompt_description(cls):
        """Generates a formatted string of all labels for the prompt."""
        return "\n".join([f"- {label.name}: {label.value}" for label in cls])


def generate_message(
    node_id_source: str,
    node_id_target: str,
    code_edge: str,
    subgraph_context: str,
    code: str,
) -> list:
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
    domain_labels_str = DomainLabel.get_prompt_description()
    escaped_code_edge = json.dumps(code_edge)[1:-1]
    message_content = f"""
You are an expert in analyzing Python code for machine learning pipelines. Your task is to classify an edge in a dataflow graph with a domain-specific label.

You will be given the full code snippet, the specific edge to analyze, and the surrounding graph context.

## Instructions
1.  **Analyze the context**: Carefully examine the `<EDGE>`, `<CODE>`, and `<CONTEXT>` sections to understand the operation occurring between the source and target nodes.
2.  **Choose `domain_label`**: Select the most fitting label from the provided `<DOMAIN_LABELS>` list.
4.  **Provide Reasoning**: Briefly explain your choice in the `reasoning` field. Your reasoning should justify your choice of both the domain label and the edge type.
5.  **Output JSON**: Your final output must be a single, valid JSON object and nothing else.

<EDGE>
{node_id_source} ---{code_edge}---> {node_id_target}
</EDGE>

<CODE>
{code}
</CODE>

<CONTEXT>
{subgraph_context}
</CONTEXT>

<DOMAIN_LABELS>
{domain_labels_str}
</DOMAIN_LABELS>

## Required JSON Output Format
{{
    "source": "{node_id_source}",
    "target": "{node_id_target}",
    "code": "{escaped_code_edge}",
    "domain_label": "...",
    "reasoning": "A brief justification for the chosen labels."
}}
"""

    return [{"role": "user", "content": message_content}]


if __name__ == "__main__":
    # --- Example Usage ---
    source_node = "pandas.read_csv.return"
    target_node = "df.train_test_split.df"
    edge_code = "df"

    full_code = """
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("data.csv")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'])
    """

    graph_context = """
    digraph {
        "pandas.read_csv" [label="pandas.read_csv(...)"];
        "pandas.read_csv.return" [label="<return>"];
        "df.train_test_split" [label="df.train_test_split(...)"];
        "df.train_test_split.df" [label="df"];
        
        "pandas.read_csv" -> "pandas.read_csv.return" [label=""];
        "pandas.read_csv.return" -> "df.train_test_split.df" [label="df"];
    }
    """

    llm_message = generate_message(
        node_id_source=source_node,
        node_id_target=target_node,
        code_edge=edge_code,
        subgraph_context=graph_context,
        code=full_code,
    )

    print(llm_message[0]["content"])
