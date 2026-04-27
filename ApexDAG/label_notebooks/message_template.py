from ApexDAG.label_notebooks.schema import DomainLabel


def generate_system_prompt() -> str:
    domain_labels_str = DomainLabel.get_prompt_description()
    return f"""You are a principal machine learning engineer analyzing an
 Abstract Syntax Tree (AST) dataflow graph.
Your task is to classify a specific computational edge using the provided
 domain taxonomy.

## Taxonomy
{domain_labels_str}

## Instructions
1. Analyze the surrounding <SUBGRAPH_CONTEXT> to understand the data lineage.
2. Examine the specific <EDGE_OF_INTEREST> and the raw <CODE> it represents.
3. Determine the correct `domain_label` based strictly on the Taxonomy.
4. If an operation spans multiple categories, default to the one that mutates the data
   state most significantly.
5. Provide a concise, technical justification in the `reasoning` field."""


def generate_user_message(
    source_id: str,
    target_id: str,
    edge_code: str,
    subgraph_context: str,
    raw_code: str,
) -> str:
    return f"""
<EDGE_OF_INTEREST>
{source_id} --[{edge_code}]--> {target_id}
</EDGE_OF_INTEREST>

<RAW_CODE>
{raw_code}
</RAW_CODE>

<SUBGRAPH_CONTEXT>
{subgraph_context}
</SUBGRAPH_CONTEXT>
"""
