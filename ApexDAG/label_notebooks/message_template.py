def generate_system_prompt() -> str:
    descriptions = {
        "MODEL_OPERATION": "Training, evaluating, tuning, or predicting.",
        "DATA_IMPORT_EXTRACTION": "Loading data from files/APIs.",
        "DATA_TRANSFORM": "Cleaning, preprocessing, feature engineering.",
        "EDA": "Plotting, stats, head(), tail().",
        "DATA_EXPORT": "Saving files, exporting environment.",
        "NOT_RELEVANT": "Comments, logging, environment setup.",
    }
    taxonomy_str = "\n".join([f"- {k}: {v}" for k, v in descriptions.items()])

    return f"""You are a principal ML engineer. Classify the <EDGE_OF_INTEREST>.

## Taxonomy (Valid Keys)
{taxonomy_str}

## Instructions
- You MUST return one of the VALID KEYS above as the `domain_label`.
- Do NOT return the description; return the uppercase KEY.
- Example: "DATA_TRANSFORM"
- Provide technical reasoning."""


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
