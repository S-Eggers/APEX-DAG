from textwrap import dedent
from typing import Final

from ApexDAG.label_notebooks.models import DomainLabel


class EdgeClassificationTemplate:
    SYSTEM_HEADER: Final[str] = dedent("""
        You are a Principal ML Engineer. Your task is to classify a specific
         graph edge (the <EDGE_OF_INTEREST>) within the context of an AST-based
         representation of a Jupyter Notebook.
    """).strip()

    INSTRUCTIONS: Final[str] = dedent("""
        ## Instructions
        1. Analyze the <RAW_CODE> and its relationship to the <SUBGRAPH_CONTEXT>.
        2. Assign exactly ONE `domain_label` from the Taxonomy below.
        3. Provide a concise technical justification for your choice.
        4. Focus on the semantic intent: Is this data movement, model math, or just noise?
    """).strip()

    def render_system_prompt(self) -> str:
        """Dynamically generates the prompt using the Enum source of truth."""
        taxonomy = DomainLabel.get_prompt_description()

        return f"""
{self.SYSTEM_HEADER}

## Taxonomy (Valid Keys)
{taxonomy}

{self.INSTRUCTIONS}
        """.strip()

    def render_user_message(self, **kwargs: object) -> str:
        """
        Serializes the graph context into the prompt.
        Uses the context's internal string representation for consistency.
        """
        # Extract the specific edge we are asking about
        (src,) = kwargs["source_id"]
        tgt = kwargs["target_id"]
        key = kwargs["edge_code"]
        context = kwargs["subgraph_context"]
        raw_code = kwargs["raw_code"]

        return f"""
<EDGE_OF_INTEREST>
Source: {src}
Target: {tgt}
Key: {key}
</EDGE_OF_INTEREST>

<RAW_CODE>
{raw_code}
</RAW_CODE>

<SUBGRAPH_CONTEXT>
{context!s}
</SUBGRAPH_CONTEXT>

Based on the code snippets and graph connections above, classify the <EDGE_OF_INTEREST>.
""".strip()
