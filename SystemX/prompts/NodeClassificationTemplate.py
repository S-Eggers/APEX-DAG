from textwrap import dedent
from typing import Final

from SystemX.labeling.models import DomainLabelDescription

class NodeClassificationTemplate:
    SYSTEM_HEADER: Final[str] = dedent("""
        You are a Principal ML Engineer. Your task is to classify an <OPERATION_NODE>
        within a bipartite Dataflow Graph. This node represents a specific execution
        event (such as a function call, method invocation, or loop) in a Jupyter Notebook.
    """).strip()

    INSTRUCTIONS: Final[str] = dedent("""
        ## Instructions
        1. Analyze the <OPERATION_NODE> code and its <TOPOLOGICAL_CONTEXT> (inputs and outputs).
        2. Assign exactly ONE `domain_label` from the Taxonomy below to describe the semantic intent of this execution event.
        3. Return only the chosen `domain_label`. Do not explain your choice.

        ## Decision rules (apply in order; these resolve the common confusions)
        - NOT_RELEVANT is a LAST RESORT, never a default for uncertainty. If the operation touches data, a model, or an
          artifact in ANY way, you MUST pick the specific category. Reserve NOT_RELEVANT for pure environment/utility code
          (imports, logging, seeds, filesystem setup, status prints).
        - DATA_TRANSFORM vs EDA - use the topology: if the operation's OUTPUT is consumed by a later transform/model step,
          it is DATA_TRANSFORM. If the result is only inspected/printed/plotted and nothing downstream uses it, it is EDA.
        - A metric or score computed about a MODEL (e.g. accuracy_score, model.score, a loss) is MODEL_OPERATION, even if it
          is then printed or plotted.
        - DATA_IMPORT_EXTRACTION is only the FIRST load of data from an external source; later in-memory manipulation of that
          data is DATA_TRANSFORM.
        - Writing/saving anything to disk or a database (data, model, figure) is DATA_EXPORT.
        - When two categories seem plausible, prefer the one describing the operation's EFFECT on the dataflow over a generic label.
    """).strip()

    def render_system_message(self) -> str:
        """Dynamically generates the system prompt using the Enum source of truth."""
        taxonomy = DomainLabelDescription.get_prompt_description()

        return f"""
{self.SYSTEM_HEADER}

## Taxonomy (Valid Keys)
{taxonomy}

{self.INSTRUCTIONS}
        """.strip()

    def render_user_message(self, **kwargs: object) -> str:
        """Serializes the bipartite graph context into the prompt."""
        node_id = kwargs.get("node_id", "UNKNOWN")
        node_code = kwargs.get("node_code", "UNKNOWN")
        context = kwargs.get("subgraph_context", "")
        raw_code = kwargs.get("raw_code", "")

        return f"""
<OPERATION_NODE>
ID: {node_id}
Code Snippet: {node_code}
</OPERATION_NODE>

<RAW_CODE_CONTEXT>
{raw_code}
</RAW_CODE_CONTEXT>

<TOPOLOGICAL_CONTEXT>
{context!s}
</TOPOLOGICAL_CONTEXT>

Based on the execution context and the inputs/outputs shown in the topology, classify the <OPERATION_NODE>.
""".strip()
