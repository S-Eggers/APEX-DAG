from textwrap import dedent
from typing import Final

from SystemX.labeling.models import DomainLabelDescription

class LeanNodeClassificationTemplate:
    """Zero-shot, support-free prompt for operational-hub classification."""

    SYSTEM_HEADER: Final[str] = dedent("""
        You are an expert ML engineer. Classify a single Python operation (a function
        call, method invocation, or loop) from a Jupyter notebook into exactly ONE
        category of the taxonomy below, based on its semantic intent.
    """).strip()

    INSTRUCTION: Final[str] = (
        "Return only the chosen `domain_label` (one of the taxonomy keys). "
        "Do not explain your choice."
    )

    def render_system_message(self) -> str:
        taxonomy = DomainLabelDescription.get_prompt_description()
        return f"{self.SYSTEM_HEADER}\n\n## Taxonomy (valid keys)\n{taxonomy}\n\n{self.INSTRUCTION}"

    def render_user_message(self, node_code: str) -> str:
        return f"Classify this operation:\n\n{node_code}"
