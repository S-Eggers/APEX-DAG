from typing import Final

from .models import OperationMetric


class KBPropagationTemplate:
    SYSTEM_CONTEXT: Final[str] = (
        "You are an expert AI Engineer constructing a Knowledge Base for an AST-based data provenance tracker. Your task is to map Python ML API calls to their semantic inputs and outputs."
    )

    LABELS: Final[str] = (
        "Inputs/Outputs: ['data', 'dataset', 'validation features', 'validation labels', "
        "'features', 'labels', 'model', 'trained model', 'hyperparameter', 'file path', "
        "'metric', 'predictions']\n"
        "Callers: ['data', 'dataset', 'model', 'trained model', 'metric', null]"
    )

    RULES: Final[str] = (
        "1. If the API returns a float/score, the Output is strictly ['metric'].\n"
        "2. If the API is a class method, the object is the Caller. DO NOT include it in Inputs.\n"
        "3. If an API unpacks multiple concepts, specify them in order."
    )

    def render_user_message(self, missing_ops: list[OperationMetric]) -> str:
        formatted_ops = "\n".join([f"- {op.name} (Found {op.count} times)" for op in missing_ops])

        return f"""
{self.SYSTEM_CONTEXT}

VALID LABELS ONLY:
{self.LABELS}

STRICT RULES:
{self.RULES}

Currently, our system fails to annotate:
{formatted_ops}

Generate accurate KB entries. Do not hallucinate APIs.
"""
