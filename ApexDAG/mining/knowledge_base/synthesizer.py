import logging

import pandas as pd

from ApexDAG.llm.llm_provider import StructuredLLMProvider
from ApexDAG.util.logger import configure_apexdag_logger

from .models import BatchKBProposal

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class BatchSynthesizer:
    def __init__(self, provider: StructuredLLMProvider) -> None:
        self.provider = provider

    def _build_prompt(self, ops_list: str) -> str:
        return f"""
You are an expert AI Engineer constructing a Knowledge Base for an AST-based data provenance tracker.
Your task is to map Python ML API calls to their semantic inputs and outputs.

VALID LABELS ONLY:
Inputs/Outputs: ['data', 'dataset', 'features', 'labels', 'model', 'trained_model', 'hyperparameter', 'file_path', 'metric', 'predictions']
Callers: ['pandas', 'data', 'model', 'trained_model', 'metric', null]

STRICT RULES:
1. If the API returns a float/score (e.g., accuracy_score, r2_score), the Output is strictly ['metric'].
2. If the API is a class method (e.g., model.predict(X)), the object is the Caller. DO NOT include the Caller in the Inputs array.
3. If an API unpacks multiple distinct concepts (e.g., train_test_split), specify them in order (e.g., ['features', 'validation features', 'labels', 'validation labels']).

GOLDEN EXAMPLES:
- API: `train_test_split` (Library: sklearn, Module: model_selection, Caller: null) -> Inputs: ['features', 'labels'], Outputs: ['features', 'validation features', 'labels', 'validation labels']
- API: `accuracy_score` (Library: sklearn, Module: metrics, Caller: null) -> Inputs: ['labels', 'predictions'], Outputs: ['metric']
- API: `fit` (Library: sklearn, Module: null, Caller: model) -> Inputs: ['features', 'labels'], Outputs: ['trained_model']
- API: `predict` (Library: sklearn, Module: null, Caller: trained_model) -> Inputs: ['features'], Outputs: ['predictions']

Currently, our system fails to annotate the following highly frequent operations:
{ops_list}

Generate accurate KB entries for these operations. Do not hallucinate APIs.
"""

    def synthesize(self, missing_ops: list[tuple[str, int]], current_kb_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Phase 2: Synthesizing KB entries for top {len(missing_ops)} missing APIs...")

        ops_list = "\n".join([f"- {op} (Found {count} times)" for op, count in missing_ops])
        prompt = self._build_prompt(ops_list)
        response = self.provider.generate(prompt=prompt, response_schema=BatchKBProposal)

        new_entries = [entry.model_dump() for entry in response.data.entries]
        new_df = pd.DataFrame(new_entries)
        logger.info(f"Synthesized {len(new_df)} new KB entries.")
        return pd.concat([current_kb_df, new_df], ignore_index=True)
