import logging
from collections.abc import Sequence

import pandas as pd

from SystemX.llm.llm_provider import StructuredLLMProvider
from SystemX.mining.knowledge_base.models import BatchKBProposal
from SystemX.prompts.KBPropagationTemplate import KBPropagationTemplate
from SystemX.prompts.models import OperationMetric

logger = logging.getLogger(__name__)


class BatchSynthesizer:
    def __init__(self, provider: StructuredLLMProvider, template: KBPropagationTemplate) -> None:
        self.provider = provider
        self.template = template

    def synthesize(self, missing_ops: Sequence[OperationMetric], current_kb_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Synthesizing KB entries for {len(missing_ops)} missing APIs.")

        prompt = self.template.render_user_message(missing_ops=list(missing_ops))

        response = self.provider.generate(prompt=prompt, response_schema=BatchKBProposal)

        new_entries = [entry.model_dump() for entry in response.data.entries]
        logger.info(f"Generated {len(new_entries)} new KB entries.")
        logger.info(f"Example entries: {new_entries[:5] if len(new_entries) > 5 else new_entries}")
        new_df = pd.DataFrame(new_entries)

        return pd.concat([current_kb_df, new_df], ignore_index=True)
