import logging
import pandas as pd
from ..core.utils import remove_id
from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class KB:
    """
    Manages the Knowledge Base for Vamsa's ML Analyzer.
    Gracefully falls back to an empty dictionary if the CSV is missing.
    """
    def __init__(self, knowledge_base=None, kb_csv_path=None):
        if knowledge_base is not None:
            self.knowledge_base = knowledge_base.knowledge_base
        else:
            try:
                self.knowledge_base = pd.read_csv(kb_csv_path) if kb_csv_path else pd.DataFrame()
            except Exception as e:
                logger.warning(f"Vamsa KB not found or failed to load. Operating with empty KB. Error: {e}")
                self.knowledge_base = pd.DataFrame(columns=["Library", "Module", "Caller", "API Name", "Inputs", "Outputs"])
        
        # Fallback built-in rules
        fallback_df = pd.DataFrame([
            {"Library": None, "Module": None, "Caller": "data", "API Name": "Subscript", "Inputs": ["selected columns"]},
            {"Library": None, "Module": None, "Caller": "data", "API Name": "drop", "Inputs": ["dropped columns"]},
        ])
        self.knowledge_base = pd.concat([self.knowledge_base, fallback_df], ignore_index=True)

    def __call__(self, L, L_prime, c, p):
        filtered_kb = self.knowledge_base

        if L is not None:
            filtered_kb = filtered_kb[(filtered_kb["Library"].fillna("") == remove_id(L))]
        if (not filtered_kb.empty) and L_prime is not None:
            filtered_kb = filtered_kb[(filtered_kb["Module"].fillna("") == remove_id(L_prime))]
        if (not filtered_kb.empty) and c is not None:
            filtered_kb = filtered_kb[(filtered_kb["Caller"] == remove_id(c))]
        elif not filtered_kb.empty:
            filtered_kb = filtered_kb[(filtered_kb["Caller"].isna())]
        if (not filtered_kb.empty) and p is not None:
            filtered_kb = filtered_kb[(filtered_kb["API Name"].fillna("") == remove_id(p))]
        elif not filtered_kb.empty:
            filtered_kb = filtered_kb[(filtered_kb["API Name"].isna())]

        if len(filtered_kb) > 1:
            logger.debug(f"KB match collision for {remove_id(p)}. Taking first match.")

        if filtered_kb.empty:
            return [], []

        inputs = filtered_kb["Inputs"].values[0] if len(filtered_kb["Inputs"].values) >= 1 else []
        outputs = filtered_kb["Outputs"].values[0] if len(filtered_kb["Outputs"].values) >= 1 else []
        return inputs, outputs

    def back_query(self, O, p):
        def has_similar_elements(row_list, provided_list):
            min_len = min(len(row_list), len(provided_list))
            return all(p == r or p is None for p, r in zip(provided_list[:min_len], row_list[:min_len]))

        filtered_kb = self.knowledge_base
        if p is not None:
            filtered_kb = filtered_kb[(filtered_kb["API Name"].fillna("") == remove_id(p))]
        if (not filtered_kb.empty) and O is not None:
            filtered_kb = filtered_kb[(filtered_kb["Outputs"].apply(lambda x: has_similar_elements(x, O)))]

        if filtered_kb.empty:
            return []

        return filtered_kb["Inputs"].values[0] if len(filtered_kb["Inputs"].values) >= 1 else []