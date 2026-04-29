import ast
import logging

import pandas as pd

from ApexDAG.util.logger import configure_apexdag_logger

from ..core.utils import remove_id

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class KB:
    """
    Manages the Knowledge Base for Vamsa's ML Analyzer.
    Gracefully falls back to an empty dictionary if the CSV is missing.
    """

    def __init__(
        self, knowledge_base: pd.DataFrame | None = None, kb_csv_path: str | None = None
    ) -> None:
        if knowledge_base is not None:
            self.knowledge_base = knowledge_base.knowledge_base
        else:
            try:
                if kb_csv_path:
                    df = pd.read_csv(kb_csv_path)

                    def parse_list(val: str | list) -> list:
                        if isinstance(val, list):
                            return val
                        if isinstance(val, str) and val.strip().startswith("["):
                            try:
                                return ast.literal_eval(val)
                            except (SyntaxError, ValueError):
                                return []
                        return [] if pd.isna(val) else [val]

                    if "Inputs" in df.columns:
                        df["Inputs"] = df["Inputs"].apply(parse_list)
                    if "Outputs" in df.columns:
                        df["Outputs"] = df["Outputs"].apply(parse_list)

                    self.knowledge_base = df
                else:
                    self.knowledge_base = pd.DataFrame()
            except Exception as e:
                message = f"""
                Vamsa KB not found or failed to load.
                Operating with empty KB. Error: {e}
                """
                logger.warning(message)
                self.knowledge_base = pd.DataFrame(
                    columns=[
                        "Library",
                        "Module",
                        "Caller",
                        "API Name",
                        "Inputs",
                        "Outputs",
                    ]
                )

        # Fallback built-in rules (Ensuring they are native lists, not strings)
        fallback_df = pd.DataFrame(
            [
                {
                    "Library": None,
                    "Module": None,
                    "Caller": "data",
                    "API Name": "Subscript",
                    "Inputs": ["selected columns"],
                    "Outputs": [],
                },
                {
                    "Library": None,
                    "Module": None,
                    "Caller": "data",
                    "API Name": "drop",
                    "Inputs": ["dropped columns"],
                    "Outputs": [],
                },
            ]
        )
        self.knowledge_base = pd.concat(
            [self.knowledge_base, fallback_df], ignore_index=True
        )
        logger.info(f"Initialized Vamsa KB with {len(self.knowledge_base)} entries.")
        logger.info(self.knowledge_base.head())

    def __call__(
        self, l_str: str | None, l_prime: str | None, c: str | None, p: str | None
    ) -> tuple[list, list]:
        filtered_kb = self.knowledge_base

        if l_str is not None:
            filtered_kb = filtered_kb[
                (filtered_kb["Library"].fillna("") == remove_id(l_str))
            ]
        if (not filtered_kb.empty) and l_prime is not None:
            filtered_kb = filtered_kb[
                (filtered_kb["Module"].fillna("") == remove_id(l_prime))
            ]

        if not filtered_kb.empty:
            if c is not None:
                strict_kb = filtered_kb[(filtered_kb["Caller"] == remove_id(c))]
                if not strict_kb.empty:
                    filtered_kb = strict_kb
                else:
                    filtered_kb = filtered_kb[(filtered_kb["Caller"].isna())]
            else:
                filtered_kb = filtered_kb[(filtered_kb["Caller"].isna())]

        if (not filtered_kb.empty) and p is not None:
            filtered_kb = filtered_kb[
                (filtered_kb["API Name"].fillna("") == remove_id(p))
            ]
        elif not filtered_kb.empty:
            filtered_kb = filtered_kb[(filtered_kb["API Name"].isna())]

        if len(filtered_kb) > 1:
            logger.debug(f"KB match collision for {remove_id(p)}. Taking first match.")

        if filtered_kb.empty:
            return [], []

        inputs = (
            filtered_kb["Inputs"].values[0]
            if len(filtered_kb["Inputs"].values) >= 1
            else []
        )
        outputs = (
            filtered_kb["Outputs"].values[0]
            if len(filtered_kb["Outputs"].values) >= 1
            else []
        )
        return inputs, outputs

    def back_query(self, o_list: list, p: str | None) -> list:
        def has_similar_elements(row_list: list, provided_list: list) -> bool:
            min_len = min(len(row_list), len(provided_list))
            return all(
                p == r or p is None
                for p, r in zip(
                    provided_list[:min_len], row_list[:min_len], strict=False
                )
            )

        filtered_kb = self.knowledge_base
        if p is not None:
            filtered_kb = filtered_kb[
                (filtered_kb["API Name"].fillna("") == remove_id(p))
            ]
        if (not filtered_kb.empty) and o_list is not None:
            filtered_kb = filtered_kb[
                (
                    filtered_kb["Outputs"].apply(
                        lambda x: has_similar_elements(x, o_list)
                    )
                )
            ]

        if filtered_kb.empty:
            return []

        return (
            filtered_kb["Inputs"].values[0]
            if len(filtered_kb["Inputs"].values) >= 1
            else []
        )
