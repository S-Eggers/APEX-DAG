from textwrap import dedent
from typing import Final

from SystemX.labeling.models import DomainLabelDescription

class StrongNotebookClassificationTemplate:
    """Prompt for the STRONG LLM baseline: whole-notebook, few-shot, chain-of-thought."""

    SYSTEM_HEADER: Final[str] = dedent("""
        You are a Principal ML Engineer classifying EVERY operation in a data-science
        notebook. You are given the notebook's operations in execution order, each with
        its dataflow context (inputs, produced variables, downstream consumers). Because
        you see the whole pipeline, use global context: an operation's role often depends
        on where its output goes (e.g. a cleaning step feeding a model is a transform).
    """).strip()

    DECISION_RULES: Final[str] = dedent("""
        ## Decision rules (apply in order)
        - NOT_RELEVANT is a LAST RESORT - only pure environment/utility code (imports,
          logging, seeds, filesystem setup, status prints). If it touches data, a model,
          or an artifact, pick the specific category.
        - DATA_TRANSFORM vs EDA - use the topology: if the OUTPUT is consumed by a later
          transform/model step it is DATA_TRANSFORM; if it is only inspected/printed/
          plotted and nothing downstream uses it, it is EDA.
        - A metric/score about a MODEL (accuracy_score, model.score, a loss) is
          MODEL_OPERATION, even if printed/plotted.
        - DATA_IMPORT_EXTRACTION is only the FIRST load from an external source; later
          in-memory manipulation is DATA_TRANSFORM.
        - Saving anything to disk/DB (data, model, figure) is DATA_EXPORT.
    """).strip()

    FEW_SHOT: Final[str] = dedent("""
        ## Examples (reason first, then label)
        - `pd.read_csv("train.csv")` [inputs: none; assigns: df] →
          reasoning: "First load of external data from a CSV; nothing upstream." label: DATA_IMPORT_EXTRACTION
        - `df.dropna()` [inputs: df; assigns: df2; used by: model.fit(X, y)] →
          reasoning: "Cleans data and the result feeds model.fit downstream, so it is part of the data pipeline." label: DATA_TRANSFORM
        - `df.head()` [inputs: df; result not used by any later operation] →
          reasoning: "Inspects the dataframe; output is a dead-end (only displayed)." label: EDA
        - `accuracy_score(y_test, preds)` [inputs: y_test, preds] →
          reasoning: "Computes a metric about a model's predictions." label: MODEL_OPERATION
        - `os.makedirs("out", exist_ok=True)` [inputs: none] →
          reasoning: "Filesystem/environment setup; touches no data or model." label: NOT_RELEVANT
    """).strip()

    def render_system_message(self) -> str:
        taxonomy = DomainLabelDescription.get_prompt_description()
        return f"""
{self.SYSTEM_HEADER}

## Taxonomy (valid keys)
{taxonomy}

{self.DECISION_RULES}

{self.FEW_SHOT}

## Output
Return one entry per operation as {{node_id, reasoning, domain_label}}. Emit `reasoning`
before `domain_label`. `domain_label` must be exactly one taxonomy key. Label every
operation you are given; do not add or drop any node_id.
""".strip()

    def render_user_message(self, serialized_nodes: str) -> str:
        return f"""
Classify every operation below (execution order). Each block gives the node id, its
code, and its dataflow context.

{serialized_nodes}
""".strip()
