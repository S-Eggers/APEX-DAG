import logging
import re
from enum import StrEnum
from typing import TypedDict

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from SystemX.sca.constants import (
    DOMAIN_EDGE_TYPES,
    REVERSE_EDGE_TYPES,
    REVERSE_NODE_TYPES,
    canonical_domain_label,
)

logger = logging.getLogger(__name__)

class DomainLabelDescription(StrEnum):
    """LLM prompt descriptions."""

    MODEL_OPERATION = (
        "Training, fitting, tuning, predicting, or scoring an ML model - including metrics computed on model outputs. "
        "Examples: model.fit, model.predict, model.score, cross_val_score, GridSearchCV, accuracy_score(y, y_pred), loss(...). "
        "NOT data preprocessing (that is DATA_TRANSFORM) and NOT plotting a metric for inspection (that is EDA)."
    )
    DATA_IMPORT_EXTRACTION = (
        "The FIRST entry of data into the program from an external source. "
        "Examples: pd.read_csv, pd.read_sql, np.load, open(path).read, load_dataset, requests.get(...).json, a DB/API query. "
        "NOT in-memory operations on data that is already loaded (that is DATA_TRANSFORM)."
    )
    DATA_TRANSFORM = (
        "Produces NEW or MODIFIED data that is CONSUMED DOWNSTREAM by a later transform/model step: cleaning, filtering, "
        "feature engineering, scaling, encoding, merging, aggregating, splitting. Examples: df.dropna, df.groupby().agg, "
        "pd.merge, StandardScaler.fit_transform, train_test_split. KEY TEST: its output feeds a subsequent transform/model. "
        "Mechanical reshaping/formatting (reshape, np.concatenate, np.arange, .iloc slicing, inverse_transform) is ONLY "
        "DATA_TRANSFORM if the result is actually fed downstream; if it is just printed, displayed, or used to build a plot, "
        "it is NOT_RELEVANT. NOT mere inspection or visualization (that is EDA)."
    )
    EDA = (
        "Consumes data/results to REVEAL THEIR CONTENT to a human - inspection, summary, or a data-bearing visualization. "
        "Examples: df.head, df.describe, df.info, df.shape, print(df), df.value_counts(), plt.plot(data), plt.scatter(x, y), "
        "sns.heatmap(corr), plt.imshow(img). KEY TEST: the op reads real data and surfaces information about it, and nothing "
        "downstream consumes the result. Plot STYLING/DECORATION that carries no data content (axis labels, titles, legends, "
        "limits, grids, figure setup) is NOT EDA - it is NOT_RELEVANT."
    )
    DATA_EXPORT = (
        "Persists data, models, or artifacts to external storage - output leaves the workflow to disk/DB. "
        "Examples: df.to_csv, df.to_parquet, pickle.dump, joblib.dump, model.save, torch.save, plt.savefig, open(path,'w').write."
    )
    NOT_RELEVANT = (
        "Operations that neither load data, nor produce data/models/artifacts consumed downstream, nor reveal data content to a "
        "human. This positively INCLUDES three groups: "
        "(1) utility/environment ops - import, os.makedirs, warnings.filterwarnings, logging, time.time, random seed setup, "
        "print('done'), pip install; "
        "(2) PLOT STYLING/DECORATION with no data content - plt.xlabel, plt.ylabel, plt.title, plt.legend, plt.xlim, plt.ylim, "
        "plt.grid, plt.figure, plt.show, plt.colorbar, ax.set_*; "
        "(3) mechanical array/frame reshaping or formatting whose result is only printed, displayed, or used to build a plot "
        "grid (reshape/np.concatenate/np.arange/.iloc/inverse_transform for display), NOT fed to a later transform or model. "
        "Prefer a specific data/model category only when the op truly loads, transforms (for downstream use), inspects, exports, "
        "or models data - but do NOT force plotting decoration or display-only formatting into EDA/DATA_TRANSFORM."
    )

    @classmethod
    def get_prompt_description(cls) -> str:
        return "\n".join([f"- {label.name}: {label.value}" for label in cls])

if not set(DomainLabelDescription.__members__.keys()).issubset(set(DOMAIN_EDGE_TYPES.keys())):
    raise RuntimeError("CRITICAL: DomainLabelDescription keys in models.py do not match DOMAIN_EDGE_TYPES defined in SystemX.sca.constants.")

class MultiNode(BaseModel):
    node_id: str
    node_type: str
    label: str

    def __str__(self) -> str:
        return f"""Node(id={self.node_id},
 node_type={self.node_type},
 label={self.label})"""

class MultiEdge(BaseModel):
    source: str
    target: str
    key: str
    edge_type: str
    code: str | None = None
    lineno: list[int] | None = None

    def __str__(self) -> str:
        return f"""Edge(source={self.source},
 target={self.target},
 key={self.key},
 code={self.code},
 edge_type={self.edge_type})
"""

class MultiLabelledEdge(BaseModel):
    source: str
    target: str
    domain_label: str
    predicted_label: int = 0

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("predicted_label", mode="before")
    @classmethod
    def map_label_to_id(cls, v: int, info: ValidationInfo) -> int:
        """Dynamically maps the LLM string output directly using the SCA constants."""
        label_key = info.data.get("domain_label")

        fallback_id = DOMAIN_EDGE_TYPES.get("NOT_RELEVANT", 5)

        return canonical_domain_label(DOMAIN_EDGE_TYPES.get(label_key, fallback_id))

class MultiGraphContext(BaseModel):
    nodes: list[MultiNode]
    edges: list[MultiEdge | MultiLabelledEdge]

    @classmethod
    def from_graph(cls, graph: nx.MultiDiGraph) -> "MultiGraphContext":
        """Builds the strict DTO from a networkx.MultiDiGraph."""
        nodes = []
        for node_name, node_data in graph.nodes(data=True):
            clean_label = re.sub(r"_\d+", "", str(node_name))
            nodes.append(
                MultiNode(
                    node_id=str(node_name),
                    label=node_data.get("label", clean_label),
                    node_type=REVERSE_NODE_TYPES.get(node_data.get("node_type"), "UNKNOWN"),
                )
            )

        edges = []
        for src, tgt, key, edge_data in graph.edges(data=True, keys=True):
            lineno_start = edge_data.get("lineno")
            lineno_end = edge_data.get("end_lineno", lineno_start)
            lineno_range = list(range(lineno_start, lineno_end + 1)) if lineno_start is not None else None

            edges.append(
                MultiEdge(
                    source=str(src),
                    target=str(tgt),
                    key=str(key),
                    code=str(edge_data.get("code", "")),
                    edge_type=REVERSE_EDGE_TYPES.get(edge_data.get("edge_type"), "UNKNOWN"),
                    lineno=lineno_range,
                )
            )

        return cls(nodes=nodes, edges=edges)

class NotebookCellData(TypedDict):
    cell_id: str
    source: str

class MultiLabelledNode(BaseModel):
    """Full-context classification response: the domain label only."""

    domain_label: str = Field(..., json_schema_extra={"enum": list(DomainLabelDescription.__members__)})

    @field_validator("domain_label", mode="before")
    @classmethod
    def normalize_label(cls, v: object) -> str:
        key = str(v).strip().upper()
        if key not in DOMAIN_EDGE_TYPES:
            logger.warning("Unknown domain_label %r from LLM; falling back to NOT_RELEVANT.", v)
            return "NOT_RELEVANT"
        return key

    @property
    def predicted_label(self) -> int:
        return canonical_domain_label(DOMAIN_EDGE_TYPES[self.domain_label])

class NotebookNodeLabel(BaseModel):
    """One node's label in a WHOLE-NOTEBOOK classification (the strong LLM baseline)."""

    node_id: str
    reasoning: str
    domain_label: str = Field(..., json_schema_extra={"enum": list(DomainLabelDescription.__members__)})

    @field_validator("domain_label", mode="before")
    @classmethod
    def normalize_label(cls, v: object) -> str:
        key = str(v).strip().upper()
        if key not in DOMAIN_EDGE_TYPES:
            logger.warning("Unknown domain_label %r from LLM; falling back to NOT_RELEVANT.", v)
            return "NOT_RELEVANT"
        return key

    @property
    def predicted_label(self) -> int:
        return canonical_domain_label(DOMAIN_EDGE_TYPES[self.domain_label])

class NotebookClassification(BaseModel):
    """Structured response for one whole-notebook classification call: a list of per-node labels."""

    nodes: list[NotebookNodeLabel]

class LeanLabelledNode(BaseModel):
    """Minimal classification response: the domain label only, no reasoning."""

    domain_label: str

    @property
    def predicted_label(self) -> int:
        key = self.domain_label.upper() if isinstance(self.domain_label, str) else self.domain_label
        return canonical_domain_label(DOMAIN_EDGE_TYPES.get(key, 5))

class MultiSubgraphContext(MultiGraphContext):
    node_of_interest: str

    def __str__(self) -> str:
        return f"Node of Interest: {self.node_of_interest}\nNodes: {self.nodes}\nEdges: {self.edges}"
