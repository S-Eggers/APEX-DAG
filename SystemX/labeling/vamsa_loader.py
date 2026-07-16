import ast
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NewType, Protocol

from SystemX.sca.constants import DOMAIN_EDGE_TYPES, canonical_domain_label

logger = logging.getLogger(__name__)

DomainEdgeId = NewType("DomainEdgeId", int)

@dataclass(frozen=True)
class VamsaEntry:
    """Immutable representation of a Vamsa Knowledge Base row."""

    library: str
    module: str
    caller: str
    api_name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

class LineageMappingPolicy(Protocol):
    """Protocol dictating how a VamsaEntry is mapped to a DomainEdgeId."""

    def map_entry(self, entry: VamsaEntry) -> DomainEdgeId: ...

class IOSignatureMappingPolicy:
    """Analyzes the semantic input/output signature of a knowledge base entry to map it to a specific Lineage Node taxonomy."""

    def map_entry(self, entry: VamsaEntry) -> DomainEdgeId:
        return DomainEdgeId(canonical_domain_label(self._map_signature(entry)))

    def _map_signature(self, entry: VamsaEntry) -> int:
        in_set = set(entry.inputs)
        out_set = set(entry.outputs)
        consumes_path = bool(in_set.intersection({"file_path", "file_paths"}))
        consumed = in_set | ({entry.caller} if entry.caller else set())

        if out_set.intersection({"model artifact", "data_artifact"}) or (consumes_path and not out_set):
            return DomainEdgeId(DOMAIN_EDGE_TYPES["DATA_EXPORT"])

        if consumes_path or (not consumed and out_set.intersection({"data", "features", "labels", "target"})):
            return DomainEdgeId(DOMAIN_EDGE_TYPES["DATA_IMPORT_EXTRACTION"])

        if "predictions" in out_set:
            return DomainEdgeId(DOMAIN_EDGE_TYPES["MODEL_OPERATION"])

        if out_set.intersection({"model", "trained_model"}):
            return DomainEdgeId(DOMAIN_EDGE_TYPES["MODEL_OPERATION"])

        if out_set.intersection({"metric", "metric results", "multi_confusion_matrix"}):
            if consumed.intersection({"predictions", "labels", "model", "trained_model"}):
                return DomainEdgeId(DOMAIN_EDGE_TYPES["MODEL_OPERATION"])
            return DomainEdgeId(DOMAIN_EDGE_TYPES["EDA"])

        if "display_object" in out_set:
            return DomainEdgeId(DOMAIN_EDGE_TYPES["EDA"])

        if out_set.intersection({"data", "features", "data transformer", "data splitter", "tokenized data", "shuffled arrays", "document-term matrix"}) and consumed.intersection(
            {"features", "data", "labels", "raw text documents", "transformer definitions", "hyperparameter", "arrays", "data transformer"}
        ):
            return DomainEdgeId(DOMAIN_EDGE_TYPES["DATA_TRANSFORM"])

        return DomainEdgeId(DOMAIN_EDGE_TYPES["NOT_RELEVANT"])

class VamsaKBLoader:
    """Handles file I/O safely and delegates mapping to the injected policy."""

    def __init__(self, policy: LineageMappingPolicy) -> None:
        self._policy = policy

    @staticmethod
    def _parse_list_string(list_str: str) -> tuple[str, ...]:
        if not list_str or list_str == "[]":
            return ()
        try:
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list):
                return tuple(parsed)
            return ()
        except (SyntaxError, ValueError):
            logger.warning("Failed to parse list string safely: %s", list_str)
            return ()

    def load_and_map(self, file_path: Path) -> dict[VamsaEntry, DomainEdgeId]:
        mapping: dict[VamsaEntry, DomainEdgeId] = {}

        if not file_path.exists():
            logger.error("Vamsa KB file missing at path: %s", file_path)
            raise FileNotFoundError(f"Missing knowledge base file: {file_path}")

        try:
            with file_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entry = VamsaEntry(
                        library=row.get("Library", ""),
                        module=row.get("Module", ""),
                        caller=row.get("Caller", ""),
                        api_name=row.get("API Name", ""),
                        inputs=self._parse_list_string(row.get("Inputs", "[]")),
                        outputs=self._parse_list_string(row.get("Outputs", "[]")),
                    )
                    mapping[entry] = self._policy.map_entry(entry)

        except OSError as e:
            logger.error("System error reading Vamsa KB: %s", e)
            raise
        except csv.Error as e:
            logger.error("Malformed CSV structure in Vamsa KB: %s", e)
            raise

        return mapping
