from dataclasses import dataclass


@dataclass(frozen=True)
class LineageTuple:
    """Represents an extracted lineage relationship for the golden dataset."""

    tuple_type: str
    subject_id: str
    object_id: str
