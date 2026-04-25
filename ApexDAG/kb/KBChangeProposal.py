from dataclasses import dataclass


@dataclass
class KBChangeProposal:
    """Represents a proposed change to the KB"""
    change_type: str  # "add_annotation", "add_traversal", "modify", "remove"
    description: str

    # For annotation KB changes
    library: str | None = None
    module: str | None = None
    caller: str | None = None
    api_name: str | None = None
    inputs: list[str] | None = None
    outputs: list[str] | None = None

    # For traversal KB changes
    traversal_api_name: str | None = None
    column_exclusion: bool | None = None
    traversal_rule_name: str | None = None  # Name of function to use

    # Metadata
    rationale: str = ""
    expected_impact: str = ""
