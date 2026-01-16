from typing import List, Optional
from dataclasses import dataclass


@dataclass
class KBChangeProposal:
    """Represents a proposed change to the KB"""
    change_type: str  # "add_annotation", "add_traversal", "modify", "remove"
    description: str
    
    # For annotation KB changes
    library: Optional[str] = None
    module: Optional[str] = None
    caller: Optional[str] = None
    api_name: Optional[str] = None
    inputs: Optional[List[str]] = None
    outputs: Optional[List[str]] = None
    
    # For traversal KB changes
    traversal_api_name: Optional[str] = None
    column_exclusion: Optional[bool] = None
    traversal_rule_name: Optional[str] = None  # Name of function to use
    
    # Metadata
    rationale: str = ""
    expected_impact: str = ""