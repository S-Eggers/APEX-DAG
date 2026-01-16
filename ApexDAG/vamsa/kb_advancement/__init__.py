"""
KB Advancement Submodule
Tools for evolving and improving the Knowledge Base
"""

from .propose_changes import KBChangeManager, KBChangeProposal, ChangeImpactReport
from .suggest_entries import KBEntrySuggester
from .extract_patterns import PatternExtractor

__all__ = [
    'KBChangeManager',
    'KBChangeProposal', 
    'ChangeImpactReport',
    'KBEntrySuggester'
]
