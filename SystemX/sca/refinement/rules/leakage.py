from __future__ import annotations

import networkx as nx

from SystemX.sca.leakage import analyze_leakage

from ..interfaces import GraphRefinementRule
from ..state import RefinementState

class LeakageAnalysisRule(GraphRefinementRule):
    """Runs D1-D5 static leakage detectors and records per-class findings."""

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None:
        findings = analyze_leakage(g)
        state.leakage_findings.extend(findings)
        for f in findings:
            state.leakage_nodes.add(f.node)
