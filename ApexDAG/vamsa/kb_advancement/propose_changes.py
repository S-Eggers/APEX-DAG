"""
KB Change Proposal Framework
Propose, apply, and measure impact of KB changes
"""
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from copy import deepcopy
import pandas as pd

from ApexDAG.vamsa.annotate_wir import KB
from ApexDAG.vamsa.evaluate_kb import KBEvaluator
from ApexDAG.vamsa.kb_advancement.KBChangeProposal import KBChangeProposal
from ApexDAG.vamsa.kb_advancement.proposal_maker import ProposalMaker


@dataclass
class ChangeImpactReport:
    """Report showing impact of a KB change"""
    proposal: KBChangeProposal
    
    # Before metrics
    baseline_avg_annotation_coverage: float
    baseline_avg_kb_hit_rate: float
    baseline_avg_c_plus: float
    baseline_avg_c_minus: float
    baseline_dead_entries: int
    baseline_unannotated_ops: int
    
    # After metrics
    enhanced_avg_annotation_coverage: float
    enhanced_avg_kb_hit_rate: float
    enhanced_avg_c_plus: float
    enhanced_avg_c_minus: float
    enhanced_dead_entries: int
    enhanced_unannotated_ops: int
    
    # Deltas
    annotation_coverage_delta: float
    kb_hit_rate_delta: float
    c_plus_delta: float
    c_minus_delta: float
    
    # Impact assessment
    impact_level: str  # "HIGH", "MEDIUM", "LOW", "NEGATIVE"
    recommendation: str
    
    # Detailed reports
    baseline_report_path: str
    enhanced_report_path: str
    comparison_report_path: str


class KBChangeManager:
    """Manager for proposing and testing KB changes"""
    
    def __init__(self, baseline_kb: Optional[KB] = None, corpus_path: str = ""):
        self.baseline_kb = baseline_kb if baseline_kb else KB()
        self.corpus_path = corpus_path
        self.proposals: List[KBChangeProposal] = []
        self.impact_reports: List[ChangeImpactReport] = []
        
    def propose_annotation_entry(
        self,
        library: str,
        api_name: str,
        inputs: List[str],
        outputs: List[str],
        module: Optional[str] = None,
        caller: Optional[str] = None,
        rationale: str = "",
        expected_impact: str = ""
    ) -> KBChangeProposal:
        """Propose adding a new annotation KB entry"""
        proposal = KBChangeProposal(
            change_type="add_annotation",
            description=f"Add annotation entry: {library}.{module or ''}.{api_name}",
            library=library,
            module=module,
            caller=caller,
            api_name=api_name,
            inputs=inputs,
            outputs=outputs,
            rationale=rationale,
            expected_impact=expected_impact
        )
        self.proposals.append(proposal)
        return proposal
    
    def propose_traversal_rule(
        self,
        api_name: str,
        column_exclusion: bool,
        traversal_rule_name: str,
        library: Optional[str] = None,
        module: Optional[str] = None,
        caller: Optional[str] = None,
        rationale: str = "",
        expected_impact: str = ""
    ) -> KBChangeProposal:
        """Propose adding a new traversal rule"""
        proposal = KBChangeProposal(
            change_type="add_traversal",
            description=f"Add traversal rule for: {api_name}",
            library=library,
            module=module,
            caller=caller,
            traversal_api_name=api_name,
            column_exclusion=column_exclusion,
            traversal_rule_name=traversal_rule_name,
            rationale=rationale,
            expected_impact=expected_impact
        )
        self.proposals.append(proposal)
        return proposal
    
    def load_proposals_from_json(self, json_path: str):
        """Load proposals from a JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            proposal = KBChangeProposal(**item)
            self.proposals.append(proposal)
    
    def save_proposals_to_json(self, json_path: str):
        """Save proposals to a JSON file"""
        with open(json_path, 'w') as f:
            json.dump([asdict(p) for p in self.proposals], f, indent=2)
    
    def apply_proposal(self, proposalmaker: ProposalMaker, kb: KB, baseline_data: dict) -> KB:
        """Apply a proposal to create an enhanced KB"""
        enhanced_kb = deepcopy(kb)
        proposal = proposalmaker(KB=enhanced_kb, baseline_report=baseline_data)
        
        if proposal.change_type == "add_annotation":
            new_row = pd.DataFrame([{
                "Library": proposal.library,
                "Module": proposal.module,
                "Caller": proposal.caller,
                "API Name": proposal.api_name,
                "Inputs": proposal.inputs,
                "Outputs": proposal.outputs
            }])
            enhanced_kb.knowledge_base = pd.concat(
                [enhanced_kb.knowledge_base, new_row], 
                ignore_index=True
            )
        
        elif proposal.change_type == "add_traversal":
            new_row = pd.DataFrame([{
                "Library": proposal.library,
                "Module": proposal.module,
                "Caller": proposal.caller,
                "API Name": proposal.traversal_api_name,
                "Inputs": []  # Traversal KB has different structure
            }])
            enhanced_kb.knowledge_base_traversal = pd.concat(
                [enhanced_kb.knowledge_base_traversal, new_row],
                ignore_index=True
            )
            
            # Note: You'll need to manually add the actual traversal rule function
            # to the KBC dictionary in track_provenance.py
            print(f"WARNING: Traversal rule '{proposal.traversal_rule_name}' needs to be manually added to KBC in track_provenance.py")
        
        return enhanced_kb, proposal
    
    def evaluate_proposal(
        self, 
        proposal_object: KBChangeProposal,
        baseline_data: dict,
        output_dir: str = "output/proposals"
    ) -> ChangeImpactReport:
        """Evaluate the impact of a single proposal"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Sanitize filename
        safe_name = "temp_report"
        
        # Baseline evaluation
        print("Step 1/3: Evaluating baseline KB...")
        evaluator_baseline = KBEvaluator(self.baseline_kb)
        baseline_reports = evaluator_baseline.evaluate_corpus(self.corpus_path)
        baseline_report_path = os.path.join(output_dir, f"{safe_name}_baseline.json")
        baseline_data = evaluator_baseline.generate_report(baseline_reports, baseline_report_path)
        
        # get the baseline reports and pass it as input to the proposal maker
        
        # Apply proposal
        print("\nStep 2/3: Applying proposal and creating enhanced KB...")
        enhanced_kb, proposal = self.apply_proposal(proposal_object, self.baseline_kb, baseline_data)
        
        # Enhanced evaluation
        print("\nStep 3/3: Evaluating enhanced KB...")
        evaluator_enhanced = KBEvaluator(enhanced_kb)
        enhanced_reports = evaluator_enhanced.evaluate_corpus(self.corpus_path)
        enhanced_report_path = os.path.join(output_dir, f"{safe_name}_enhanced.json")
        enhanced_data = evaluator_enhanced.generate_report(enhanced_reports, enhanced_report_path)
        
        # Calculate deltas
        annotation_coverage_delta = (
            enhanced_data["summary"]["avg_annotation_coverage"] - 
            baseline_data["summary"]["avg_annotation_coverage"]
        )
        kb_hit_rate_delta = (
            enhanced_data["summary"]["avg_kb_hit_rate"] - 
            baseline_data["summary"]["avg_kb_hit_rate"]
        )
        c_plus_delta = (
            enhanced_data["summary"]["avg_c_plus_size"] - 
            baseline_data["summary"]["avg_c_plus_size"]
        )
        c_minus_delta = (
            enhanced_data["summary"]["avg_c_minus_size"] - 
            baseline_data["summary"]["avg_c_minus_size"]
        )
        
        # Assess impact level
        impact_level = self._assess_impact_level(
            annotation_coverage_delta,
            kb_hit_rate_delta,
            c_plus_delta,
            c_minus_delta
        )
        
        recommendation = self._generate_recommendation(
            impact_level,
            annotation_coverage_delta,
            kb_hit_rate_delta,
            proposal
        )
        
        # Create impact report
        impact_report = ChangeImpactReport(
            proposal=proposal,
            baseline_avg_annotation_coverage=baseline_data["summary"]["avg_annotation_coverage"],
            baseline_avg_kb_hit_rate=baseline_data["summary"]["avg_kb_hit_rate"],
            baseline_avg_c_plus=baseline_data["summary"]["avg_c_plus_size"],
            baseline_avg_c_minus=baseline_data["summary"]["avg_c_minus_size"],
            baseline_unannotated_ops=len(baseline_data["most_common_unannotated_operations"]),
            enhanced_avg_annotation_coverage=enhanced_data["summary"]["avg_annotation_coverage"],
            enhanced_avg_kb_hit_rate=enhanced_data["summary"]["avg_kb_hit_rate"],
            enhanced_avg_c_plus=enhanced_data["summary"]["avg_c_plus_size"],
            enhanced_avg_c_minus=enhanced_data["summary"]["avg_c_minus_size"],
            enhanced_unannotated_ops=len(enhanced_data["most_common_unannotated_operations"]),
            annotation_coverage_delta=annotation_coverage_delta,
            kb_hit_rate_delta=kb_hit_rate_delta,
            c_plus_delta=c_plus_delta,
            c_minus_delta=c_minus_delta,
            impact_level=impact_level,
            recommendation=recommendation,
            baseline_report_path=baseline_report_path,
            enhanced_report_path=enhanced_report_path,
            comparison_report_path=os.path.join(output_dir, f"{safe_name}_comparison.json")
        )
        
        # Save comparison report
        comparison_data = {
            "proposal": asdict(proposal),
            "impact": {
                "annotation_coverage_delta": annotation_coverage_delta,
                "kb_hit_rate_delta": kb_hit_rate_delta,
                "c_plus_delta": c_plus_delta,
                "c_minus_delta": c_minus_delta,
                "impact_level": impact_level,
                "recommendation": recommendation
            },
            "baseline_summary": baseline_data["summary"],
            "enhanced_summary": enhanced_data["summary"]
        }
        
        with open(impact_report.comparison_report_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        self.impact_reports.append(impact_report)
        
        # Print summary
        self._print_impact_summary(impact_report)
        
        if annotation_coverage_delta < 0 or kb_hit_rate_delta < 0:
            print("WARNING: The proposed change resulted in a negative impact on KB performance.")
        
        elif annotation_coverage_delta >= 0:
            print("The proposed change improved KB performance.")
        
        return impact_report
    
    def evaluate_all_proposals(self, output_dir: str = "output/proposals") -> List[ChangeImpactReport]:
        """Evaluate all pending proposals"""
        print(f"\n{'='*80}")
        print(f"EVALUATING {len(self.proposals)} PROPOSALS")
        print(f"{'='*80}\n")
        
        impact_reports = []
        for i, proposal in enumerate(self.proposals, 1):
            print(f"\n[{i}/{len(self.proposals)}] Processing proposal...")
            try:
                report = self.evaluate_proposal(proposal, output_dir)
                impact_reports.append(report)
            except Exception as e:
                print(f"ERROR evaluating proposal: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate summary report
        self._generate_summary_report(impact_reports, output_dir)
        
        return impact_reports
    
    def _assess_impact_level(
        self, 
        annotation_delta: float,
        hit_rate_delta: float,
        c_plus_delta: float,
        c_minus_delta: float
    ) -> str:
        """Assess the impact level of a change"""
        
        # Negative impact
        if annotation_delta < -0.01 or hit_rate_delta < -0.01:
            return "NEGATIVE"
        
        # High impact: significant improvement in coverage or hit rate
        if annotation_delta > 0.1 or hit_rate_delta > 0.1:
            return "HIGH"
        
        # Medium impact: moderate improvement
        if annotation_delta > 0.05 or hit_rate_delta > 0.05 or abs(c_plus_delta) > 5:
            return "MEDIUM"
        
        # Low impact: minimal change
        return "LOW"
    
    def _generate_recommendation(
        self,
        impact_level: str,
        annotation_delta: float,
        hit_rate_delta: float,
        proposal: KBChangeProposal
    ) -> str:
        """Generate recommendation based on impact"""
        
        if impact_level == "NEGATIVE":
            return f"❌ REJECT - This change decreases performance (coverage: {annotation_delta:+.2%}, hit rate: {hit_rate_delta:+.2%})"
        
        if impact_level == "HIGH":
            return f"✅ STRONGLY RECOMMEND - Significant improvement (coverage: {annotation_delta:+.2%}, hit rate: {hit_rate_delta:+.2%})"
        
        if impact_level == "MEDIUM":
            return f"✅ RECOMMEND - Moderate improvement (coverage: {annotation_delta:+.2%}, hit rate: {hit_rate_delta:+.2%})"
        
        if impact_level == "LOW":
            return f"⚠️ OPTIONAL - Minimal impact (coverage: {annotation_delta:+.2%}, hit rate: {hit_rate_delta:+.2%}). Consider if this entry will be used more in future datasets."
        
        return "UNKNOWN"
    
    def _print_impact_summary(self, report: ChangeImpactReport):
        """Print a summary of the impact report"""
        print(f"\n{'='*80}")
        print(f"IMPACT SUMMARY")
        print(f"{'='*80}")
        print(f"Proposal: {report.proposal.description}")
        print(f"Impact Level: {report.impact_level}")
        print(f"\nMetric Changes:")
        print(f"  Annotation Coverage: {report.baseline_avg_annotation_coverage:.2%} → {report.enhanced_avg_annotation_coverage:.2%} ({report.annotation_coverage_delta:+.2%})")
        print(f"  KB Hit Rate:         {report.baseline_avg_kb_hit_rate:.2%} → {report.enhanced_avg_kb_hit_rate:.2%} ({report.kb_hit_rate_delta:+.2%})")
        print(f"  Avg C+ Size:         {report.baseline_avg_c_plus:.1f} → {report.enhanced_avg_c_plus:.1f} ({report.c_plus_delta:+.1f})")
        print(f"  Avg C- Size:         {report.baseline_avg_c_minus:.1f} → {report.enhanced_avg_c_minus:.1f} ({report.c_minus_delta:+.1f})")
        print(f"\nRecommendation:")
        print(f"  {report.recommendation}")
        print(f"\nReports saved:")
        print(f"  Baseline: {report.baseline_report_path}")
        print(f"  Enhanced: {report.enhanced_report_path}")
        print(f"  Comparison: {report.comparison_report_path}")
        print(f"{'='*80}\n")
    
    def _generate_summary_report(self, reports: List[ChangeImpactReport], output_dir: str):
        """Generate a summary report of all proposals"""
        
        summary_path = os.path.join(output_dir, "all_proposals_summary.json")
        
        summary_data = {
            "total_proposals": len(reports),
            "by_impact_level": {
                "HIGH": len([r for r in reports if r.impact_level == "HIGH"]),
                "MEDIUM": len([r for r in reports if r.impact_level == "MEDIUM"]),
                "LOW": len([r for r in reports if r.impact_level == "LOW"]),
                "NEGATIVE": len([r for r in reports if r.impact_level == "NEGATIVE"])
            },
            "proposals": [
                {
                    "description": r.proposal.description,
                    "impact_level": r.impact_level,
                    "annotation_coverage_delta": r.annotation_coverage_delta,
                    "kb_hit_rate_delta": r.kb_hit_rate_delta,
                    "recommendation": r.recommendation,
                    "comparison_report": r.comparison_report_path
                }
                for r in reports
            ]
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"ALL PROPOSALS SUMMARY")
        print(f"{'='*80}")
        print(f"Total Proposals: {len(reports)}")
        print(f"\nBy Impact Level:")
        for level, count in summary_data["by_impact_level"].items():
            print(f"  {level}: {count}")
        print(f"\nTop 5 Recommendations:")
        sorted_reports = sorted(reports, key=lambda r: r.annotation_coverage_delta, reverse=True)
        for i, report in enumerate(sorted_reports[:5], 1):
            print(f"  {i}. {report.proposal.description}")
            print(f"     Impact: {report.impact_level} (coverage: {report.annotation_coverage_delta:+.2%})")
        print(f"\nSummary saved to: {summary_path}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # Example usage
    corpus_path = "C:\\Users\\ismyn\\UNI\\BIFOLD\\APEXDAG_datasets\\catboost"
    documentation_link = "https://catboost.ai/docs/en/concepts/python-reference_catboost"
    output_dir = "output/proposals"
    
    print("="*80)
    print("KB CHANGE PROPOSAL FRAMEWORK")
    print("="*80)
    
    # Create manager
    running_KB = KB()
    for iteration in range(1):
        print(f"\n--- ITERATION {iteration+1} ---\n")
        manager = KBChangeManager(baseline_kb=running_KB, corpus_path=corpus_path)
        
        # Propose a change
        proposal_object = ProposalMaker(link_to_documentation=documentation_link) # takes in data, scrapes documentation and outputs a proposal object after querying an llm
        
        # Evaluate impact
        report = manager.evaluate_proposal(proposal_object, output_dir)
        
    print("\n=== DONE ===")
    print(f"Results in: {output_dir}")
