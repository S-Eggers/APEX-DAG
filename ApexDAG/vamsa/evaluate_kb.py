"""
KB Evaluation Framework
Automated testing and evaluation of Knowledge Base entries
"""
import ast
import os
import json
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd

from ApexDAG.vamsa.generate_wir import GenWIR
from ApexDAG.vamsa.annotate_wir import AnnotationWIR, KB
from ApexDAG.vamsa.track_provenance import track_provenance
from ApexDAG.vamsa.utils import remove_id
import logging

# At the top of your file, set the root logger level
logging.basicConfig(level=logging.WARNING)  # Shows WARNING, ERROR, CRITICAL only


@dataclass
class AnnotationMetrics:
    """Metrics for annotation KB evaluation"""
    total_operation_nodes: int = 0
    annotated_operation_nodes: int = 0
    total_nodes: int = 0
    annotated_nodes: int = 0
    annotation_coverage: float = 0.0
    operation_annotation_coverage: float = 0.0
    unsuccessful_operations: set = field(default_factory=set)
    
    # KB utilization
    kb_queries: int = 0
    kb_hits: int = 0
    kb_hit_rate: float = 0.0
    
    # Per-library coverage
    library_coverage: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Entry usage tracking
    kb_entry_usage: Dict[str, int] = field(default_factory=dict)
    # dead_entries: List[str] = field(default_factory=list)
    
    # Missed operations
    unannotated_operations: List[Tuple[str, str, str]] = field(default_factory=list)  # (library, caller, operation)


@dataclass
class TraversalMetrics:
    """Metrics for traversal KB evaluation"""
    total_tracked: int = 0
    c_plus_size: int = 0
    c_minus_size: int = 0
    
    # Traversal rule usage
    traversal_rule_usage: Dict[str, int] = field(default_factory=dict)
    # dead_rules: List[str] = field(default_factory=list)
    
    # Tracking depth
    max_traversal_depth: int = 0
    avg_traversal_depth: float = 0.0
    
    # Columns tracked
    columns_tracked: Set[str] = field(default_factory=set)


@dataclass
class KBEvaluationReport:
    """Complete evaluation report"""
    notebook_path: str
    annotation_metrics: AnnotationMetrics
    traversal_metrics: TraversalMetrics
    errors: List[str] = field(default_factory=list)


class InstrumentedKB(KB):
    """KB wrapper that tracks usage statistics"""
    
    def __init__(self, knowledge_base=None):
        super().__init__(knowledge_base)
        self.query_log = []
        self.hit_log = []
        self.entry_usage = defaultdict(int)
        self.unsuccessful_operations = set()
        
    def __call__(self, L, L_prime, c, p):
        # Log the query
        query = {
            "library": L,
            "module": L_prime,
            "caller": c,
            "operation": p
        }
        self.query_log.append(query)
        
        # Call parent method
        inputs, outputs = super().__call__(L, L_prime, c, p)
        
        # Log hit/miss
        is_hit = bool(inputs or outputs)
        self.hit_log.append(is_hit)
        
        if is_hit:
            # Track which entry was used
            entry_key = f"{L or 'None'}::{L_prime or 'None'}::{c or 'None'}::{p or 'None'}"
            self.entry_usage[entry_key] += 1
        else:
            self.unsuccessful_operations.add(str({'library': L, 'caller': c, 'operation': p.split(':')[0]}))        
        return inputs, outputs
    
    def back_query(self, O, p):
        # Track back queries too
        query = {
            "type": "back_query",
            "outputs": O,
            "operation": p
        }
        self.query_log.append(query)
        
        inputs = super().back_query(O, p)
        
        is_hit = bool(inputs)
        self.hit_log.append(is_hit)
        
        return inputs
    
    def get_stats(self):
        """Get usage statistics"""
        return {
            "total_queries": len(self.query_log),
            "hits": sum(self.hit_log),
            "misses": len(self.hit_log) - sum(self.hit_log),
            "hit_rate": sum(self.hit_log) / len(self.hit_log) if self.hit_log else 0,
            "entry_usage": dict(self.entry_usage),
            "unsuccessful_operations": list(self.unsuccessful_operations)
        }


class InstrumentedProvenanceTracker:
    """Wrapper to track traversal rule usage"""
    
    def __init__(self, tracker):
        self.tracker = tracker
        self.rule_usage = defaultdict(int)
        self.traversal_depth = []
        
    def track(self, what_track):
        """Track with instrumentation"""
        original_guide_eval = self.tracker._guide_eval
        depth_stack = [0]
        
        def instrumented_guide_eval(pr, col_excl=None):
            input_nodes, _, operation_node, _ = pr
            op_name = remove_id(operation_node) if operation_node else None
            
            # Track rule usage
            if op_name and op_name in self.tracker.kbc:
                self.rule_usage[op_name] += 1
            
            # Track depth
            depth_stack[0] += 1
            max_depth = depth_stack[0]
            
            result = original_guide_eval(pr, col_excl)
            
            depth_stack[0] -= 1
            
            if depth_stack[0] == 0:
                self.traversal_depth.append(max_depth)
            
            return result
        
        self.tracker._guide_eval = instrumented_guide_eval
        
        # Run tracking
        C_plus, C_minus = self.tracker.track(what_track)
        
        # Restore original method
        self.tracker._guide_eval = original_guide_eval
        
        return C_plus, C_minus
    
    def get_stats(self):
        """Get traversal statistics"""
        return {
            "rule_usage": dict(self.rule_usage),
            "max_depth": max(self.traversal_depth) if self.traversal_depth else 0,
            "avg_depth": sum(self.traversal_depth) / len(self.traversal_depth) if self.traversal_depth else 0,
            "total_traversals": len(self.traversal_depth)
        }


class KBEvaluator:
    """Main evaluation framework for KB entries"""
    
    def __init__(self, kb: Optional[KB] = None):
        self.kb = InstrumentedKB() # if kb is None else kb
        
    def evaluate_notebook(self, notebook_path: str, what_track: Set[str] = {"features"}) -> KBEvaluationReport:
        """Evaluate KB on a single notebook - NO TRY/EXCEPT, errors should propagate"""
        
        metrics = AnnotationMetrics()
        traversal_metrics = TraversalMetrics()
        errors = []
        
        # Read code
        code = self._read_code(notebook_path)
        if not code or not code.strip():
            raise ValueError(f"Empty code in {notebook_path}")
        
        parsed_ast = ast.parse(code)
        
        # Generate WIR with proper output path
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            output_filename = tmp.name
        
        wir, prs, tuples = GenWIR(
            parsed_ast,
            output_filename=output_filename,
            if_draw_graph=False
        )
        
        # Clean up temp file
        if os.path.exists(output_filename):
            os.remove(output_filename)
        txt_file = output_filename.replace(".png", ".txt")
        if os.path.exists(txt_file):
            os.remove(txt_file)
        
        input_nodes, output_nodes, caller_nodes, operation_nodes = tuples
        
        # Collect basic metrics
        metrics.total_nodes = len(wir.nodes)
        metrics.total_operation_nodes = len(operation_nodes)
        
        # Annotate WIR with instrumented KB
        annotated_wir = AnnotationWIR(wir, prs, self.kb)
        annotated_wir.annotate()
        
        # Collect annotation metrics
        metrics = self._collect_annotation_metrics(
            annotated_wir, 
            operation_nodes,
            metrics
        )
        
        # Track provenance with instrumentation
        from ApexDAG.vamsa.track_provenance import ProvenanceTracker
        tracker = ProvenanceTracker(annotated_wir, prs[::-1])
        instrumented_tracker = InstrumentedProvenanceTracker(tracker)
        
        C_plus, C_minus = instrumented_tracker.track(what_track)
        
        # Collect traversal metrics
        traversal_metrics = self._collect_traversal_metrics(
            C_plus, C_minus, instrumented_tracker, traversal_metrics
        )
        
        return KBEvaluationReport(
            notebook_path=notebook_path,
            annotation_metrics=metrics,
            traversal_metrics=traversal_metrics,
            errors=errors
        )
    
    def evaluate_corpus(self, corpus_path: str, what_track: Set[str] = {"features"}) -> List[KBEvaluationReport]:
        """Evaluate KB on multiple notebooks - with per-file error handling"""
        reports = []
        
        if os.path.isfile(corpus_path):
            files = [corpus_path]
        else:
            files = []
            for root, _, filenames in os.walk(corpus_path):
                for filename in filenames:
                    if filename.endswith((".py", ".ipynb")):
                        files.append(os.path.join(root, filename))
        
        
        for i, file_path in enumerate(files, 1):
            try:
                report = self.evaluate_notebook(file_path, what_track)
                reports.append(report)
            except Exception as e:
                error_report = KBEvaluationReport(
                    notebook_path=file_path,
                    annotation_metrics=AnnotationMetrics(),
                    traversal_metrics=TraversalMetrics(),
                    errors=[f"Evaluation failed: {str(e)}"]
                )
                reports.append(error_report)
        
        successful = len([r for r in reports if not r.errors])
        print(f"\n{'='*80}")
        print(f"Completed: {successful}/{len(files)} successful")
        print('='*80)
        
        return reports
    
    def compare_kbs(self, kb1: KB, kb2: KB, corpus_path: str) -> Dict:
        """Compare two KBs on the same corpus"""
        print("Evaluating KB 1 (baseline)...")
        self.kb = kb1 if isinstance(kb1, InstrumentedKB) else InstrumentedKB()
        self.kb.knowledge_base = kb1.knowledge_base
        reports1 = self.evaluate_corpus(corpus_path)
        
        print("\nEvaluating KB 2 (enhanced)...")
        self.kb = kb2 if isinstance(kb2, InstrumentedKB) else InstrumentedKB()
        self.kb.knowledge_base = kb2.knowledge_base
        reports2 = self.evaluate_corpus(corpus_path)
        
        return self._generate_comparison_report(reports1, reports2)
    
    def _read_code(self, path: str) -> str:
        """Read code from .py or .ipynb file"""
        if path.endswith(".ipynb"):
            from nbconvert import PythonExporter
            from nbformat import read
            with open(path, "r", encoding="utf-8") as f:
                nb_node = read(f, as_version=4)
                exporter = PythonExporter()
                file_content, _ = exporter.from_notebook_node(nb_node)
                return file_content
        else:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    
    def _collect_annotation_metrics(self, annotated_wir, operation_nodes, metrics: AnnotationMetrics) -> AnnotationMetrics:
        """Collect metrics from annotated WIR"""
        
        # Count annotated nodes
        metrics.annotated_nodes = sum(["annotations" in annotated_wir.annotated_wir.nodes[node] for node in annotated_wir.annotated_wir.nodes])
        metrics.annotated_operation_nodes = sum(["annotations" in annotated_wir.annotated_wir.nodes[node] for node in operation_nodes])
        
        # Calculate coverage
        if metrics.total_nodes > 0:
            metrics.annotation_coverage = metrics.annotated_nodes / metrics.total_nodes
        if metrics.total_operation_nodes > 0:
            metrics.operation_annotation_coverage = metrics.annotated_operation_nodes / metrics.total_operation_nodes
        
        # KB stats
        if isinstance(self.kb, InstrumentedKB):
            kb_stats = self.kb.get_stats()
            metrics.kb_queries = kb_stats["total_queries"]
            metrics.kb_hits = kb_stats["hits"]
            metrics.kb_hit_rate = kb_stats["hit_rate"]
            metrics.kb_entry_usage = kb_stats["entry_usage"]
            metrics.unannotated_operations = kb_stats["unsuccessful_operations"]
        return metrics
    
    def _collect_traversal_metrics(self, C_plus, C_minus, instrumented_tracker, metrics: TraversalMetrics) -> TraversalMetrics:
        """Collect traversal metrics"""
        
        metrics.c_plus_size = len(C_plus)
        metrics.c_minus_size = len(C_minus)
        metrics.total_tracked = metrics.c_plus_size + metrics.c_minus_size
        metrics.columns_tracked = C_plus.union(C_minus)
        # coverage
        
        
        # Get traversal stats
        stats = instrumented_tracker.get_stats()
        metrics.traversal_rule_usage = stats["rule_usage"]
        metrics.max_traversal_depth = stats["max_depth"]
        metrics.avg_traversal_depth = stats["avg_depth"]
        
        return metrics
    
    def _generate_comparison_report(self, reports1: List[KBEvaluationReport], reports2: List[KBEvaluationReport]) -> Dict:
        """Generate comparison report between two KB evaluations"""
        
        def aggregate_reports(reports):
            total_coverage = 0
            total_hit_rate = 0
            total_c_plus = 0
            total_c_minus = 0
            unsuccessful_ops = set()
            count = len(reports)
            
            all_entry_usage = defaultdict(int)
            all_rule_usage = defaultdict(int)
            
            for report in reports:
                total_coverage += report.annotation_metrics.operation_annotation_coverage
                total_hit_rate += report.annotation_metrics.kb_hit_rate
                total_c_plus += report.traversal_metrics.c_plus_size
                total_c_minus += report.traversal_metrics.c_minus_size
                unsuccessful_ops.update(report.annotation_metrics.unsuccessful_operations)
                
                for entry, usage in report.annotation_metrics.kb_entry_usage.items():
                    all_entry_usage[entry] += usage
                
                for rule, usage in report.traversal_metrics.traversal_rule_usage.items():
                    all_rule_usage[rule] += usage
            
            return {
                "avg_annotation_coverage": total_coverage / count if count > 0 else 0,
                "avg_kb_hit_rate": total_hit_rate / count if count > 0 else 0,
                "avg_c_plus_size": total_c_plus / count if count > 0 else 0,
                "avg_c_minus_size": total_c_minus / count if count > 0 else 0,
                "total_notebooks": count,
                "entry_usage": dict(all_entry_usage),
                "rule_usage": dict(all_rule_usage),
                "unsuccessful_operations": list(unsuccessful_ops)
            }
        
        agg1 = aggregate_reports(reports1)
        agg2 = aggregate_reports(reports2)
        
        return {
            "baseline": agg1,
            "enhanced": agg2,
            "improvements": {
                "annotation_coverage_delta": agg2["avg_annotation_coverage"] - agg1["avg_annotation_coverage"],
                "kb_hit_rate_delta": agg2["avg_kb_hit_rate"] - agg1["avg_kb_hit_rate"],
                "c_plus_size_delta": agg2["avg_c_plus_size"] - agg1["avg_c_plus_size"],
                "c_minus_size_delta": agg2["avg_c_minus_size"] - agg1["avg_c_minus_size"],
                "new_entries_used": len(set(agg2["entry_usage"].keys()) - set(agg1["entry_usage"].keys())),
                "new_rules_used": len(set(agg2["rule_usage"].keys()) - set(agg1["rule_usage"].keys()))
            }
        }
    
    def generate_report(self, reports: List[KBEvaluationReport], output_path: str):
        """Generate comprehensive evaluation report"""
        
        # Aggregate metrics
        total_notebooks = len(reports)
        successful_notebooks = len([r for r in reports if not r.errors])
        
        # Average metrics
        avg_annotation_coverage = sum(r.annotation_metrics.annotation_coverage for r in reports) / total_notebooks
        avg_kb_hit_rate = sum(r.annotation_metrics.kb_hit_rate for r in reports) / total_notebooks
        avg_c_plus = sum(r.traversal_metrics.c_plus_size for r in reports) / total_notebooks
        avg_c_minus = sum(r.traversal_metrics.c_minus_size for r in reports) / total_notebooks
        
        # Aggregate entry usage
        all_entry_usage = defaultdict(int)
        all_rule_usage = defaultdict(int)
        all_unannotated_ops = []
        
        for report in reports:
            for entry, count in report.annotation_metrics.kb_entry_usage.items():
                all_entry_usage[entry] += count
            for rule, count in report.traversal_metrics.traversal_rule_usage.items():
                all_rule_usage[rule] += count
            all_unannotated_ops.extend(report.annotation_metrics.unannotated_operations)
        
        # Find dead entries (assuming we have access to full KB)
        if isinstance(self.kb, InstrumentedKB):
            all_kb_entries = set()
            for _, row in self.kb.knowledge_base.iterrows():
                entry_key = f"{row['Library']}::{row.get('Module', 'None')}::{row.get('Caller', 'None')}::{row['API Name']}"
                all_kb_entries.add(entry_key)
            
            used_entries = set(all_entry_usage.keys())
            dead_entries = all_kb_entries - used_entries
        else:
            dead_entries = set()
        
        # Generate report
        report_data = {
            "summary": {
                "total_notebooks": total_notebooks,
                "successful_notebooks": successful_notebooks,
                "avg_annotation_coverage": avg_annotation_coverage,
                "avg_kb_hit_rate": avg_kb_hit_rate,
                "avg_c_plus_size": avg_c_plus,
                "avg_c_minus_size": avg_c_minus
            },
            # "kb_entry_usage": dict(sorted(all_entry_usage.items(), key=lambda x: x[1], reverse=True)),
            # "dead_entries": list(dead_entries),
            "traversal_rule_usage": dict(sorted(all_rule_usage.items(), key=lambda x: x[1], reverse=True)),
            "most_common_unannotated_operations": Counter(all_unannotated_ops).most_common(100),
            # "per_notebook_details": [
            #     {
            #         "path": r.notebook_path,
            #         "annotation_coverage": r.annotation_metrics.operation_annotation_coverage,
            #         "kb_hit_rate": r.annotation_metrics.kb_hit_rate,
            #         "c_plus_size": r.traversal_metrics.c_plus_size,
            #         "c_minus_size": r.traversal_metrics.c_minus_size,
            #         "errors": r.errors
            #     }
            #     for r in reports
            # ]
        }
        
        # Save report
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n{'='*80}")
        print("KB EVALUATION REPORT")
        print('='*80)
        print(f"Total Notebooks: {total_notebooks}")
        print(f"Successful: {successful_notebooks}")
        print(f"\nAnnotation Coverage: {avg_annotation_coverage}")
        print(f"KB Hit Rate: {avg_kb_hit_rate:.2%}")
        print(f"Avg Columns Tracked (C+): {avg_c_plus:.1f}")
        print(f"Avg Columns Excluded (C-): {avg_c_minus:.1f}")
        print(f"\nDead KB Entries: {len(dead_entries)}")
        print(f"Most Used KB Entries:")
        for entry, count in list(all_entry_usage.items())[:5]:
            print(f"  {entry}: {count}")
        print(f"\nMost Common Unannotated Operations:")
        for op, count in Counter([op[2] for op in all_unannotated_ops]).most_common(5):
            print(f"  {op}: {count}")
        print(f"\nFull report saved to: {output_path}")
        print('='*80)
        
        return report_data


if __name__ == "__main__":
    import sys
    
    # Configuration
    corpus_path = "C:\\Users\\ismyn\\UNI\\BIFOLD\\APEXDAG_datasets\\catboost"
    output_path = "output/kb_evaluation_report.json"
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    print("="*80)
    print("KB EVALUATION FRAMEWORK")
    print("="*80)
    print(f"Corpus: {corpus_path}")
    print(f"Output: {output_path}")
    print("="*80)
    
    # Create evaluator
    evaluator = KBEvaluator()
    
    # Evaluate on corpus
    reports = evaluator.evaluate_corpus(corpus_path)
    
    # Generate report
    evaluator.generate_report(reports, output_path)
