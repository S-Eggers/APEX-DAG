import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from ApexDAG.notebook import Notebook
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph


def analyze_notebook(notebook_path: str) -> dict:
    """
    Analyze a single notebook and return stats.
    Returns dict with 'success', 'error', 'node_count', 'edge_count'.
    """
    result = {
        "success": False,
        "error": None,
        "node_count": 0,
        "edge_count": 0,
    }
    
    try:
        Notebook.VERBOSE = False
        DataFlowGraph.VERBOSE = False
        
        notebook = Notebook(notebook_path, cell_window_size=5)
        notebook.create_execution_graph(greedy=True)
        
        dfg = DataFlowGraph(replace_dataflow=False)
        dfg.parse_notebook(notebook)
        dfg.optimize()
        
        G = dfg.get_graph()
        result["success"] = True
        result["node_count"] = G.number_of_nodes()
        result["edge_count"] = G.number_of_edges()
        
    except Exception as e:
        result["error"] = str(e)[:100]  # Truncate long errors
    
    return result


def analyze_dataset(root_path: str, output_file: str = "analysis_report.txt") -> None:
    """
    Analyze all notebooks in the dataset and generate a report.
    """
    root = Path(root_path)
    
    if not root.exists():
        print(f"Error: Path '{root_path}' does not exist.")
        sys.exit(1)
    
    # Stats storage
    library_stats = defaultdict(lambda: {
        "total": 0,
        "success": 0,
        "errors": 0,
        "node_counts": [],
        "error_messages": defaultdict(int),
    })
    
    # Find all notebooks
    notebooks = list(root.rglob("*.ipynb"))
    total_notebooks = len(notebooks)
    
    print(f"Found {total_notebooks} notebooks in {root_path}")
    print("=" * 60)
    
    # Process each notebook
    for i, nb_path in enumerate(notebooks, 1):
        # Get library name from parent directory
        library = nb_path.parent.name
        
        print(f"[{i}/{total_notebooks}] {library}/{nb_path.name[:40]}...", end=" ", flush=True)
        
        result = analyze_notebook(str(nb_path))
        
        library_stats[library]["total"] += 1
        
        if result["success"]:
            library_stats[library]["success"] += 1
            library_stats[library]["node_counts"].append(result["node_count"])
            print(f"✓ ({result['node_count']} nodes)")
        else:
            library_stats[library]["errors"] += 1
            # Group similar errors
            error_key = result["error"][:50] if result["error"] else "Unknown"
            library_stats[library]["error_messages"][error_key] += 1
            print(f"✗ ({error_key[:30]}...)")
    
    # Generate report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("NOTEBOOK DATASET ANALYSIS REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Dataset: {root_path}")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Overall summary
    total_success = sum(s["success"] for s in library_stats.values())
    total_errors = sum(s["errors"] for s in library_stats.values())
    all_node_counts = []
    for s in library_stats.values():
        all_node_counts.extend(s["node_counts"])
    
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total notebooks:     {total_notebooks}")
    report_lines.append(f"Successful:          {total_success} ({100*total_success/max(total_notebooks,1):.1f}%)")
    report_lines.append(f"Errors:              {total_errors} ({100*total_errors/max(total_notebooks,1):.1f}%)")
    report_lines.append("")
    
    if all_node_counts:
        report_lines.append("NODE COUNT DISTRIBUTION (ALL)")
        report_lines.append("-" * 40)
        report_lines.append(f"Min:     {min(all_node_counts)}")
        report_lines.append(f"Max:     {max(all_node_counts)}")
        report_lines.append(f"Mean:    {sum(all_node_counts)/len(all_node_counts):.1f}")
        sorted_counts = sorted(all_node_counts)
        report_lines.append(f"Median:  {sorted_counts[len(sorted_counts)//2]}")
        report_lines.append("")
        
        # Histogram buckets
        buckets = [0, 10, 25, 50, 100, 250, 500, 1000, float('inf')]
        bucket_labels = ["0-9", "10-24", "25-49", "50-99", "100-249", "250-499", "500-999", "1000+"]
        bucket_counts = [0] * len(bucket_labels)
        
        for count in all_node_counts:
            for j in range(len(buckets) - 1):
                if buckets[j] <= count < buckets[j + 1]:
                    bucket_counts[j] += 1
                    break
        
        report_lines.append("NODE COUNT HISTOGRAM")
        report_lines.append("-" * 40)
        max_bar = 40
        max_count = max(bucket_counts) if bucket_counts else 1
        for label, count in zip(bucket_labels, bucket_counts):
            bar_len = int(max_bar * count / max_count) if max_count > 0 else 0
            bar = "█" * bar_len
            report_lines.append(f"{label:>10}: {count:4d} {bar}")
        report_lines.append("")
    
    # Per-library breakdown
    report_lines.append("PER-LIBRARY BREAKDOWN")
    report_lines.append("=" * 70)
    
    for library in sorted(library_stats.keys()):
        stats = library_stats[library]
        report_lines.append("")
        report_lines.append(f"📚 {library.upper()}")
        report_lines.append("-" * 40)
        report_lines.append(f"  Total:     {stats['total']}")
        report_lines.append(f"  Success:   {stats['success']} ({100*stats['success']/max(stats['total'],1):.1f}%)")
        report_lines.append(f"  Errors:    {stats['errors']} ({100*stats['errors']/max(stats['total'],1):.1f}%)")
        
        if stats["node_counts"]:
            nc = stats["node_counts"]
            report_lines.append(f"  Nodes:     min={min(nc)}, max={max(nc)}, mean={sum(nc)/len(nc):.1f}")
        
        if stats["error_messages"]:
            report_lines.append("  Top errors:")
            for err, cnt in sorted(stats["error_messages"].items(), key=lambda x: -x[1])[:3]:
                report_lines.append(f"    - ({cnt}x) {err}...")
    
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)
    
    # Write report
    report_text = "\n".join(report_lines)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print("\n" + "=" * 60)
    print(report_text)
    print(f"\nReport saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze notebook dataset and report errors/node distributions"
    )
    parser.add_argument(
        "--path",
        help="Path to the root of the notebook dataset",
        default = "/home/nina/projects/APEX-DAG/data/notebook_data/APEXDAG Datasets"
    )
    parser.add_argument(
        "-o", "--output",
        default="analysis_report.txt",
        help="Output report file (default: analysis_report.txt)"
    )
    
    args = parser.parse_args()
    analyze_dataset(args.path, args.output)


if __name__ == "__main__":
    main()