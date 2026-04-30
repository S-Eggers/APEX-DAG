import hashlib

from ApexDAG.mining.notebooks.validator import ValidationMetrics


class GraphSamplingPolicy:
    """
    Deterministic, density-aware sampling policy.
    Ensures dataset reproducibility and prioritizes high-signal ML graphs.
    """

    @staticmethod
    def _deterministic_probability(identifier: str) -> float:
        """
        Hashes the unique filename to generate a reproducible float between 0.0 and 1.0.
        Guarantees the exact same notebooks are kept across pipeline restarts.
        """
        hash_int = int(hashlib.md5(identifier.encode("utf-8")).hexdigest(), 16)
        return (hash_int % 10000) / 10000.0

    @staticmethod
    def evaluate(metrics: ValidationMetrics, filename: str) -> tuple[bool, str]:
        if not metrics.success:
            return False, f"Failed: {metrics.error_type}"

        num_edges = metrics.edge_count

        if num_edges < 5:
            return False, "Dropped: Too Few Edges (<5)"

        density = num_edges / max(metrics.lines_of_code, 1)
        # 1. High-Value Signal
        if metrics.contains_ml_semantics:
            return True, f"Kept: ML Semantics (Density: {density:.3f})"

        # 2. Graph Density Check
        if density < 0.05:
            return False, f"Dropped: Low Density ({density:.3f})"

        # 3. Deterministic Size-Based Sampling
        probability = GraphSamplingPolicy._deterministic_probability(filename)

        if num_edges < 50:
            keep = probability < 0.3
            return (
                keep,
                "Kept: Small Graph (30%)" if keep else "Dropped: Small Graph Sample",
            )
        elif 50 <= num_edges <= 250:
            keep = probability < 0.8
            return (
                keep,
                "Kept: Mid Graph (80%)" if keep else "Dropped: Mid Graph Sample",
            )
        else:
            keep = probability < 0.5
            return (
                keep,
                "Kept: Large Graph (50%)" if keep else "Dropped: Large Graph Sample",
            )
