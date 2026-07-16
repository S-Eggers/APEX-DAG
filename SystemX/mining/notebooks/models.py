from dataclasses import dataclass, field

@dataclass
class ValidationMetrics:
    success: bool
    edge_count: int = 0
    node_count: int = 0
    lines_of_code: int = 0
    contains_ml_semantics: bool = False
    extraction_time_sec: float = -1.0
    error_type: str | None = None
    stacktrace: str | None = None
    matched_libraries: set[str] = field(default_factory=set)
