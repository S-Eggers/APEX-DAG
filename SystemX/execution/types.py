from dataclasses import dataclass, field
from enum import Enum


class CellStatus(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    DIRTY = "dirty"
    UNEXECUTED = "unexecuted"
    MISSING_DEPS = "missing-deps"
    UNKNOWN = "unknown"


@dataclass
class CellState:
    cell_id: str
    status: CellStatus
    reasons: list[str] = field(default_factory=list)
    predicted_rank: int = -1
    confidence: float = 1.0
    unsafe_reorder: bool = False
    hidden_state_risk: bool = False

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "status": self.status.value,
            "reasons": self.reasons,
            "predicted_rank": self.predicted_rank,
            "confidence": round(self.confidence, 4),
            "unsafe_reorder": self.unsafe_reorder,
            "hidden_state_risk": self.hidden_state_risk,
        }


@dataclass
class ExecutionStateReport:
    predicted_order: list[str]
    constraints: list[dict]
    ambiguities: list[dict]
    cell_states: dict[str, CellState]
    notebook_flags: dict

    def to_dict(self) -> dict:
        return {
            "predicted_order": self.predicted_order,
            "constraints": self.constraints,
            "ambiguities": self.ambiguities,
            "cell_states": {cell: state.to_dict() for cell, state in self.cell_states.items()},
            "notebook_flags": self.notebook_flags,
        }
