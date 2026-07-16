import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TextSpan:
    """Represents a spatial boundary in a source file."""

    start_line: int
    start_col: int
    end_line: int
    end_col: int

    def is_valid(self) -> bool:
        return all(v is not None and v >= 0 for v in (self.start_line, self.start_col, self.end_line, self.end_col))

    def _to_1d_interval(self, max_line_length: int = 2000) -> tuple[int, int]:
        """Flattens 2D text coordinates into a 1D interval for easy overlap calculation."""
        start = (self.start_line * max_line_length) + self.start_col
        end = (self.end_line * max_line_length) + self.end_col
        return start, end

    def overlap_coefficient(self, other: "TextSpan") -> float:
        """Calculates the Overlap Coefficient (Szymkiewicz-Simpson)."""
        if not self.is_valid() or not other.is_valid():
            return 0.0

        s_start, s_end = self._to_1d_interval()
        o_start, o_end = other._to_1d_interval()

        intersection_start = max(s_start, o_start)
        intersection_end = min(s_end, o_end)

        if intersection_start >= intersection_end:
            return 0.0

        intersection_len = intersection_end - intersection_start
        s_len = s_end - s_start
        o_len = o_end - o_start

        min_len = min(s_len, o_len)
        if min_len <= 0:
            return 0.0

        return intersection_len / min_len
