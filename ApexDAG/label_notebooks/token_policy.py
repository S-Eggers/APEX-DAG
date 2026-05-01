import threading


class TokenBudgetPolicy:
    """Thread-safe state management for API spending."""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self._used_tokens = 0
        self._lock = threading.Lock()
        self.stop_event = threading.Event()

    def record_usage(self, count: int) -> None:
        with self._lock:
            self._used_tokens += count
            if self._used_tokens >= self.max_tokens:
                self.stop_event.set()

    @property
    def total_used(self) -> int:
        return self._used_tokens
