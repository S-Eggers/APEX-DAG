import threading
import time
from collections import deque


class ExecutionPolicy:
    """
    Orchestrates both financial (Total Tokens) and technical (RPM) constraints.
    Thread-safe and designed for concurrent workers.
    """

    def __init__(self, max_tokens: int, max_rpm: int) -> None:
        self.max_tokens = max_tokens
        self.max_rpm = max_rpm

        # Token state
        self._used_tokens: int = 0
        self._token_lock = threading.Lock()

        # Rate limit state
        self._request_timestamps: deque[float] = deque()
        self._rpm_lock = threading.Lock()

        self.stop_event = threading.Event()

    def wait_for_slot(self) -> None:
        """
        Blocks the calling thread until a request slot is available
        based on the RPM limit.
        """
        while not self.stop_event.is_set():
            with self._rpm_lock:
                now = time.time()
                # Remove timestamps older than 60 seconds
                while self._request_timestamps and now - self._request_timestamps[0] > 60:
                    self._request_timestamps.popleft()

                if len(self._request_timestamps) < self.max_rpm:
                    self._request_timestamps.append(now)
                    return

            # Slot not available, back off briefly before checking again
            time.sleep(0.1)

    def record_usage(self, token_count: int) -> None:
        """Updates the global token budget and halts if exceeded."""
        with self._token_lock:
            self._used_tokens += token_count
            if self._used_tokens >= self.max_tokens:
                self.stop_event.set()

    @property
    def total_used(self) -> int:
        with self._token_lock:
            return self._used_tokens
