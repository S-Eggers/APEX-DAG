import logging
import random
import threading
import time
from collections import deque

from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class ExecutionPolicy:
    def __init__(self, max_tokens: int, max_rpm: int) -> None:
        self.max_tokens = max_tokens
        self.max_rpm = max_rpm
        self._used_tokens: int = 0
        self._token_lock = threading.Lock()
        self._request_timestamps: deque[float] = deque()
        self._rpm_lock = threading.Lock()
        self.stop_event = threading.Event()

    def wait_for_slot(self) -> None:
        thread_id = threading.get_ident()
        logger.debug(f"[Thread {thread_id}] Requesting RPM slot...")

        start_time = time.time()
        while not self.stop_event.is_set():
            with self._rpm_lock:
                now = time.time()
                while self._request_timestamps and now - self._request_timestamps[0] > 60:
                    self._request_timestamps.popleft()

                if len(self._request_timestamps) < self.max_rpm:
                    self._request_timestamps.append(now)
                    wait_duration = time.time() - start_time
                    logger.debug(f"[Thread {thread_id}] Slot acquired after {wait_duration:.2f}s. Current window: {len(self._request_timestamps)}/{self.max_rpm}")
                    return

            time.sleep(0.5 + random.uniform(0, 0.1))

    def record_usage(self, token_count: int) -> None:
        with self._token_lock:
            self._used_tokens += token_count
            logger.info(f"Budget Update: {self._used_tokens}/{self.max_tokens} tokens used.")
            if self._used_tokens >= self.max_tokens:
                logger.critical("TOKEN BUDGET EXHAUSTED. Setting stop event.")
                self.stop_event.set()
