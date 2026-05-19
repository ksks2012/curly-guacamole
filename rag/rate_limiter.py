"""Thread-safe sliding-window rate limiter (shared across all API clients)."""

import threading
import time
from collections import deque


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Blocks the calling thread until a request slot is available within the
    rolling one-minute window.  Pass ``requests_per_minute=0`` to disable.
    """

    def __init__(self, requests_per_minute: int = 0) -> None:
        self._rpm = requests_per_minute
        self._window = 60.0  # seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is available, then consume it."""
        if self._rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                # Drop timestamps outside the rolling window
                while self._timestamps and self._timestamps[0] <= now - self._window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._rpm:
                    self._timestamps.append(now)
                    return
                # Wait until the oldest slot falls out of the window
                wait = self._timestamps[0] + self._window - now
            time.sleep(max(wait, 0.05))
