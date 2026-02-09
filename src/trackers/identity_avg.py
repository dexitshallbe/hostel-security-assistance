from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple
import time


@dataclass
class IdentityDecision:
    ready: bool
    name: str
    avg_sim: float


class IdentityAverager:
    """
    Averages similarity for a stable identity over a small time window.
    Prevents attendance flip-flop from one noisy frame.
    """
    def __init__(self, window_sec: float = 2.0, min_avg_sim: float = 0.55, cooldown_sec: float = 10.0):
        self.window_sec = window_sec
        self.min_avg_sim = min_avg_sim
        self.cooldown_sec = cooldown_sec

        self.current_name: Optional[str] = None
        self.buf: Deque[Tuple[float, float]] = deque()  # (t, sim)
        self.last_decision_t = 0.0

    def update(self, name: str, sim: float) -> Optional[IdentityDecision]:
        now = time.time()
        if (now - self.last_decision_t) < self.cooldown_sec:
            return None

        if self.current_name is None:
            self.current_name = name
            self.buf.clear()

        if name != self.current_name:
            self.current_name = name
            self.buf.clear()

        self.buf.append((now, sim))
        while self.buf and (now - self.buf[0][0]) > self.window_sec:
            self.buf.popleft()

        if not self.buf:
            return None
        if (now - self.buf[0][0]) < (0.9 * self.window_sec):
            return IdentityDecision(False, name, 0.0)

        avg_sim = sum(x[1] for x in self.buf) / len(self.buf)
        if avg_sim >= self.min_avg_sim:
            self.last_decision_t = now
            self.current_name = None
            self.buf.clear()
            return IdentityDecision(True, name, avg_sim)

        return IdentityDecision(False, name, avg_sim)
