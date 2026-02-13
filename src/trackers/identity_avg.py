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
    Stabilizes identity across a small window.

    Key changes:
    - Unknown is treated as "no vote" (doesn't reset the buffer)
    - Name changes reset ONLY when the new name is a real identity
    """
    def __init__(self, window_sec: float = 2.0, min_avg_sim: float = 0.68, cooldown_sec: float = 10.0):
        self.window_sec = float(window_sec)
        self.min_avg_sim = float(min_avg_sim)
        self.cooldown_sec = float(cooldown_sec)

        self.current_name: Optional[str] = None
        self.buf: Deque[Tuple[float, float]] = deque()
        self.last_decision_t = 0.0

    def reset(self):
        self.current_name = None
        self.buf.clear()

    def update(self, name: str, sim: float) -> Optional[IdentityDecision]:
        now = time.time()

        # Cooldown after a decision
        if (now - self.last_decision_t) < self.cooldown_sec:
            return None

        # Ignore Unknown as "no vote"
        if name == "Unknown":
            # keep buffer; do not reset
            return None

        if self.current_name is None:
            self.current_name = name
            self.buf.clear()

        # Only reset when flipping between two real names
        if name != self.current_name:
            self.current_name = name
            self.buf.clear()

        self.buf.append((now, sim))
        while self.buf and (now - self.buf[0][0]) > self.window_sec:
            self.buf.popleft()

        if not self.buf:
            return None

        # require enough window coverage
        if (now - self.buf[0][0]) < (0.9 * self.window_sec):
            avg_sim = sum(x[1] for x in self.buf) / len(self.buf)
            return IdentityDecision(False, self.current_name, avg_sim)

        avg_sim = sum(x[1] for x in self.buf) / len(self.buf)

        if avg_sim >= self.min_avg_sim:
            self.last_decision_t = now
            decided_name = self.current_name
            self.reset()
            return IdentityDecision(True, decided_name, avg_sim)

        return IdentityDecision(False, self.current_name, avg_sim)