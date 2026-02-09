from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple
import time


@dataclass
class UnknownDecision:
    should_trigger: bool
    avg_face_prob: float
    avg_best_sim: float


class UnknownAverager:
    def __init__(self, window_sec: float = 2.0, cooldown_sec: float = 45.0):
        self.window_sec = window_sec
        self.cooldown_sec = cooldown_sec
        self.buf: Deque[Tuple[float, float, float]] = deque()  # (t, face_prob, best_sim)
        self.last_trigger_time = 0.0

    def reset(self):
        self.buf.clear()

    def update(self, face_prob: float, best_sim: float) -> Optional[UnknownDecision]:
        now = time.time()

        # cooldown
        if (now - self.last_trigger_time) < self.cooldown_sec:
            return UnknownDecision(False, 0.0, 1.0)

        self.buf.append((now, face_prob, best_sim))
        while self.buf and (now - self.buf[0][0]) > self.window_sec:
            self.buf.popleft()

        if not self.buf:
            return None
        if (now - self.buf[0][0]) < (0.9 * self.window_sec):
            return None

        avg_face_prob = sum(x[1] for x in self.buf) / len(self.buf)
        avg_best_sim = sum(x[2] for x in self.buf) / len(self.buf)
        return UnknownDecision(True, avg_face_prob, avg_best_sim)

    def mark_triggered(self):
        self.last_trigger_time = time.time()
        self.reset()
