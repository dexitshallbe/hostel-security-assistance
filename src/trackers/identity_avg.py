# src/trackers/identity_avg.py
# K-of-N vote based identity stabilizer (works great for "don't let 1 lucky frame decide")
#
# Usage:
#   decision = identity_avg.update(name, sim)
#   if decision and decision.ready:
#       print(decision.name, decision.avg_sim)

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple
import time


@dataclass
class IdentityDecision:
    ready: bool
    name: str
    avg_sim: float
    votes: int
    total: int


class IdentityAverager:
    """
    Stabilizes identity across a rolling time window using K-of-N votes.

    - Unknown is ignored (no vote)
    - We collect (t, name, sim) samples for a window (window_sec)
    - Decision triggers when:
        * enough samples exist (min_samples)
        * top_name has votes >= k_votes
        * avg_sim(top_name) >= min_avg_sim
        * optional cooldown is respected
    - When a decision triggers, internal buffer resets (so it can decide again later)
    """

    def __init__(
        self,
        window_sec: float = 2.0,
        min_avg_sim: float = 0.78,
        cooldown_sec: float = 10.0,
        min_samples: int = 8,
        k_votes: int = 6,
        require_consecutive: bool = False,
    ):
        self.window_sec = float(window_sec)
        self.min_avg_sim = float(min_avg_sim)
        self.cooldown_sec = float(cooldown_sec)

        self.min_samples = int(min_samples)
        self.k_votes = int(k_votes)
        self.require_consecutive = bool(require_consecutive)

        self.buf: Deque[Tuple[float, str, float]] = deque()
        self.last_decision_t = 0.0

    def reset(self):
        self.buf.clear()

    def _prune(self, now: float):
        while self.buf and (now - self.buf[0][0]) > self.window_sec:
            self.buf.popleft()

    def update(self, name: str, sim: float) -> Optional[IdentityDecision]:
        now = time.time()

        # Cooldown after a decision
        if (now - self.last_decision_t) < self.cooldown_sec:
            return None

        # Ignore Unknown as "no vote"
        if not name or name == "Unknown":
            self._prune(now)
            return None

        # Add sample, prune old
        self.buf.append((now, name, float(sim)))
        self._prune(now)

        # Need enough samples in window
        if len(self.buf) < self.min_samples:
            return IdentityDecision(False, name, float(sim), votes=0, total=len(self.buf))

        # Count votes + sims per identity
        counts = {}
        sim_sums = {}
        sim_counts = {}

        for _, nm, s in self.buf:
            counts[nm] = counts.get(nm, 0) + 1
            sim_sums[nm] = sim_sums.get(nm, 0.0) + s
            sim_counts[nm] = sim_counts.get(nm, 0) + 1

        # pick top voted identity
        top_name = max(counts.items(), key=lambda kv: kv[1])[0]
        top_votes = counts[top_name]
        top_avg_sim = sim_sums[top_name] / max(1, sim_counts[top_name])

        # Optional: require last K samples to all be the same identity (extra strict)
        if self.require_consecutive:
            last_k = list(self.buf)[-self.k_votes:]
            if any(nm != top_name for _, nm, _ in last_k):
                return IdentityDecision(False, top_name, top_avg_sim, votes=top_votes, total=len(self.buf))

        # Decision condition
        if top_votes >= self.k_votes and top_avg_sim >= self.min_avg_sim:
            self.last_decision_t = now
            decision = IdentityDecision(True, top_name, top_avg_sim, votes=top_votes, total=len(self.buf))
            self.reset()
            return decision

        return IdentityDecision(False, top_name, top_avg_sim, votes=top_votes, total=len(self.buf))