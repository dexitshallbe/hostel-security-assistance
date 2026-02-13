import time


class MismatchTracker:
    def __init__(self, min_duration_sec: float = 1.8, every_sec: float = 15.0):
        self.min_duration_sec = float(min_duration_sec)
        self.every_sec = float(every_sec)

        self._last_log = 0.0
        self._window_start = None

    def should_trigger(self, person_count: int, face_count: int) -> bool:
        now = time.time()
        mismatch = (person_count != face_count)

        if mismatch:
            if self._window_start is None:
                self._window_start = now
        else:
            self._window_start = None
            return False

        # mismatch ongoing, but not long enough
        if (now - self._window_start) < self.min_duration_sec:
            return False

        # rate-limit logs/triggers
        if (now - self._last_log) >= self.every_sec:
            self._last_log = now
            return True

        return False