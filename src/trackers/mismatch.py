import time


class MismatchTracker:
    def __init__(self, every_sec: float = 15.0):
        self.every_sec = every_sec
        self._last = 0.0
        self._window_start = None

    def should_log(self, person_count: int, face_count: int) -> bool:
        now = time.time()
        mismatch = (person_count != face_count)
        if mismatch and self._window_start is None:
            self._window_start = now
        if not mismatch:
            self._window_start = None
            return False

        if (now - self._last) >= self.every_sec:
            self._last = now
            return True
        return False

    def window_start_iso(self):
        import datetime
        if self._window_start is None:
            return None
        return datetime.datetime.fromtimestamp(self._window_start).isoformat(timespec="seconds")

    def window_end_iso(self):
        import datetime
        return datetime.datetime.now().isoformat(timespec="seconds")
