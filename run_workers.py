import os
import shutil
import sqlite3
import time
from datetime import datetime

from src.config import Config
from src.db.schema import init_db
from src.workers.camera_worker import CameraWorker


def touch_reload_flag(reload_flag_path: str):
    os.makedirs(os.path.dirname(reload_flag_path), exist_ok=True)
    with open(reload_flag_path, "a") as f:
        f.write("")
    os.utime(reload_flag_path, None)


def cleanup_expired_guests(db_path: str, reload_flag_path: str) -> int:
    now = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT guest_id, name, folder_path, expires_ts
            FROM guest_access
            WHERE expires_ts <= ?
        """, (now,)).fetchall()
        expired = [dict(r) for r in rows]

    if not expired:
        return 0

    for g in expired:
        fp = g.get("folder_path")
        if fp and os.path.exists(fp):
            shutil.rmtree(fp, ignore_errors=True)

    with sqlite3.connect(db_path) as con:
        con.execute("DELETE FROM guest_access WHERE expires_ts <= ?", (now,))

    touch_reload_flag(reload_flag_path)
    return len(expired)


def main():
    cfg = Config(
        entry_url="http://10.216.138.103:8080/video",
        exit_url="http://10.216.138.217:5000/video",
    )

    init_db(cfg.log_db_path)
    print(f"[OK] DB ready: {cfg.log_db_path}")

    entry = CameraWorker("entry", cfg.entry_url, cfg)
    exit_ = CameraWorker("exit", cfg.exit_url, cfg)

    entry.start()
    exit_.start()

    print("[..] Connecting cameras...")
    while True:
        if entry.is_connected() and exit_.is_connected():
            print("[OK] Entry camera connected")
            print("[OK] Exit camera connected")
            print("[LIVE] System is LIVE ✅ (workers running)")
            break

        if entry.is_connected() and not getattr(entry, "_printed_connected", False):
            print("[OK] Entry camera connected")
            entry._printed_connected = True
        if exit_.is_connected() and not getattr(exit_, "_printed_connected", False):
            print("[OK] Exit camera connected")
            exit_._printed_connected = True

        e1 = entry.last_open_error()
        e2 = exit_.last_open_error()
        if e1:
            print(f"[WARN] Entry camera: {e1}")
        if e2:
            print(f"[WARN] Exit camera: {e2}")

        time.sleep(0.5)

    CLEANUP_EVERY_SEC = 300  # 5 minutes
    last_cleanup = 0.0

    try:
        while True:
            entry.join(timeout=1.0)
            exit_.join(timeout=1.0)

            now = time.time()
            if now - last_cleanup >= CLEANUP_EVERY_SEC:
                removed = cleanup_expired_guests(cfg.log_db_path, cfg.guests_reload_flag)
                if removed:
                    print(f"[CLEANUP] Removed {removed} expired guest(s)")
                last_cleanup = now

    except KeyboardInterrupt:
        print("\n[INFO] Stopping workers...")
        entry.stop()
        exit_.stop()
        entry.join(timeout=3.0)
        exit_.join(timeout=3.0)
        print("[OK] Workers stopped.")


if __name__ == "__main__":
    main()
