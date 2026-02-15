from src.config import Config
from src.db.schema import init_db
from src.workers.camera_worker import CameraWorker


def main():
    cfg = Config(
        # put your real IP Webcam URLs here
        entry_url="http://10.191.222.104:8080/video",
        exit_url="http://100.91.156.215:5000/video",
    )

    init_db(cfg.log_db_path)
    print(f"[OK] DB ready: {cfg.log_db_path}")

    entry = CameraWorker("entry", cfg.entry_url, cfg)
    exit_ = CameraWorker("exit", cfg.exit_url, cfg)

    entry.start()
    exit_.start()

    # Wait until both cameras confirm first frame
    print("[..] Connecting cameras...")
    while True:
        if entry.is_connected() and exit_.is_connected():
            print("[OK] Entry camera connected")
            print("[OK] Exit camera connected")
            print("[LIVE] System is LIVE ✅ (workers running)")
            break
        # show individual progress
        if entry.is_connected() and not getattr(entry, "_printed_connected", False):
            print("[OK] Entry camera connected")
            entry._printed_connected = True
        if exit_.is_connected() and not getattr(exit_, "_printed_connected", False):
            print("[OK] Exit camera connected")
            exit_._printed_connected = True

        # If a worker reports a fatal open error, show it
        e1 = entry.last_open_error()
        e2 = exit_.last_open_error()
        if e1:
            print(f"[WARN] Entry camera: {e1}")
        if e2:
            print(f"[WARN] Exit camera: {e2}")

        import time
        time.sleep(0.5)

    try:
        while True:
            entry.join(timeout=1.0)
            exit_.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping workers...")
        entry.stop()
        exit_.stop()
        entry.join(timeout=3.0)
        exit_.join(timeout=3.0)
        print("[OK] Workers stopped.")


if __name__ == "__main__":
    main()