from src.config import Config
from src.db.schema import init_db
from src.workers.camera_worker import CameraWorker

def main():
    cfg = Config(
        entry_url="http://10.120.254.104:8080/video",
        exit_url="http://172.18.116.86:5000/video",
    )
    init_db(cfg.log_db_path)

    entry = CameraWorker("entry", cfg.entry_url, cfg)
    exit_ = CameraWorker("exit", cfg.exit_url, cfg)

    entry.start()
    exit_.start()

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
        print("[INFO] Workers stopped.")

if __name__ == "__main__":
    main()
