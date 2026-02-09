from ..config import Config
from .camera_worker import CameraWorker


def run_two_cameras(cfg: Config):
    entry = CameraWorker("entry", cfg.entry_url, cfg)
    exit_ = CameraWorker("exit", cfg.exit_url, cfg)

    entry.start()
    exit_.start()

    entry.join()
    exit_.join()
