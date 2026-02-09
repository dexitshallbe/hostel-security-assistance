from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Config:
    yolo_model_path: str = "models/yolo26n.pt"

    known_dir: str = "data/known_faces"
    guests_root: str = "data/guests"
    guests_reload_flag: str = "data/guests/_reload.flag"

    log_db_path: str = "data/logs/logs.db"
    log_images_dir: str = "data/logs/images"

    # RTSP streams (EDIT THESE)
    entry_url: str = "rtsp://USER:PASS@IP:554/entry"
    exit_url: str = "rtsp://USER:PASS@IP:554/exit"

    yolo_every_n_frames: int = 3
    yolo_conf: float = 0.35
    person_class_id: int = 0

    mtcnn_image_size: int = 160
    mtcnn_margin: int = 20
    mtcnn_min_face_size: int = 40

    sim_threshold: float = 0.55
    unknown_face_prob_threshold: float = 0.95
    unknown_sim_upper: float = 0.50

    unknown_avg_window_sec: float = 2.0
    unknown_avg_cooldown_sec: float = 45.0

    attendance_avg_window_sec: float = 2.0
    attendance_min_avg_sim: float = 0.55
    attendance_update_cooldown_sec: float = 10.0

    mismatch_log_every_sec: float = 15.0

    tailgate_window_sec: float = 2.5
    tailgate_min_persons: int = 2

    occlusion_window_sec: float = 2.5
    occlusion_ignore_persons_gt: int = 50

    alert_cooldowns: Dict[str, float] = field(default_factory=lambda: {
        "TAILGATING": 20.0,
        "UNKNOWN_PERSON": 45.0,
        "FACE_OCCLUSION": 30.0,
    })
