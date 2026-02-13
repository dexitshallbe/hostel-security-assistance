from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Config:
    # ---- Paths ----
    yolo_model_path: str = "models/yolo26n.pt"

    known_dir: str = "data/known_faces"
    guests_root: str = "data/guests"
    guests_reload_flag: str = "data/guests/_reload.flag"

    log_db_path: str = "data/logs/logs.db"
    log_images_dir: str = "data/logs/images"

    # ---- Camera sources ----
    entry_url: str = "http://PHONE1_IP:8080/video"
    exit_url: str = "http://PHONE2_IP:8080/video"

    # ---- YOLO gate ----
    yolo_every_n_frames: int = 3
    yolo_conf: float = 0.35
    person_class_id: int = 0  # COCO: person

    # ---- Face detection (MTCNN) ----
    mtcnn_image_size: int = 160
    mtcnn_margin: int = 20
    mtcnn_min_face_size: int = 40

    # ---- Recognition (security-ish defaults) ----
    sim_threshold: float = 0.68          # ↑ from 0.55
    sim_margin: float = 0.06             # best - second-best must exceed this

    min_face_prob: float = 0.92          # face quality gate (0.90–0.97)
    min_face_w: int = 60                 # minimum face bbox size
    min_face_h: int = 60

    # ---- Safe-zone / border filtering ----
    safe_border_pct: float = 0.10        # ignore 10% border
    min_inside_ratio: float = 0.75       # bbox must be ≥75% inside safe zone

    # ---- Unknown logging/alert smoothing ----
    unknown_face_prob_threshold: float = 0.95
    unknown_sim_upper: float = 0.50

    unknown_avg_window_sec: float = 2.0
    unknown_avg_cooldown_sec: float = 45.0

    # ---- Attendance smoothing ----
    attendance_avg_window_sec: float = 2.0
    attendance_min_avg_sim: float = 0.68     # match your sim_threshold
    attendance_update_cooldown_sec: float = 10.0

    # ---- Mismatch logging cadence + persistence ----
    mismatch_log_every_sec: float = 15.0
    mismatch_min_duration_sec: float = 1.8   # mismatch must persist before logging/alert

    # ---- Tailgating & occlusion alerts ----
    tailgate_window_sec: float = 2.5
    tailgate_min_persons: int = 2

    occlusion_window_sec: float = 2.5
    occlusion_ignore_persons_gt: int = 50

    # ---- Alert cooldowns (per type) ----
    alert_cooldowns: Dict[str, float] = field(default_factory=lambda: {
        "TAILGATING": 20.0,
        "UNKNOWN_PERSON": 45.0,
        "FACE_OCCLUSION": 30.0,
    })