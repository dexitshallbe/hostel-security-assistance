from dataclasses import dataclass, field
import os


@dataclass
class Config:
    entry_url: str = ""
    exit_url: str = ""

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    base_dir: str = "data"
    known_dir: str = "data/known"
    guests_dir: str = "data/guests"
    log_db_path: str = "data/logs/log.db"
    log_images_dir: str = "data/logs/images"
    guests_reload_flag: str = "data/guests_reload.flag"

    # --------------------------------------------------
    # Stream CPU controls
    # --------------------------------------------------
    target_width: int = 800
    target_process_fps_per_cam: float = 7.0
    person_detect_fps_per_cam: float = 4.0

    # --------------------------------------------------
    # YOLO
    # --------------------------------------------------
    yolo_model_path: str = "models/yolo26n.pt"
    yolo_conf: float = 0.5
    person_class_id: int = 0
    yolo_every_n_frames: int = 3  # legacy, no longer the preferred limiter

    # --------------------------------------------------
    # MTCNN
    # --------------------------------------------------
    mtcnn_image_size: int = 160
    mtcnn_margin: int = 20
    mtcnn_min_face_size: int = 40

    # --------------------------------------------------
    # Face quality gates
    # --------------------------------------------------
    min_face_prob: float = 0.80
    min_face_w: int = 120  # legacy alias for min_face_size
    min_face_h: int = 120  # legacy alias for min_face_size
    min_face_size: int = 120
    blur_threshold: float = 60.0

    # --------------------------------------------------
    # Recognition thresholds
    # --------------------------------------------------
    # New knobs
    accept_threshold: float = 0.80
    gap_threshold: float = 0.08
    unknown_reject_threshold: float = 0.72

    # Legacy knobs retained for compatibility
    sim_threshold_1: float = 0.82
    sim_margin_1: float = 0.0
    sim_threshold_few: float = 0.77
    sim_margin_few: float = 0.08
    sim_threshold_many: float = 0.80
    sim_margin_many: float = 0.10

    # --------------------------------------------------
    # Temporal confirmation (N-of-M)
    # --------------------------------------------------
    temporal_confirm_n: int = 3
    temporal_confirm_m: int = 5

    # --------------------------------------------------
    # Identity vote system (legacy attendance smoother)
    # --------------------------------------------------
    attendance_avg_window_sec: float = 2.0
    attendance_update_cooldown_sec: float = 10.0
    attendance_min_avg_sim: float = 0.80
    attendance_min_samples: int = 10
    attendance_k_votes: int = 7
    attendance_require_consecutive: bool = False

    # --------------------------------------------------
    # Unknown averaging
    # --------------------------------------------------
    unknown_avg_window_sec: float = 2.0
    unknown_avg_cooldown_sec: float = 20.0
    unknown_face_prob_threshold: float = 0.98
    unknown_sim_upper: float = 0.70

    # --------------------------------------------------
    # Mismatch / tailgating logic
    # --------------------------------------------------
    mismatch_min_duration_sec: float = 1.8
    mismatch_log_every_sec: float = 15.0
    safe_border_pct: float = 0.10
    min_inside_ratio: float = 0.75

    # --------------------------------------------------
    # Alert cooldowns (per type)
    # --------------------------------------------------
    alert_cooldowns: dict = field(default_factory=lambda: {
        "UNKNOWN_PERSON": 30.0,
        "TAILGATING": 20.0,
        "OCCLUSION": 20.0,
    })

    # --------------------------------------------------
    # Ensure directories exist
    # --------------------------------------------------
    def ensure_dirs(self):
        os.makedirs(self.known_dir, exist_ok=True)
        os.makedirs(self.guests_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_db_path), exist_ok=True)
        os.makedirs(self.log_images_dir, exist_ok=True)
