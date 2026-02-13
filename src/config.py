from dataclasses import dataclass
import os


@dataclass
class Config:

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
    # YOLO
    # --------------------------------------------------
    yolo_model_path: str = "models/yolo26n.pt"
    yolo_conf: float = 0.5
    person_class_id: int = 0
    yolo_every_n_frames: int = 3

    # --------------------------------------------------
    # MTCNN
    # --------------------------------------------------
    mtcnn_image_size: int = 160
    mtcnn_margin: int = 20
    mtcnn_min_face_size: int = 40

    # --------------------------------------------------
    # --- Face quality gates (1080p tuned) ---
    # --------------------------------------------------
    min_face_prob: float = 0.97
    min_face_w: int = 120
    min_face_h: int = 120

    # --------------------------------------------------
    # --- Threshold presets (dynamic by identity count) ---
    # --------------------------------------------------

    Difference = 6-7

    # Single identity (strict open-set)
    sim_threshold_1: float = 0.86
    sim_margin_1: float = 0.0   # unused for 1 identity

    # Few identities (3–9)
    sim_threshold_few: float = 0.77
    sim_margin_few: float = 0.08

    # Many identities (10+)
    sim_threshold_many: float = 0.80
    sim_margin_many: float = 0.10

    # --------------------------------------------------
    # --- Identity vote system (K-of-N) ---
    # --------------------------------------------------
    attendance_avg_window_sec: float = 2.0
    attendance_update_cooldown_sec: float = 10.0

    attendance_min_avg_sim: float = 0.80  # must match "many" threshold

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
    alert_cooldowns = {
        "UNKNOWN_PERSON": 30.0,
        "TAILGATING": 20.0,
        "OCCLUSION": 20.0,
    }

    # --------------------------------------------------
    # Ensure directories exist
    # --------------------------------------------------
    def ensure_dirs(self):
        os.makedirs(self.known_dir, exist_ok=True)
        os.makedirs(self.guests_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_db_path), exist_ok=True)
        os.makedirs(self.log_images_dir, exist_ok=True)