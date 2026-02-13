# src/workers/camera_worker.py

import os
import time
import threading
from datetime import datetime

import cv2

from ..config import Config
from ..db.queries import add_log, add_alert, add_event, set_inside_state
from ..yolo_gate.gate import YoloPersonGate
from ..face_recog.recognizer import FaceRecognizer
from ..trackers.mismatch import MismatchTracker
from ..trackers.unknown_avg import UnknownAverager
from ..trackers.identity_avg import IdentityAverager


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------- Safe-zone helpers (A) ----------
def _area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return (x1, y1, x2, y2)


def inside_ratio(bbox, safe):
    inter = _intersection(bbox, safe)
    return _area(inter) / (_area(bbox) + 1e-9)


def safe_box(frame_shape, border_pct: float):
    H, W = frame_shape[:2]
    mx = int(border_pct * W)
    my = int(border_pct * H)
    return (mx, my, W - mx, H - my)


class CameraWorker(threading.Thread):
    """
    Headless camera worker:
    - reads network stream (HTTP MJPEG / RTSP)
    - YOLO person detection (bboxes)
    - face recognition (known + unexpired guests)
    - writes alerts/logs/evidence to SQLite + disk
    - updates attendance inside/outside per day
    """

    def is_connected(self) -> bool:
        return bool(self._connected)

    def last_open_error(self) -> str:
        return self._open_error

    def db_summary(self) -> str:
        return self._db_summary

    def __init__(self, camera_name: str, url: str, cfg: Config):
        super().__init__(daemon=True)
        self.camera_name = camera_name  # "entry" or "exit"
        self.url = url
        self.cfg = cfg

        self.stop_flag = False
        self.frame_idx = 0

        self._connected = False
        self._open_error = ""
        self._db_summary = ""

        self.gate = YoloPersonGate(cfg.yolo_model_path, cfg.yolo_conf, cfg.person_class_id)
        self.recognizer = FaceRecognizer(
            device="cuda" if self._has_cuda() else "cpu",
            image_size=cfg.mtcnn_image_size,
            margin=cfg.mtcnn_margin,
            min_face_size=cfg.mtcnn_min_face_size,
        )
        self._db_build()

        # ---------- Trackers ----------
        # (B) mismatch must persist before triggering
        self.mismatch = MismatchTracker(
            min_duration_sec=cfg.mismatch_min_duration_sec,
            every_sec=cfg.mismatch_log_every_sec,
        )
        self.unknown_avg = UnknownAverager(cfg.unknown_avg_window_sec, cfg.unknown_avg_cooldown_sec)
        self.identity_avg = IdentityAverager(
            cfg.attendance_avg_window_sec,
            cfg.attendance_min_avg_sim,
            cfg.attendance_update_cooldown_sec,
        )

        # tailgate/occlusion timers
        self.tailgate_start = None
        self.occlusion_start = None

        # reload flag mtime
        self._reload_mtime = self._get_reload_mtime()

        # cooldown per alert type
        self._last_alert = {}

        # cache last YOLO detections (boxes)
        self.last_person_boxes = []

        # ensure image dirs
        for sub in ["alerts", "entry", "exit", "unknown", "mismatch"]:
            os.makedirs(os.path.join(cfg.log_images_dir, sub), exist_ok=True)

    def _has_cuda(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def _get_reload_mtime(self):
        try:
            return os.path.getmtime(self.cfg.guests_reload_flag)
        except Exception:
            return 0.0

    def _db_build(self):
        add_log(self.cfg.log_db_path, "INFO", f"[{self.camera_name}] Building face DB (known + guests)")

        t0 = time.time()
        self.recognizer.build_database(self.cfg.known_dir, self.cfg.log_db_path)
        dt = time.time() - t0

        # count total reference embeddings/images loaded
        ident_count = len(self.recognizer.db_embeddings)
        img_count = sum(v.shape[0] for v in self.recognizer.db_embeddings.values())

        self._db_summary = f"{ident_count} identities, {img_count} images, build={dt:.2f}s"
        print(f"[OK] Face DB ({self.camera_name}): {self._db_summary}")

        add_log(self.cfg.log_db_path, "INFO", f"[{self.camera_name}] Face DB ready: {self._db_summary}")

    def _maybe_reload_db(self):
        mt = self._get_reload_mtime()
        if mt > self._reload_mtime:
            self._reload_mtime = mt
            self._db_build()

    def _cooldown_ok(self, alert_type: str) -> bool:
        cd = float(self.cfg.alert_cooldowns.get(alert_type, 20.0))
        now = time.time()
        last = self._last_alert.get(alert_type, 0.0)
        if (now - last) >= cd:
            self._last_alert[alert_type] = now
            return True
        return False

    def stop(self):
        self.stop_flag = True

    def _save_evidence(self, subdir: str, frame_bgr, prefix: str) -> str:
        path = os.path.join(self.cfg.log_images_dir, subdir, f"{prefix}_{now_stamp()}.jpg")
        cv2.imwrite(path, frame_bgr)
        return path

    def _open(self):
        # Force FFmpeg path; helps with phone MJPEG/RTSP sources
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        cap = self._open()
        add_log(self.cfg.log_db_path, "INFO", f"[{self.camera_name}] Started: {self.url}")

        last_ok = time.time()

        while not self.stop_flag:
            ok, frame = cap.read()
            if not ok or frame is None:
                # reconnect
                if (time.time() - last_ok) > 3.0:
                    add_log(self.cfg.log_db_path, "WARN", f"[{self.camera_name}] Stream read failed; reconnecting")
                    cap.release()
                    time.sleep(1.0)
                    cap = self._open()
                    last_ok = time.time()
                continue

            last_ok = time.time()
            self.frame_idx += 1

            if not self._connected:
                self._connected = True
                print(f"[OK] {self.camera_name.upper()} camera stream: first frame received")

            # periodic guest DB reload
            if self.frame_idx % 30 == 0:
                self._maybe_reload_db()

            # ---------- (C) YOLO boxes + SAFE + valid persons ----------
            if self.frame_idx % self.cfg.yolo_every_n_frames == 0:
                self.last_person_boxes = self.gate.detect_person_boxes(frame)

            person_boxes = self.last_person_boxes
            persons = len(person_boxes)

            SAFE = safe_box(frame.shape, self.cfg.safe_border_pct)
            valid_person_boxes = [
                b for b in person_boxes if inside_ratio(b, SAFE) >= self.cfg.min_inside_ratio
            ]
            valid_persons = len(valid_person_boxes)

            # ---------- (D) Face recognition with new args + valid faces ----------
            dets = []
            if persons > 0:
                dets = self.recognizer.recognize(
                    frame,
                    sim_threshold=self.cfg.sim_threshold,
                    sim_margin=self.cfg.sim_margin,
                    min_face_prob=self.cfg.min_face_prob,
                    min_face_w=self.cfg.min_face_w,
                    min_face_h=self.cfg.min_face_h,
                )

            faces = len(dets)
            valid_faces = [m for m in dets if inside_ratio(m.bbox, SAFE) >= self.cfg.min_inside_ratio]
            valid_face_count = len(valid_faces)

            # ---------- (E) Mismatch based on valid counts (persistent) ----------
            if self.mismatch.should_trigger(valid_persons, valid_face_count):
                ev = self._save_evidence("mismatch", frame, f"{self.camera_name}_mismatch")
                add_event(
                    self.cfg.log_db_path,
                    self.camera_name,
                    "mismatch",
                    evidence_path=ev,
                    message=f"valid_persons={valid_persons} valid_faces={valid_face_count} (raw persons={persons} faces={faces})",
                )
                add_log(
                    self.cfg.log_db_path,
                    "WARN",
                    f"[{self.camera_name}] MISMATCH logged valid_persons={valid_persons} valid_faces={valid_face_count} (raw persons={persons} faces={faces})",
                )

            # ---- (rest of your logic can remain as-is) ----
            # If you want tailgating/occlusion updated too, use:
            #   valid_persons and valid_face_count instead of persons/faces
            #
            # And for identity/unknown averaging, use valid_faces instead of dets.

            # Minimal continuation for stability: use valid_faces for downstream logic
            if valid_faces:
                best_known = None
                unknown_candidates = []

                for r in valid_faces:
                    if r.name == "Unknown":
                        unknown_candidates.append((r.sim, r.face_prob))
                    else:
                        if best_known is None or r.sim > best_known.sim:
                            best_known = r

                # Attendance smoothing on best_known
                if best_known is not None:
                    decision = self.identity_avg.update(best_known.name, best_known.sim)
                    if decision and decision.ready:
                        inside = 1 if self.camera_name == "entry" else 0
                        set_inside_state(self.cfg.log_db_path, decision.name, inside, self.camera_name)

                        ev = self._save_evidence(self.camera_name, frame, f"{self.camera_name}_{decision.name}")
                        add_event(
                            self.cfg.log_db_path,
                            self.camera_name,
                            "known_entry" if self.camera_name == "entry" else "known_exit",
                            name=decision.name,
                            sim=decision.avg_sim,
                            evidence_path=ev,
                            message=f"avgSim={decision.avg_sim:.3f}",
                        )
                        add_log(
                            self.cfg.log_db_path,
                            "INFO",
                            f"[{self.camera_name}] Attendance updated: {decision.name} inside={inside} avgSim={decision.avg_sim:.3f}",
                        )

                # Unknown averaging (pick unknown with best face_prob)
                if unknown_candidates:
                    best_sim, best_prob = sorted(unknown_candidates, key=lambda x: x[1], reverse=True)[0]
                    dec = self.unknown_avg.update(best_prob, best_sim)
                    if dec and dec.should_trigger:
                        if (
                            dec.avg_face_prob >= self.cfg.unknown_face_prob_threshold
                            and dec.avg_best_sim <= self.cfg.unknown_sim_upper
                        ):
                            if self._cooldown_ok("UNKNOWN_PERSON"):
                                ev = self._save_evidence("unknown", frame, f"{self.camera_name}_unknown")
                                add_alert(
                                    self.cfg.log_db_path,
                                    self.camera_name,
                                    "UNKNOWN_PERSON",
                                    f"avgFaceP={dec.avg_face_prob:.2f} avgSim={dec.avg_best_sim:.3f}",
                                    evidence_path=ev,
                                    sim=dec.avg_best_sim,
                                    face_prob=dec.avg_face_prob,
                                )
                                add_event(
                                    self.cfg.log_db_path,
                                    self.camera_name,
                                    "unknown",
                                    sim=dec.avg_best_sim,
                                    face_prob=dec.avg_face_prob,
                                    evidence_path=ev,
                                    message=f"avgFaceP={dec.avg_face_prob:.2f} avgSim={dec.avg_best_sim:.3f}",
                                )
                                add_log(self.cfg.log_db_path, "ALERT", f"[{self.camera_name}] Unknown person alert fired")
                            self.unknown_avg.mark_triggered()
                else:
                    self.unknown_avg.reset()
            else:
                self.unknown_avg.reset()

        cap.release()
        add_log(self.cfg.log_db_path, "INFO", f"[{self.camera_name}] Stopped")