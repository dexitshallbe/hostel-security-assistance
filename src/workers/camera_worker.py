import os
import time
import threading
from datetime import datetime

import cv2

from ..config import Config
from ..db.queries import add_log, add_alert, add_event, set_inside_state
from ..yolo_gate.gate import YoloPersonGate
from ..face_recog.recognizer import FaceRecognizer, MatchResult
from ..trackers.mismatch import MismatchTracker
from ..trackers.unknown_avg import UnknownAverager
from ..trackers.identity_avg import IdentityAverager


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class CameraWorker(threading.Thread):
    """
    Headless camera worker:
    - reads RTSP
    - YOLO person gate
    - face recognition (known + unexpired guests)
    - writes alerts/logs/evidence to SQLite + disk
    - updates attendance inside/outside per day
    """
    def __init__(self, camera_name: str, url: str, cfg: Config):
        super().__init__(daemon=True)
        self.camera_name = camera_name  # "entry" or "exit"
        self.url = url
        self.cfg = cfg

        self.stop_flag = False
        self.frame_idx = 0
        self.last_person_count = 0

        self.gate = YoloPersonGate(cfg.yolo_model_path, cfg.yolo_conf, cfg.person_class_id)
        self.recognizer = FaceRecognizer(
            device="cuda" if self._has_cuda() else "cpu",
            image_size=cfg.mtcnn_image_size,
            margin=cfg.mtcnn_margin,
            min_face_size=cfg.mtcnn_min_face_size,
        )
        self._db_build()

        # trackers
        self.mismatch = MismatchTracker(every_sec=cfg.mismatch_log_every_sec)
        self.unknown_avg = UnknownAverager(cfg.unknown_avg_window_sec, cfg.unknown_avg_cooldown_sec)
        self.identity_avg = IdentityAverager(cfg.attendance_avg_window_sec, cfg.attendance_min_avg_sim, cfg.attendance_update_cooldown_sec)

        # tailgate/occlusion timers
        self.tailgate_start = None
        self.occlusion_start = None

        # reload flag mtime
        self._reload_mtime = self._get_reload_mtime()

        # cooldown per alert type
        self._last_alert = {}

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
        self.recognizer.build_database(self.cfg.known_dir, self.cfg.log_db_path)
        add_log(self.cfg.log_db_path, "INFO", f"[{self.camera_name}] Face DB ready: {len(self.recognizer.db_embeddings)} identities")

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
        cap = cv2.VideoCapture(self.url)
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
                    add_log(self.cfg.log_db_path, "WARN", f"[{self.camera_name}] RTSP read failed; reconnecting")
                    cap.release()
                    time.sleep(1.0)
                    cap = self._open()
                    last_ok = time.time()
                continue

            last_ok = time.time()
            self.frame_idx += 1

            # periodic guest DB reload
            if self.frame_idx % 30 == 0:
                self._maybe_reload_db()

            # YOLO gate every N frames
            if self.frame_idx % self.cfg.yolo_every_n_frames == 0:
                self.last_person_count = self.gate.person_count(frame)
            persons = self.last_person_count

            dets = []
            if persons > 0:
                dets = self.recognizer.recognize(frame, sim_threshold=self.cfg.sim_threshold)
            faces = len(dets)

            # mismatch logging (optional)
            if self.mismatch.should_log(persons, faces):
                ev = self._save_evidence("mismatch", frame, f"{self.camera_name}_mismatch")
                add_event(self.cfg.log_db_path, self.camera_name, "mismatch", evidence_path=ev, message=f"persons={persons} faces={faces}")
                add_log(self.cfg.log_db_path, "WARN", f"[{self.camera_name}] MISMATCH logged persons={persons} faces={faces}")

            # tailgating: persons>faces for a window
            if persons >= self.cfg.tailgate_min_persons and faces < persons:
                if self.tailgate_start is None:
                    self.tailgate_start = time.time()
                elif (time.time() - self.tailgate_start) >= self.cfg.tailgate_window_sec:
                    if self._cooldown_ok("TAILGATING"):
                        ev = self._save_evidence("alerts", frame, f"{self.camera_name}_tailgating")
                        add_alert(self.cfg.log_db_path, self.camera_name, "TAILGATING",
                                  f"persons={persons} faces={faces}", evidence_path=ev)
                        add_event(self.cfg.log_db_path, self.camera_name, "tailgating", evidence_path=ev,
                                  message=f"persons={persons} faces={faces}")
                        add_log(self.cfg.log_db_path, "ALERT", f"[{self.camera_name}] Tailgating alert fired")
                    self.tailgate_start = None
            else:
                self.tailgate_start = None

            # occlusion: persons>0 faces==0 for window, ignore if crowd huge
            if persons > 0 and faces == 0 and persons <= self.cfg.occlusion_ignore_persons_gt:
                if self.occlusion_start is None:
                    self.occlusion_start = time.time()
                elif (time.time() - self.occlusion_start) >= self.cfg.occlusion_window_sec:
                    if self._cooldown_ok("FACE_OCCLUSION"):
                        ev = self._save_evidence("alerts", frame, f"{self.camera_name}_occlusion")
                        add_alert(self.cfg.log_db_path, self.camera_name, "FACE_OCCLUSION",
                                  f"persons={persons} faces=0", evidence_path=ev)
                        add_event(self.cfg.log_db_path, self.camera_name, "occlusion", evidence_path=ev,
                                  message=f"persons={persons} faces=0")
                        add_log(self.cfg.log_db_path, "ALERT", f"[{self.camera_name}] Face occlusion alert fired")
                    self.occlusion_start = None
            else:
                self.occlusion_start = None

            # attendance + unknown (based on dets)
            if dets:
                # choose best known identity (max sim among non-Unknown)
                best_known = None
                unknown_candidates = []

                for r in dets:
                    if r.name == "Unknown":
                        unknown_candidates.append((r.sim, r.face_prob))
                    else:
                        if best_known is None or r.sim > best_known.sim:
                            best_known = r

                # attendance smoothing on best_known
                if best_known is not None:
                    decision = self.identity_avg.update(best_known.name, best_known.sim)
                    if decision and decision.ready:
                        # entry -> inside=1, exit -> inside=0
                        inside = 1 if self.camera_name == "entry" else 0
                        set_inside_state(self.cfg.log_db_path, decision.name, inside, self.camera_name)

                        ev = self._save_evidence(self.camera_name, frame, f"{self.camera_name}_{decision.name}")
                        add_event(self.cfg.log_db_path, self.camera_name,
                                  "known_entry" if self.camera_name == "entry" else "known_exit",
                                  name=decision.name, sim=decision.avg_sim, evidence_path=ev,
                                  message=f"avgSim={decision.avg_sim:.3f}")
                        add_log(self.cfg.log_db_path, "INFO",
                                f"[{self.camera_name}] Attendance updated: {decision.name} inside={inside} avgSim={decision.avg_sim:.3f}")

                # unknown averaging (pick unknown with best face_prob)
                if unknown_candidates:
                    best_sim, best_prob = sorted(unknown_candidates, key=lambda x: x[1], reverse=True)[0]
                    dec = self.unknown_avg.update(best_prob, best_sim)
                    if dec and dec.should_trigger:
                        if dec.avg_face_prob >= self.cfg.unknown_face_prob_threshold and dec.avg_best_sim <= self.cfg.unknown_sim_upper:
                            if self._cooldown_ok("UNKNOWN_PERSON"):
                                ev = self._save_evidence("unknown", frame, f"{self.camera_name}_unknown")
                                add_alert(self.cfg.log_db_path, self.camera_name, "UNKNOWN_PERSON",
                                          f"avgFaceP={dec.avg_face_prob:.2f} avgSim={dec.avg_best_sim:.3f}",
                                          evidence_path=ev, sim=dec.avg_best_sim, face_prob=dec.avg_face_prob)
                                add_event(self.cfg.log_db_path, self.camera_name, "unknown",
                                          sim=dec.avg_best_sim, face_prob=dec.avg_face_prob, evidence_path=ev,
                                          message=f"avgFaceP={dec.avg_face_prob:.2f} avgSim={dec.avg_best_sim:.3f}")
                                add_log(self.cfg.log_db_path, "ALERT",
                                        f"[{self.camera_name}] Unknown person alert fired")
                            self.unknown_avg.mark_triggered()
                else:
                    self.unknown_avg.reset()
            else:
                self.unknown_avg.reset()

        cap.release()
        add_log(self.cfg.log_db_path, "INFO", f"[{self.camera_name}] Stopped")
