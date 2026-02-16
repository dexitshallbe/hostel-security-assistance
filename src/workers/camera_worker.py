import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional, Tuple

import cv2

from ..config import Config
from ..db.queries import add_alert, add_event, add_log, set_inside_state
from ..face_recog.recognizer import FaceRecognizer, MatchResult
from ..trackers.identity_avg import IdentityAverager
from ..trackers.mismatch import MismatchTracker
from ..trackers.unknown_avg import UnknownAverager
from ..yolo_gate.gate import YoloPersonGate


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
    return x1, y1, x2, y2


def inside_ratio(bbox, safe):
    inter = _intersection(bbox, safe)
    return _area(inter) / (_area(bbox) + 1e-9)


def safe_box(frame_shape, border_pct: float):
    h, w = frame_shape[:2]
    mx = int(border_pct * w)
    my = int(border_pct * h)
    return mx, my, w - mx, h - my


class CameraWorker(threading.Thread):
    def is_connected(self) -> bool:
        return bool(self._connected)

    def last_open_error(self) -> str:
        return self._open_error

    def db_summary(self) -> str:
        return self._db_summary

    def __init__(self, camera_name: str, url: str, cfg: Config):
        super().__init__(daemon=True)
        self.camera_name = camera_name
        self.url = url
        self.cfg = cfg

        self.stop_flag = False
        self.frame_idx = 0
        self._connected = False
        self._open_error = ""
        self._db_summary = ""

        self.gate = YoloPersonGate(cfg.yolo_model_path, cfg.yolo_conf, cfg.person_class_id)
        self.recognizer = FaceRecognizer(
            device="cpu",  # keep worker CPU-friendly by default
            image_size=cfg.mtcnn_image_size,
            margin=cfg.mtcnn_margin,
            min_face_size=cfg.mtcnn_min_face_size,
        )
        self._db_build()

        self.mismatch = MismatchTracker(min_duration_sec=cfg.mismatch_min_duration_sec, every_sec=cfg.mismatch_log_every_sec)
        self.unknown_avg = UnknownAverager(cfg.unknown_avg_window_sec, cfg.unknown_avg_cooldown_sec)
        self.identity_avg = IdentityAverager(
            window_sec=cfg.attendance_avg_window_sec,
            min_avg_sim=cfg.attendance_min_avg_sim,
            cooldown_sec=cfg.attendance_update_cooldown_sec,
            min_samples=getattr(cfg, "attendance_min_samples", 8),
            k_votes=getattr(cfg, "attendance_k_votes", 6),
            require_consecutive=getattr(cfg, "attendance_require_consecutive", False),
        )
        self.temporal_names = deque(maxlen=max(1, int(cfg.temporal_confirm_m)))

        self._reload_mtime = self._get_reload_mtime()
        self._last_alert = {}
        self.last_person_boxes = []

        self.capture_lock = threading.Lock()
        self.latest_frame = None
        self.latest_seq = 0
        self.latest_ts = 0.0
        self.latest_consumed_seq = 0
        self.capture_drop_count = 0
        self.capture_frame_count = 0
        self.capture_start = time.time()

        self.infer_ms_acc = 0.0
        self.infer_count = 0
        self.proc_frame_count = 0
        self.proc_start = time.time()
        self.last_perf_log = time.time()
        self.last_person_detect_ts = 0.0

        for sub in ["alerts", "entry", "exit", "unknown", "mismatch"]:
            os.makedirs(os.path.join(cfg.log_images_dir, sub), exist_ok=True)

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

        ident_count = len(self.recognizer.db_embeddings)
        img_count = sum(v.shape[0] for v in self.recognizer.db_embeddings.values())
        thr = self.recognizer.calibrated_thresholds

        self._db_summary = (
            f"{ident_count} identities, {img_count} images, build={dt:.2f}s, "
            f"accept={thr.get('accept_threshold', 0.0):.3f}, gap={thr.get('gap_threshold', 0.0):.3f}"
        )
        print(f"[OK] Face DB ({self.camera_name}): {self._db_summary}")
        add_log(self.cfg.log_db_path, "INFO", f"[{self.camera_name}] Face DB ready: {self._db_summary}")

    def _maybe_reload_db(self):
        mt = self._get_reload_mtime()
        if mt != self._reload_mtime:
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

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _capture_loop(self):
        cap = self._open()
        while not self.stop_flag:
            ok, frame = cap.read()
            if not ok or frame is None:
                self._connected = False
                cap.release()
                time.sleep(0.5)
                cap = self._open()
                continue

            if not self._connected:
                self._connected = True
                print(f"[OK] {self.camera_name.upper()} camera stream: first frame received")

            with self.capture_lock:
                if self.latest_seq > self.latest_consumed_seq:
                    self.capture_drop_count += 1
                self.latest_seq += 1
                self.latest_frame = frame
                self.latest_ts = time.time()
                self.capture_frame_count += 1

        cap.release()

    def _downscale(self, frame):
        target_w = int(getattr(self.cfg, "target_width", 800))
        h, w = frame.shape[:2]
        if w <= target_w:
            return frame
        scale = target_w / float(w)
        return cv2.resize(frame, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    def _annotate_frame(self, frame, bbox: Optional[Tuple[int, int, int, int]], text: str, color=(0, 255, 255)):
        out = frame.copy()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return out

    def _save_evidence(self, subdir: str, frame_bgr, prefix: str, alert_type: str, identity: str = "", sim: Optional[float] = None,
                       conf: Optional[float] = None, bbox: Optional[Tuple[int, int, int, int]] = None) -> str:
        ts = now_stamp()
        reason = alert_type.lower()
        raw_path = os.path.join(self.cfg.log_images_dir, subdir, f"{prefix}_{reason}_{ts}_raw.jpg")
        ann_path = os.path.join(self.cfg.log_images_dir, subdir, f"{prefix}_{reason}_{ts}_ann.jpg")

        cv2.imwrite(raw_path, frame_bgr)
        label = f"{alert_type} {identity}".strip()
        if sim is not None:
            label += f" sim={sim:.3f}"
        if conf is not None:
            label += f" conf={conf:.2f}"
        ann = self._annotate_frame(frame_bgr, bbox, label)
        cv2.imwrite(ann_path, ann)
        return ann_path

    def _temporal_confirmed(self, name: str) -> bool:
        self.temporal_names.append(name)
        n = int(getattr(self.cfg, "temporal_confirm_n", 3))
        return sum(1 for x in self.temporal_names if x == name) >= n

    def _log_perf(self):
        now = time.time()
        if (now - self.last_perf_log) < 5.0:
            return

        capture_fps = self.capture_frame_count / max(1e-6, now - self.capture_start)
        proc_fps = self.proc_frame_count / max(1e-6, now - self.proc_start)
        avg_infer_ms = self.infer_ms_acc / max(1, self.infer_count)
        msg = (
            f"[{self.camera_name}] perf capture_fps={capture_fps:.1f} proc_fps={proc_fps:.1f} "
            f"avg_infer_ms={avg_infer_ms:.1f} drops={self.capture_drop_count}"
        )
        print(msg)
        add_log(self.cfg.log_db_path, "INFO", msg)
        self.last_perf_log = now

    def run(self):
        add_log(self.cfg.log_db_path, "INFO", f"[{self.camera_name}] Started: {self.url}")
        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        capture_thread.start()

        target_period = 1.0 / max(1.0, float(getattr(self.cfg, "target_process_fps_per_cam", 7.0)))
        person_period = 1.0 / max(1.0, float(getattr(self.cfg, "person_detect_fps_per_cam", 4.0)))

        while not self.stop_flag:
            loop_t0 = time.time()
            with self.capture_lock:
                seq = self.latest_seq
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                self.latest_consumed_seq = seq

            if frame is None:
                time.sleep(0.02)
                continue

            self.frame_idx += 1
            self.proc_frame_count += 1
            self._maybe_reload_db()

            frame = self._downscale(frame)
            infer_t0 = time.time()
            if (time.time() - self.last_person_detect_ts) >= person_period:
                self.last_person_boxes = self.gate.detect_person_boxes(frame)
                self.last_person_detect_ts = time.time()

            person_boxes = self.last_person_boxes
            safe = safe_box(frame.shape, self.cfg.safe_border_pct)
            valid_person_boxes = [b for b in person_boxes if inside_ratio(b, safe) >= self.cfg.min_inside_ratio]

            dets = self.recognizer.recognize(frame, self.cfg) if person_boxes else []
            valid_faces = [m for m in dets if inside_ratio(m.bbox, safe) >= self.cfg.min_inside_ratio]

            infer_ms = (time.time() - infer_t0) * 1000.0
            self.infer_ms_acc += infer_ms
            self.infer_count += 1

            # Per-attempt scoring logs
            for m in valid_faces:
                add_log(
                    self.cfg.log_db_path,
                    "INFO",
                    (
                        f"[{self.camera_name}] rec name={m.name} sim={m.sim:.3f} second={m.second_sim:.3f} "
                        f"margin={m.margin:.3f} prob={m.face_prob:.2f} blur={m.blur_var:.1f} reason={m.decision_reason}"
                    ),
                )

            valid_persons = len(valid_person_boxes)
            valid_face_count = len(valid_faces)
            if self.mismatch.should_trigger(valid_persons, valid_face_count):
                bbox = valid_person_boxes[0] if valid_person_boxes else None
                ev = self._save_evidence("mismatch", frame, self.camera_name, "MISMATCH", bbox=bbox)
                add_event(
                    self.cfg.log_db_path,
                    self.camera_name,
                    "mismatch",
                    evidence_path=ev,
                    message=f"valid_persons={valid_persons} valid_faces={valid_face_count}",
                )
                add_log(self.cfg.log_db_path, "WARN", f"[{self.camera_name}] MISMATCH valid_persons={valid_persons} valid_faces={valid_face_count}")

            if valid_faces:
                best_known: Optional[MatchResult] = None
                unknown_candidates = []
                for r in valid_faces:
                    if r.name == "Unknown":
                        unknown_candidates.append(r)
                    elif best_known is None or r.sim > best_known.sim:
                        best_known = r

                if best_known is not None and self._temporal_confirmed(best_known.name):
                    decision = self.identity_avg.update(best_known.name, best_known.sim)
                    if decision and decision.ready:
                        inside = 1 if self.camera_name == "entry" else 0
                        set_inside_state(self.cfg.log_db_path, decision.name, inside, self.camera_name)
                        ev = self._save_evidence(
                            self.camera_name,
                            frame,
                            f"{self.camera_name}_{decision.name}",
                            "KNOWN",
                            identity=decision.name,
                            sim=decision.avg_sim,
                            conf=best_known.face_prob,
                            bbox=best_known.bbox,
                        )
                        add_event(
                            self.cfg.log_db_path,
                            self.camera_name,
                            "known_entry" if self.camera_name == "entry" else "known_exit",
                            name=decision.name,
                            sim=decision.avg_sim,
                            evidence_path=ev,
                            message=f"avgSim={decision.avg_sim:.3f}",
                        )

                if unknown_candidates:
                    best_unknown = sorted(unknown_candidates, key=lambda x: x.face_prob, reverse=True)[0]
                    dec = self.unknown_avg.update(best_unknown.face_prob, best_unknown.sim)
                    if dec and dec.should_trigger:
                        if dec.avg_face_prob >= self.cfg.unknown_face_prob_threshold and dec.avg_best_sim <= self.cfg.unknown_sim_upper:
                            if self._cooldown_ok("UNKNOWN_PERSON"):
                                ev = self._save_evidence(
                                    "unknown",
                                    frame,
                                    self.camera_name,
                                    "UNKNOWN_PERSON",
                                    identity="Unknown",
                                    sim=dec.avg_best_sim,
                                    conf=dec.avg_face_prob,
                                    bbox=best_unknown.bbox,
                                )
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

            self._log_perf()
            elapsed = time.time() - loop_t0
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

        capture_thread.join(timeout=1.0)
        add_log(self.cfg.log_db_path, "INFO", f"[{self.camera_name}] Stopped")
