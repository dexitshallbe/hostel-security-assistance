import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from .database import list_identity_folders, iter_person_folders


@dataclass
class MatchResult:
    bbox: Tuple[int, int, int, int]
    name: str
    sim: float
    face_prob: float
    second_sim: float
    margin: float
    decision_reason: str = ""
    blur_var: float = 0.0


class FaceRecognizer:
    """Face recognition with adaptive thresholding and quality gates."""

    def __init__(self, device: str = "cpu", image_size: int = 160, margin: int = 20, min_face_size: int = 40):
        self.device = device
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            keep_all=True,
            device=device,
        )
        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

        self.db_embeddings: Dict[str, torch.Tensor] = {}
        self.db_centroids: Dict[str, torch.Tensor] = {}
        self.db_built_at = 0.0
        self.calibrated_thresholds: Dict[str, float] = {}
        self.calibration_stats: Dict[str, float] = {}

    @staticmethod
    def _to_idx_from_probs(probs) -> int:
        if probs is None:
            return 0
        if isinstance(probs, torch.Tensor):
            return int(torch.argmax(probs).item())
        return int(np.argmax(probs))

    @staticmethod
    def _l2_norm_vec(v: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(v, p=2, dim=0)

    @staticmethod
    def _l2_norm_mat(m: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(m, p=2, dim=1)

    def identity_count(self) -> int:
        return len(self.db_embeddings)

    def _effective_thresholds(self, cfg) -> Tuple[float, float, float]:
        base_accept = float(getattr(cfg, "accept_threshold", 0.82))
        base_gap = float(getattr(cfg, "gap_threshold", 0.08))
        unknown_reject = float(getattr(cfg, "unknown_reject_threshold", 0.72))

        if self.calibrated_thresholds:
            accept = max(base_accept, float(self.calibrated_thresholds.get("accept_threshold", base_accept)))
            gap = max(base_gap, float(self.calibrated_thresholds.get("gap_threshold", base_gap)))
            unknown_reject = min(unknown_reject, float(self.calibrated_thresholds.get("unknown_reject_threshold", unknown_reject)))
            return accept, gap, unknown_reject

        # Backward compatibility fallback if calibration unavailable
        n_id = self.identity_count()
        if n_id <= 1:
            return float(cfg.sim_threshold_1), base_gap, unknown_reject
        if n_id < 10:
            return float(cfg.sim_threshold_few), float(cfg.sim_margin_few), unknown_reject
        return float(cfg.sim_threshold_many), float(cfg.sim_margin_many), unknown_reject

    def calibrate_thresholds(self) -> Dict[str, float]:
        intra: List[float] = []
        inter: List[float] = []
        ids = list(self.db_embeddings.keys())

        # Intra-class pairwise sims
        for name, embs in self.db_embeddings.items():
            if embs.shape[0] < 2:
                continue
            sim_mat = embs @ embs.T
            iu = torch.triu_indices(sim_mat.shape[0], sim_mat.shape[1], offset=1)
            intra.extend(sim_mat[iu[0], iu[1]].cpu().numpy().tolist())

        # Inter-class centroid similarities
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                c1 = self.db_centroids[ids[i]]
                c2 = self.db_centroids[ids[j]]
                inter.append(float(torch.dot(c1, c2).item()))

        accept = 0.82
        gap = 0.08
        unknown_reject = 0.72

        if intra:
            p10 = float(np.percentile(intra, 10))
            accept = max(0.70, min(0.95, p10 - 0.01))

        if inter:
            p95_inter = float(np.percentile(inter, 95))
            accept = max(accept, min(0.96, p95_inter + 0.05))
            gap = max(0.05, min(0.2, accept - p95_inter))
            unknown_reject = min(unknown_reject, max(0.60, p95_inter))
        elif self.identity_count() <= 1:
            # Single-identity open-set guard, remain conservative.
            accept = max(accept, 0.86)
            gap = 0.0

        self.calibration_stats = {
            "intra_count": float(len(intra)),
            "inter_count": float(len(inter)),
            "intra_mean": float(np.mean(intra)) if intra else 0.0,
            "inter_mean": float(np.mean(inter)) if inter else 0.0,
        }
        self.calibrated_thresholds = {
            "accept_threshold": float(accept),
            "gap_threshold": float(gap),
            "unknown_reject_threshold": float(unknown_reject),
        }
        return self.calibrated_thresholds

    def diagnostics(self) -> Dict[str, object]:
        return {
            "identities": len(self.db_embeddings),
            "embeddings_per_identity": {k: int(v.shape[0]) for k, v in self.db_embeddings.items()},
            "calibrated_thresholds": self.calibrated_thresholds,
            "calibration_stats": self.calibration_stats,
        }

    def build_database(self, known_dir: str, db_path: str):
        dirs = list_identity_folders(known_dir, db_path)
        all_embs: Dict[str, List[torch.Tensor]] = {}

        for root in dirs:
            for person_name, person_dir in iter_person_folders(root):
                imgs = [
                    os.path.join(person_dir, fn)
                    for fn in os.listdir(person_dir)
                    if fn.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                for path in imgs:
                    bgr = cv2.imread(path)
                    if bgr is None:
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    faces, probs = self.mtcnn(rgb, return_prob=True)
                    if faces is None or probs is None:
                        continue
                    if faces.ndim == 3:
                        faces = faces.unsqueeze(0)
                    idx = self._to_idx_from_probs(probs)
                    face = faces[idx:idx + 1].to(self.device)
                    with torch.no_grad():
                        emb = self.embedder(face).detach().cpu().squeeze(0)
                        all_embs.setdefault(person_name, []).append(self._l2_norm_vec(emb))

        self.db_embeddings = {}
        self.db_centroids = {}
        for name, embs in all_embs.items():
            stack = self._l2_norm_mat(torch.stack(embs, dim=0))
            self.db_embeddings[name] = stack
            self.db_centroids[name] = self._l2_norm_vec(stack.mean(dim=0))

        self.db_built_at = time.time()
        self.calibrate_thresholds()

    def _blur_variance(self, frame_bgr, box: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = box
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        crop = frame_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def recognize(self, frame_bgr, cfg) -> List[MatchResult]:
        accept_threshold, gap_threshold, unknown_reject = self._effective_thresholds(cfg)
        min_face_prob = float(getattr(cfg, "min_face_prob", 0.97))
        min_face_size = int(getattr(cfg, "min_face_size", min(getattr(cfg, "min_face_w", 120), getattr(cfg, "min_face_h", 120))))
        blur_threshold = float(getattr(cfg, "blur_threshold", 70.0))

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs = self.mtcnn.detect(rgb)
        if boxes is None or len(boxes) == 0:
            return []

        faces = self.mtcnn.extract(rgb, boxes, save_path=None)
        if faces is None:
            return []
        faces = faces.to(self.device)

        with torch.no_grad():
            embs = self._l2_norm_mat(self.embedder(faces).detach().cpu())

        n_id = self.identity_count()
        results: List[MatchResult] = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            w, h = (x2 - x1), (y2 - y1)
            face_prob = float(probs[i]) if probs is not None else 0.0
            blur_var = self._blur_variance(frame_bgr, (x1, y1, x2, y2))

            if face_prob < min_face_prob:
                results.append(MatchResult((x1, y1, x2, y2), "Unknown", -1.0, face_prob, -1.0, 0.0, f"reject:low_prob<{min_face_prob:.2f}", blur_var))
                continue
            if min(w, h) < min_face_size:
                results.append(MatchResult((x1, y1, x2, y2), "Unknown", -1.0, face_prob, -1.0, 0.0, f"reject:small_face<{min_face_size}", blur_var))
                continue
            if blur_var < blur_threshold:
                results.append(MatchResult((x1, y1, x2, y2), "Unknown", -1.0, face_prob, -1.0, 0.0, f"reject:blurry<{blur_threshold:.1f}", blur_var))
                continue
            if n_id == 0:
                results.append(MatchResult((x1, y1, x2, y2), "Unknown", -1.0, face_prob, -1.0, 0.0, "reject:empty_db", blur_var))
                continue

            emb = embs[i]
            best_name, best_sim, second_sim = "Unknown", -1.0, -1.0

            for name, db_embs in self.db_embeddings.items():
                s_refs = float(torch.nn.functional.cosine_similarity(emb.unsqueeze(0), db_embs, dim=1).max().item())
                s_cent = float(torch.nn.functional.cosine_similarity(emb.unsqueeze(0), self.db_centroids[name].unsqueeze(0), dim=1).item())
                s = max(s_refs, s_cent)
                if s > best_sim:
                    second_sim = best_sim
                    best_sim = s
                    best_name = name
                elif s > second_sim:
                    second_sim = s

            margin_val = (best_sim - second_sim) if second_sim > -0.5 else 0.0
            decision_reason = "accept"
            accept = best_sim >= accept_threshold
            if not accept:
                decision_reason = f"reject:sim<{accept_threshold:.3f}"
            elif n_id > 1 and second_sim > -0.5 and margin_val < gap_threshold:
                accept = False
                decision_reason = f"reject:gap<{gap_threshold:.3f}"
            elif n_id <= 1 and best_sim < max(accept_threshold, 0.86):
                accept = False
                decision_reason = "reject:single_id_strict"
            elif best_sim < unknown_reject:
                accept = False
                decision_reason = f"reject:unknown_guard<{unknown_reject:.3f}"

            if not accept:
                best_name = "Unknown"

            results.append(MatchResult(
                (x1, y1, x2, y2),
                best_name,
                float(best_sim),
                float(face_prob),
                float(second_sim),
                float(margin_val),
                decision_reason,
                blur_var,
            ))

        return results
