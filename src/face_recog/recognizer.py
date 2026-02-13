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


class FaceRecognizer:
    """
    Face recognition with:
    - MTCNN detect + align
    - InceptionResnetV1 embeddings (vggface2)
    - L2 normalization
    - DB per-identity ref stack + centroid
    - dynamic thresholds based on identity count
    - proper open-set logic for 1 identity
    - optional best-vs-second margin when 2+ identities exist
    """

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

        # name -> (N,512) normalized
        self.db_embeddings: Dict[str, torch.Tensor] = {}
        # name -> (512,) normalized centroid
        self.db_centroids: Dict[str, torch.Tensor] = {}

        self.db_built_at = 0.0

    # ---------------------------
    # Internal helpers
    # ---------------------------
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

    def _thresholds_for_count(self, n_id: int, cfg) -> Tuple[float, float]:
        """
        Returns (sim_threshold, sim_margin) based on identity count.
        """
        if n_id <= 1:
            return float(cfg.sim_threshold_1), float(cfg.sim_margin_1)

        # For 2 identities, treat similarly to "few"
        if 2 <= n_id < 10:
            return float(cfg.sim_threshold_few), float(cfg.sim_margin_few)

        return float(cfg.sim_threshold_many), float(cfg.sim_margin_many)

    # ---------------------------
    # DB build
    # ---------------------------
    def build_database(self, known_dir: str, db_path: str):
        dirs = list_identity_folders(known_dir, db_path)
        all_embs: Dict[str, List[torch.Tensor]] = {}

        for root in dirs:
            for person_name, person_dir in iter_person_folders(root):
                imgs = []
                for fn in os.listdir(person_dir):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        imgs.append(os.path.join(person_dir, fn))
                if not imgs:
                    continue

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
                        emb = self._l2_norm_vec(emb)

                    all_embs.setdefault(person_name, []).append(emb)

        self.db_embeddings = {}
        self.db_centroids = {}

        for name, embs in all_embs.items():
            stack = torch.stack(embs, dim=0)            # (N,512)
            stack = self._l2_norm_mat(stack)            # normalize rows
            self.db_embeddings[name] = stack

            centroid = stack.mean(dim=0)                # (512,)
            centroid = self._l2_norm_vec(centroid)
            self.db_centroids[name] = centroid

        self.db_built_at = time.time()

    # ---------------------------
    # Recognition
    # ---------------------------
    def recognize(self, frame_bgr, cfg) -> List[MatchResult]:
        """
        Uses cfg dynamic thresholds + gates:
          cfg.min_face_prob, cfg.min_face_w, cfg.min_face_h
          cfg.sim_threshold_1/few/many and sim_margin_*
        """
        n_id = self.identity_count()
        sim_threshold, sim_margin = self._thresholds_for_count(n_id, cfg)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        boxes, probs = self.mtcnn.detect(rgb)
        if boxes is None or len(boxes) == 0:
            return []

        faces = self.mtcnn.extract(rgb, boxes, save_path=None)
        if faces is None:
            return []

        faces = faces.to(self.device)

        with torch.no_grad():
            embs = self.embedder(faces).detach().cpu()    # (N,512)
            embs = self._l2_norm_mat(embs)

        results: List[MatchResult] = []

        # If DB empty, everything unknown
        if n_id == 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                face_prob = float(probs[i]) if probs is not None else 0.0
                results.append(MatchResult((x1, y1, x2, y2), "Unknown", -1.0, face_prob, -1.0, 0.0))
            return results

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            w, h = (x2 - x1), (y2 - y1)

            face_prob = float(probs[i]) if probs is not None else 0.0

            # Quality gates
            if face_prob < float(cfg.min_face_prob) or w < int(cfg.min_face_w) or h < int(cfg.min_face_h):
                results.append(MatchResult((x1, y1, x2, y2), "Unknown", -1.0, face_prob, -1.0, 0.0))
                continue

            emb = embs[i]

            best_name = "Unknown"
            best_sim = -1.0
            second_sim = -1.0

            for name, db_embs in self.db_embeddings.items():
                # max similarity to stored refs
                sims_refs = torch.nn.functional.cosine_similarity(emb.unsqueeze(0), db_embs, dim=1)
                s_refs = float(sims_refs.max().item())

                # centroid similarity
                cent = self.db_centroids.get(name, None)
                if cent is not None:
                    s_cent = float(torch.nn.functional.cosine_similarity(
                        emb.unsqueeze(0), cent.unsqueeze(0), dim=1
                    ).item())
                    s = max(s_refs, s_cent)
                else:
                    s = s_refs

                if s > best_sim:
                    second_sim = best_sim
                    best_sim = s
                    best_name = name
                elif s > second_sim:
                    second_sim = s

            # margin is only meaningful if we truly have >=2 identities
            margin_val = (best_sim - second_sim) if second_sim > -0.5 else 0.0

            # Decision logic:
            # 1) Always need to pass sim_threshold
            accept = best_sim >= sim_threshold

            # 2) If 2+ identities exist, enforce margin
            if accept and n_id >= 2 and second_sim > -0.5:
                accept = (best_sim - second_sim) >= sim_margin

            # 3) If ONLY 1 identity exists, enforce strict open-set threshold (cfg.sim_threshold_1)
            # (already applied as sim_threshold in this case)
            if not accept:
                best_name = "Unknown"

            results.append(MatchResult(
                (x1, y1, x2, y2),
                best_name,
                float(best_sim),
                float(face_prob),
                float(second_sim),
                float(margin_val),
            ))

        return results