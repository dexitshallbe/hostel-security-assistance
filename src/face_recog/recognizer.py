import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


class FaceRecognizer:
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

                    # probs can be tensor or numpy
                    if isinstance(probs, torch.Tensor):
                        idx = int(torch.argmax(probs).item())
                    else:
                        idx = int(np.argmax(probs))

                    if faces.ndim == 3:  # just in case
                        faces = faces.unsqueeze(0)

                    face = faces[idx:idx+1].to(self.device)  # (1,3,160,160)

                    with torch.no_grad():
                        emb = self.embedder(face).detach().cpu().squeeze(0)  # (512,)
                        emb = torch.nn.functional.normalize(emb, p=2, dim=0)

                    all_embs.setdefault(person_name, []).append(emb)

        self.db_embeddings = {}
        self.db_centroids = {}

        for name, embs in all_embs.items():
            stack = torch.stack(embs, dim=0)  # (N,512)
            stack = torch.nn.functional.normalize(stack, p=2, dim=1)
            self.db_embeddings[name] = stack

            centroid = stack.mean(dim=0)
            centroid = torch.nn.functional.normalize(centroid, p=2, dim=0)
            self.db_centroids[name] = centroid

        self.db_built_at = time.time()

    def recognize(
        self,
        frame_bgr,
        sim_threshold: float,
        sim_margin: float,
        min_face_prob: float,
        min_face_w: int,
        min_face_h: int,
    ) -> List[MatchResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        boxes, probs = self.mtcnn.detect(rgb)
        if boxes is None or len(boxes) == 0:
            return []

        faces = self.mtcnn.extract(rgb, boxes, save_path=None)
        if faces is None:
            return []

        faces = faces.to(self.device)
        with torch.no_grad():
            embs = self.embedder(faces).detach().cpu()  # (N,512)
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)

        results: List[MatchResult] = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            w, h = (x2 - x1), (y2 - y1)

            face_prob = float(probs[i]) if probs is not None else 0.0

            # Quality gates
            if face_prob < min_face_prob or w < min_face_w or h < min_face_h:
                continue

            emb = embs[i]

            best_name = "Unknown"
            best_sim = -1.0
            second_sim = -1.0

            # Score using max(refs) and centroid
            for name, db_embs in self.db_embeddings.items():
                sims_refs = torch.nn.functional.cosine_similarity(emb.unsqueeze(0), db_embs, dim=1)
                s_refs = float(sims_refs.max().item())

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

            # Final accept decision: threshold + margin
            if best_sim < sim_threshold or (best_sim - second_sim) < sim_margin:
                best_name = "Unknown"

            results.append(MatchResult((x1, y1, x2, y2), best_name, float(best_sim), face_prob))

        return results