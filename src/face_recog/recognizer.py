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


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm(p=2) + 1e-9)
    b = b / (b.norm(p=2) + 1e-9)
    return float((a * b).sum().item())


class FaceRecognizer:
    """
    - Detect + align faces using MTCNN
    - Embed using FaceNet (InceptionResnetV1)
    - Match against a database built from folders (one identity per folder)
    """
    def __init__(self, device: str = "cpu", image_size: int = 160, margin: int = 20, min_face_size: int = 40):
        self.device = device
        self.mtcnn = MTCNN(image_size=image_size, margin=margin, min_face_size=min_face_size,
                           keep_all=True, device=device)
        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

        self.db_embeddings: Dict[str, torch.Tensor] = {}  # name -> (N,512)
        self.db_built_at = 0.0

    def build_database(self, known_dir: str, db_path: str):
        """
        Builds embeddings for known + unexpired guests.
        """
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

                    # detect+align first face
                    faces, probs = self.mtcnn(rgb, return_prob=True)
                    if faces is None or probs is None:
                        continue

                    # probs can be Tensor, numpy array, or list depending on facenet-pytorch version
                    if isinstance(probs, torch.Tensor):
                        idx = int(torch.argmax(probs).item())
                    else:
                        import numpy as np
                        idx = int(np.argmax(probs))

                    face = faces[idx:idx+1].to(self.device)


                    with torch.no_grad():
                        emb = self.embedder(face).detach().cpu().squeeze(0)  # (512,)

                    all_embs.setdefault(person_name, []).append(emb)

        self.db_embeddings = {k: torch.stack(v, dim=0) for k, v in all_embs.items() if len(v) > 0}
        self.db_built_at = time.time()

    def recognize(self, frame_bgr, sim_threshold: float = 0.55) -> List[MatchResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        boxes, probs = self.mtcnn.detect(rgb)
        if boxes is None or len(boxes) == 0:
            return []

        # align all faces
        faces = self.mtcnn.extract(rgb, boxes, save_path=None)  # (N,3,160,160) or None
        if faces is None:
            return []

        faces = faces.to(self.device)
        with torch.no_grad():
            embs = self.embedder(faces).detach().cpu()  # (N,512)

        results: List[MatchResult] = []

        for i, box in enumerate(boxes):
            face_prob = float(probs[i]) if probs is not None else 0.0
            emb = embs[i]

            best_name = "Unknown"
            best_sim = -1.0

            for name, db_embs in self.db_embeddings.items():
                # max similarity across that identity's embeddings
                sims = torch.nn.functional.cosine_similarity(
                    emb.unsqueeze(0), db_embs, dim=1
                )
                s = float(torch.max(sims).item())
                if s > best_sim:
                    best_sim = s
                    best_name = name

            if best_sim < sim_threshold:
                best_name = "Unknown"

            x1, y1, x2, y2 = [int(v) for v in box]
            results.append(MatchResult((x1, y1, x2, y2), best_name, float(best_sim), face_prob))

        return results
