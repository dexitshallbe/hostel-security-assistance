import os
import cv2
import torch
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1

def variance_of_laplacian(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

@torch.no_grad()
def main(
    video_path: str,
    out_dir: str = "out_faces",
    sample_every_n_frames: int = 3,
    min_face_prob: float = 0.80,
    min_blur_var: float = 80.0,
    min_embed_dist: float = 0.50,   # bigger = more strict about "different"
    max_saves: int = 80,
    device: str | None = None,
):
    video_path = str(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # MTCNN for detection + aligned face crops (160x160 by default)
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=60,
        thresholds=[0.6, 0.7, 0.7],
        post_process=True,
        device=device
    )

    # FaceNet embedder
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    saved = 0
    frame_idx = 0
    prev_embeds = []  # store embeddings of saved frames to avoid duplicates

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_idx += 1
        if frame_idx % sample_every_n_frames != 0:
            continue

        # quick blur filter on full frame (optional but cheap)
        if variance_of_laplacian(frame_bgr) < min_blur_var:
            continue

        # facenet_pytorch expects RGB PIL/numpy
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        face_tensor, prob = mtcnn(frame_rgb, return_prob=True)
        if face_tensor is None or prob is None or prob < min_face_prob:
            continue

        # face_tensor: [3,160,160], normalize for embedder
        face_tensor = face_tensor.unsqueeze(0).to(device)
        emb = embedder(face_tensor).cpu().numpy()[0]
        emb = emb / (np.linalg.norm(emb) + 1e-9)

        # check if this face is "different enough" from previously saved ones
        is_duplicate = False
        for e in prev_embeds:
            dist = np.linalg.norm(emb - e)
            if dist < min_embed_dist:
                is_duplicate = True
                break
        if is_duplicate:
            continue

        prev_embeds.append(emb)

        # face_tensor is [1,3,160,160]
        t = face_tensor[0].permute(1, 2, 0).cpu().numpy()  # HWC, float
        # Undo fixed_image_standardization: (x - 127.5) / 128
        img = (t * 128.0 + 127.5)
        img = np.clip(img, 0, 255).astype(np.uint8)
        face_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out_path = out_dir / f"face_{saved:04d}_p{prob:.3f}_f{frame_idx}.jpg"
        cv2.imwrite(str(out_path), face_bgr)



        saved += 1
        if saved >= max_saves:
            break

    cap.release()
    print(f"Done. Saved {saved} unique face crops to: {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to input video")
    ap.add_argument("--out", default="out_faces")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--min_prob", type=float, default=0.80)
    ap.add_argument("--blur", type=float, default=80.0)
    ap.add_argument("--min_dist", type=float, default=0.50)
    ap.add_argument("--max", type=int, default=80)
    args = ap.parse_args()

    main(
        video_path=args.video,
        out_dir=args.out,
        sample_every_n_frames=args.every,
        min_face_prob=args.min_prob,
        min_blur_var=args.blur,
        min_embed_dist=args.min_dist,
        max_saves=args.max,
    )
