"""Minimal runnable sanity check for recognizer decisions.

Example:
  python sanity_check_recognition.py --image data/known/Sandeep/IMG20260212125945.jpg
"""

import argparse
import cv2

from src.config import Config
from src.face_recog.recognizer import FaceRecognizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--known-dir", default="data/known")
    parser.add_argument("--db-path", default="data/logs/log.db")
    args = parser.parse_args()

    cfg = Config()
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Could not load image: {args.image}")

    rec = FaceRecognizer(device="cpu", image_size=cfg.mtcnn_image_size, margin=cfg.mtcnn_margin, min_face_size=cfg.mtcnn_min_face_size)
    rec.build_database(args.known_dir, args.db_path)

    results = rec.recognize(frame, cfg)
    print(f"detections={len(results)}")
    for idx, r in enumerate(results):
        print(
            f"#{idx} name={r.name} sim={r.sim:.3f} second={r.second_sim:.3f} margin={r.margin:.3f} "
            f"prob={r.face_prob:.2f} blur={r.blur_var:.1f} reason={r.decision_reason} bbox={r.bbox}"
        )


if __name__ == "__main__":
    main()
