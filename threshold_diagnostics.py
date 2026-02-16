"""Threshold diagnostics for face-recognition embeddings.

Usage:
  python threshold_diagnostics.py --known-dir data/known --db-path data/logs/log.db
"""

import argparse
import json

from src.face_recog.recognizer import FaceRecognizer


def main():
    parser = argparse.ArgumentParser(description="Print embedding stats and suggested thresholds")
    parser.add_argument("--known-dir", default="data/known")
    parser.add_argument("--db-path", default="data/logs/log.db")
    parser.add_argument("--image-size", type=int, default=160)
    args = parser.parse_args()

    rec = FaceRecognizer(device="cpu", image_size=args.image_size)
    rec.build_database(args.known_dir, args.db_path)
    diagnostics = rec.diagnostics()

    print("=== Face Threshold Diagnostics ===")
    print(json.dumps(diagnostics, indent=2))
    print("\nSuggested config values:")
    th = diagnostics.get("calibrated_thresholds", {})
    print(f"ACCEPT_THRESHOLD={th.get('accept_threshold', 0.82):.3f}")
    print(f"GAP_THRESHOLD={th.get('gap_threshold', 0.08):.3f}")
    print(f"UNKNOWN_REJECT_THRESHOLD={th.get('unknown_reject_threshold', 0.72):.3f}")


if __name__ == "__main__":
    main()
