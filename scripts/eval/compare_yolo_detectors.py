"""
Compare YOLOv8n vs a larger YOLO variant as the person detector, WITHOUT
retraining -- swap the detector only in feature extraction, then run the
same already-trained classifier on both feature sets. Tests whether a
stronger detector changes the zero-vector (no-person-detected) rate and
downstream predictions, at a fraction of the cost of a full retrain.

Usage:
  python scripts/eval/compare_yolo_detectors.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ultralytics import YOLO

import config
from model import count_zero_frames, extract_video_features, get_cnn_encoder, load_classifier

MODEL_PATH = config.MODEL_PATH
DETECTORS = [str(config.WEIGHTS_DIR / "yolov8n.pt"), str(config.WEIGHTS_DIR / "yolov8s.pt")]

RAGGING_DIR = Path(r"data/ragging_test_clips")
RAGGING_CLIPS = [
    "ragging 1.mp4", "ragging 2.mp4", "ragging 3.mp4",
    "ragging_test.mp4", "school_fight.mp4", "pouring_water_ragging.mp4",
]
NORMAL_DIR = Path(r"data/normal_clips")


def main() -> None:
    device = config.DEVICE
    print(f"Device: {device}")

    net = load_classifier(MODEL_PATH, device=device)
    k = net.num_persons
    print(f"Classifier K={k} (from {MODEL_PATH.name})\n")

    clips = [(RAGGING_DIR / name, 1) for name in RAGGING_CLIPS]
    clips += [(p, 0) for p in sorted(NORMAL_DIR.glob("*.mp4"))]

    cnn = get_cnn_encoder(device)
    results: dict[str, list[dict]] = {}

    for weights in DETECTORS:
        print(f"=== Detector: {weights} ===")
        yolo = YOLO(weights)  # fresh instance each time -- get_yolo()'s global cache ignores the weights arg after the first call
        rows = []
        start = time.perf_counter()
        for video_path, label in clips:
            feats = extract_video_features(
                video_path, add_augmentation=False, device=device, yolo=yolo, cnn=cnn, num_persons=k
            )
            zero_frames = count_zero_frames(feats)
            with torch.no_grad():
                x = feats.unsqueeze(0).to(device)
                logit = net(x).squeeze().item()
                prob = float(torch.sigmoid(torch.tensor(logit)).item())
            rows.append({
                "name": video_path.name, "label": label, "prob": prob,
                "zero_frames": zero_frames, "total_frames": int(feats.shape[0]),
            })
            print(f"  {video_path.name:45s} prob={prob:.4f} zero_frames={zero_frames}/{feats.shape[0]}")
        results[weights] = rows
        print(f"  ({time.perf_counter() - start:.0f}s)\n")

    threshold = 0.6613
    print(f"=== Comparison at threshold {threshold} ===")
    base, alt = results[DETECTORS[0]], results[DETECTORS[1]]
    total_zero_base = sum(r["zero_frames"] for r in base)
    total_zero_alt = sum(r["zero_frames"] for r in alt)
    total_frames = sum(r["total_frames"] for r in base)
    print(f"Zero-vector rate: {DETECTORS[0]}={total_zero_base}/{total_frames} ({100*total_zero_base/total_frames:.1f}%)  "
          f"{DETECTORS[1]}={total_zero_alt}/{total_frames} ({100*total_zero_alt/total_frames:.1f}%)")

    flips = 0
    for b, a in zip(base, alt):
        v_b = int(b["prob"] >= threshold)
        v_a = int(a["prob"] >= threshold)
        if v_b != v_a:
            flips += 1
            print(f"  VERDICT FLIP: {b['name']}  {DETECTORS[0]}={b['prob']:.4f}({'catch' if v_b else 'miss'})  "
                  f"{DETECTORS[1]}={a['prob']:.4f}({'catch' if v_a else 'miss'})")
    print(f"Verdict flips: {flips}/{len(base)}")


if __name__ == "__main__":
    main()
