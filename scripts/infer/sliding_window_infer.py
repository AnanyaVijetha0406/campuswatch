"""
Sliding-window inference for long or continuous video -- reuses the existing
trained classifier unchanged, no retraining needed. Fixes two things the
single whole-video sample can't:

1. Signal dilution on long video: instead of 16 frames spread across an
   entire clip (which can miss a brief incident entirely), each short
   window gets its own dedicated 16-frame sample.
2. Temporal localization: instead of one verdict for the whole video, you
   get a probability-over-time timeline -- "ragging appears to start
   around 0:16" instead of just "yes/no".

An alert only fires after CONSECUTIVE_REQUIRED windows in a row score above
threshold (hysteresis) -- a single noisy window crossing the line isn't
enough, avoiding the false-alarm-flood problem discussed for real-time use.

Usage:
  python scripts/infer/sliding_window_infer.py path/to/video.mp4
  python scripts/infer/sliding_window_infer.py path/to/video.mp4 --window-sec 4 --stride-sec 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
import model as model_module
from model import (
    encode_frame_persons,
    get_cnn_encoder,
    get_yolo,
    load_classifier,
    sample_frame_indices,
)


def make_windows(total_frames: int, fps: float, window_sec: float, stride_sec: float) -> list[tuple[int, int]]:
    window_frames = max(int(round(window_sec * fps)), 1)
    stride_frames = max(int(round(stride_sec * fps)), 1)
    if total_frames <= window_frames:
        return [(0, total_frames - 1)]
    windows = []
    start = 0
    while start < total_frames:
        end = min(start + window_frames - 1, total_frames - 1)
        windows.append((start, end))
        if end == total_frames - 1:
            break
        start += stride_frames
    return windows


def extract_window_features(cap, start: int, end: int, num_frames: int, k: int, yolo, cnn, tfm, device: str) -> torch.Tensor:
    span = max(end - start + 1, 1)
    local_indices = sample_frame_indices(span, num_frames)
    feats = []
    for local_idx in local_indices:
        frame_idx = start + int(local_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            feats.append(model_module._zero_persons(k) if k > 1 else np.zeros(model_module.FEATURE_DIM, dtype=np.float32))
            continue
        person_feats = encode_frame_persons(frame, yolo, cnn, tfm, device, k=k)
        feats.append(person_feats if k > 1 else person_feats[0])
    return torch.from_numpy(np.stack(feats, axis=0))


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sliding-window ragging/violence inference over a video timeline.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=None, help="Defaults to config.THRESHOLD / metrics.json")
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--stride-sec", type=float, default=2.0)
    parser.add_argument("--consecutive-required", type=int, default=2, help="Windows in a row above threshold before an ALERT fires")
    args = parser.parse_args()

    if not args.video.is_file():
        print(f"Error: video not found at {args.video}")
        sys.exit(1)

    device = config.DEVICE
    print(f"Device: {device}")
    net = load_classifier(args.model, device=device)
    k = net.num_persons
    print(f"Classifier K={k}")

    threshold = args.threshold if args.threshold is not None else config.THRESHOLD
    print(f"Threshold: {threshold:.4f}  Window: {args.window_sec}s  Stride: {args.stride_sec}s\n")

    yolo = get_yolo()
    cnn = get_cnn_encoder(device)
    tfm = model_module._eval_transform()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Error: cannot open {args.video}")
        sys.exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = total_frames / fps if fps else 0
    print(f"Duration: {fmt_time(duration)} ({total_frames} frames @ {fps:.1f} fps)\n")

    windows = make_windows(total_frames, fps, args.window_sec, args.stride_sec)
    print(f"{len(windows)} windows\n")

    consecutive = 0
    alerts: list[tuple[float, float]] = []
    for start, end in windows:
        feats = extract_window_features(cap, start, end, config.NUM_FRAMES, k, yolo, cnn, tfm, device)
        with torch.no_grad():
            x = feats.unsqueeze(0).to(device)
            logit = net(x).squeeze().item()
            prob = float(torch.sigmoid(torch.tensor(logit)).item())

        t0, t1 = start / fps, (end + 1) / fps
        above = prob >= threshold
        consecutive = consecutive + 1 if above else 0
        fired = consecutive == args.consecutive_required
        if fired:
            alerts.append((t0, t1))
        flag = " <-- ALERT" if fired else (" (above, building)" if above else "")
        print(f"  [{fmt_time(t0)}-{fmt_time(t1)}] prob={prob:.4f}{flag}")

    cap.release()

    print()
    if alerts:
        print(f"{len(alerts)} sustained alert(s):")
        for t0, t1 in alerts:
            print(f"  -> flagged starting around {fmt_time(t0)}")
    else:
        print("No sustained alert (no run of consecutive high-probability windows).")


if __name__ == "__main__":
    main()
