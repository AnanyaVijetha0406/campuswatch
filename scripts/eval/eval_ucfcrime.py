"""
Evaluate a checkpoint against UCF-Crime's Abuse category vs. Normal videos —
a closer proxy for ragging than pure "Fighting" clips, since Abuse involves
a one-sided perpetrator/victim dynamic (restraining, dragging, repeated
hitting) rather than a mutual fight.

We've never trained on any UCF-Crime data, so all 50 Abuse videos are valid
held-out positives, not just the 2 in the official test split. UCF-Crime
only ships frame-level anomaly windows for its official test videos though:
  <filename> <class> <start1> <end1> <start2> <end2>   (-1 = unused)
For those, we sample within the labeled window (+ margin) — a fair test.
For the rest (no known window), we sample uniformly across the whole video,
same as everywhere else in this project — a conservative lower bound, since
16 frames spread across a long video may miss a short abuse event entirely.

Usage (Kaggle):
  python scripts/eval/eval_ucfcrime.py \
    --abuse-dir "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Abuse" \
    --normal-dir "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Testing_Normal_Videos/Testing_Normal_Videos_Anomaly" \
    --annotation-file "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt" \
    --model "/kaggle/input/datasets/<you>/<checkpoints>/best_model_ubifights_finetuned.pt" \
    --threshold 0.5501 \
    --cache-dir /kaggle/working/feature_cache/ucfcrime \
    --out /kaggle/working/ucfcrime_eval_ubifights_finetuned.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
import model as model_module
from model import (
    ViolenceClassifier,
    count_zero_frames,
    encode_frame_persons,
    extract_video_features,
    get_cnn_encoder,
    get_yolo,
    load_classifier,
    sample_frame_indices,
)


def parse_annotation_file(path: Path) -> dict[str, tuple[int, int]]:
    """filename -> (min_start, max_end) merged across up to 2 labeled windows."""
    windows: dict[str, tuple[int, int]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            filename = parts[0]
            nums = [int(p) for p in parts[2:6]]
            valid = [n for n in nums if n >= 0]
            if not valid:
                continue
            windows[filename] = (min(valid), max(valid))
    return windows


def extract_windowed_features(
    video_path: Path,
    window: tuple[int, int] | None,
    num_frames: int,
    device: str,
    yolo,
    cnn,
    k: int,
) -> torch.Tensor:
    """Same YOLO Top-K -> ResNet18 -> bbox pipeline as extract_video_features,
    optionally restricted to a (start_frame, end_frame) window with a margin."""
    if window is None:
        return extract_video_features(
            video_path, add_augmentation=False, device=device, yolo=yolo, cnn=cnn, num_persons=k
        )

    tfm = model_module._eval_transform()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    start, end = window
    margin = max(int(0.15 * (end - start)), 15)
    start = max(0, start - margin)
    end = min(total - 1, end + margin) if total else end + margin
    span = max(end - start + 1, 1)

    local_indices = sample_frame_indices(span, num_frames)
    feats = []
    for local_idx in local_indices:
        frame_idx = start + int(local_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            if k > 1:
                feats.append(model_module._zero_persons(k))
            else:
                feats.append(np.zeros(model_module.FEATURE_DIM, dtype=np.float32))
            continue
        person_feats = encode_frame_persons(frame, yolo, cnn, tfm, device, k=k)
        feats.append(person_feats if k > 1 else person_feats[0])

    cap.release()
    arr = np.stack(feats, axis=0)
    return torch.from_numpy(arr)


def load_model(model_path: Path, device: str) -> ViolenceClassifier:
    return load_classifier(model_path, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on UCF-Crime Abuse vs Normal.")
    parser.add_argument("--abuse-dir", type=Path, required=True)
    parser.add_argument("--normal-dir", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit-normal", type=int, default=50, help="Cap on how many Normal videos to sample (default: 50, roughly matching Abuse count)")
    parser.add_argument("--out", type=Path, default=Path("ucfcrime_eval_results.json"))
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")

    windows = parse_annotation_file(args.annotation_file)
    abuse_videos = sorted(args.abuse_dir.glob("*.mp4"))
    normal_videos = sorted(args.normal_dir.glob("*.mp4"))[: args.limit_normal]

    rows: list[tuple[Path, int, tuple[int, int] | None]] = []
    for v in abuse_videos:
        rows.append((v, 1, windows.get(v.name)))
    for v in normal_videos:
        rows.append((v, 0, None))

    n_windowed = sum(1 for _, label, w in rows if label == 1 and w is not None)
    print(f"Evaluating {len(abuse_videos)} Abuse videos ({n_windowed} with a labeled window) + {len(normal_videos)} Normal videos")

    net = load_model(args.model, device)
    k = net.num_persons
    print(f"Classifier K={k}")
    yolo = get_yolo()
    cnn = get_cnn_encoder(device)

    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_prob, per_clip = [], [], []
    total_zero_frames = 0
    total_frames = 0
    start_t = time.perf_counter()

    for i, (video_path, label, window) in enumerate(rows, 1):
        cache_path = (args.cache_dir / f"{video_path.stem}__k{k}.pt") if args.cache_dir else None
        if cache_path and cache_path.exists():
            feats = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            feats = extract_windowed_features(
                video_path, window, config.NUM_FRAMES, device, yolo, cnn, k=k
            )
            if cache_path:
                torch.save(feats, cache_path)

        zero_frames = count_zero_frames(feats)
        total_zero_frames += zero_frames
        total_frames += feats.shape[0]

        with torch.no_grad():
            x = feats.unsqueeze(0).to(device)
            logit = net(x).squeeze().item()
            prob = float(torch.sigmoid(torch.tensor(logit)).item())

        y_true.append(label)
        y_prob.append(prob)
        per_clip.append({
            "video": video_path.name,
            "label": label,
            "windowed": window is not None,
            "probability": prob,
            "zero_frames": zero_frames,
        })

        if i % 20 == 0 or i == len(rows):
            elapsed = time.perf_counter() - start_t
            print(f"  [{i}/{len(rows)}] {elapsed:.0f}s elapsed")

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    y_pred_arr = (y_prob_arr >= args.threshold).astype(int)
    zero_frame_rate = (total_zero_frames / total_frames) if total_frames else 0.0

    results = {
        "model": str(args.model),
        "threshold": args.threshold,
        "n_samples": int(len(y_true_arr)),
        "n_abuse_windowed": n_windowed,
        "n_abuse_unwindowed": len(abuse_videos) - n_windowed,
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
        "confusion_matrix_labels": ["normal", "abuse"],
        "zero_vector_frame_rate": zero_frame_rate,
        "per_clip": per_clip,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1:        {results['f1']:.4f}")
    print("Confusion matrix (rows=true, cols=pred) [normal, abuse]:")
    print(np.array(results["confusion_matrix"]))
    print(f"Zero-vector frame rate: {zero_frame_rate * 100:.1f}%")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
