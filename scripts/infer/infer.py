"""
Single-video inference from the command line.

Usage:
    python scripts/infer/infer.py path/to/video.mp4
    python scripts/infer/infer.py path/to/video.mp4 --model results/k1/best_model_ubifights_finetuned.pt --metrics results/k1/metrics_ubifights_finetuned.json

Defaults to the deployed model, config.MODEL_PATH / config.METRICS_PATH
(weights/), if --model/--metrics are omitted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from model import ViolenceClassifier, extract_video_features, load_classifier


def load_threshold(metrics_path: Path) -> float:
    if not metrics_path.is_file():
        print(f"Warning: {metrics_path} not found — using config.THRESHOLD={config.THRESHOLD}")
        return float(config.THRESHOLD)
    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)
    if "best_threshold" not in metrics:
        print(
            f"Warning: {metrics_path} has no 'best_threshold' key — "
            f"falling back to config.THRESHOLD={config.THRESHOLD}, which was tuned "
            "for a different checkpoint. Pass --threshold explicitly instead."
        )
    return float(metrics.get("best_threshold", config.THRESHOLD))


def load_model(model_path: Path, device: str):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    model = load_classifier(model_path, device=device)
    print(f"Loaded classifier K={model.num_persons} from {model_path}")
    return model


@torch.no_grad()
def predict(video_path: Path, model: ViolenceClassifier, threshold: float, device: str) -> dict:
    feats = extract_video_features(
        video_path,
        add_augmentation=False,
        device=device,
        num_persons=model.num_persons,
    )
    x = feats.unsqueeze(0).to(device)
    logit = model(x).squeeze().item()
    prob = float(torch.sigmoid(torch.tensor(logit)).item())
    is_violence = prob >= threshold
    confidence = prob if is_violence else (1.0 - prob)
    return {
        "is_violence": is_violence,
        "verdict": "Violence / ragging detected" if is_violence else "No violence detected",
        "probability": prob,
        "confidence": confidence,
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run violence-detection inference on a single video.")
    parser.add_argument("video", type=Path, help="Path to a video file (.mp4, .avi, .mov, .mkv, .webm)")
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH, help="Path to checkpoint .pt (default: deployed model in weights/)")
    parser.add_argument("--metrics", type=Path, default=config.METRICS_PATH, help="Path to metrics.json for tuned threshold (default: deployed metrics in weights/)")
    parser.add_argument("--threshold", type=float, default=None, help="Override the decision threshold directly, skipping metrics.json")
    args = parser.parse_args()

    if not args.video.is_file():
        print(f"Error: video not found at {args.video}")
        sys.exit(1)

    device = config.DEVICE
    print(f"Device: {device}")
    print(f"Loading model from {args.model} ...")
    model = load_model(args.model, device)
    threshold = args.threshold if args.threshold is not None else load_threshold(args.metrics)
    print(f"Using threshold: {threshold:.4f}")

    start = time.perf_counter()
    result = predict(args.video, model, threshold, device)
    elapsed = time.perf_counter() - start

    print()
    print(f"Video:      {args.video}")
    print(f"Verdict:    {result['verdict']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Confidence:  {result['confidence'] * 100:.1f}%")
    print(f"Threshold:   {result['threshold']:.4f}")
    print(f"Time:        {elapsed:.1f}s")


if __name__ == "__main__":
    main()
