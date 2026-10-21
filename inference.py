"""
Shared inference core for the Flask dashboard (app.py) and the Gradio Space
(gradio_app.py): model loading, warm-up, video checks and single-clip prediction.

Settings come from environment variables:
  MAX_VIDEO_SECONDS, TORCH_THREADS
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

import config
from model import extract_video_features, get_cnn_encoder, get_yolo, load_classifier

MAX_VIDEO_SECONDS = float(os.environ.get("MAX_VIDEO_SECONDS", "60"))

# Oversubscribed CPU threads made inference time swing from ~4 s to ~50 s.
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", str(min(4, os.cpu_count() or 1))))
torch.set_num_threads(TORCH_THREADS)
cv2.setNumThreads(1)

device: str = config.DEVICE
threshold: float = float(config.THRESHOLD)
accuracy_label: str = "VALIDATION ACC."
accuracy_display: str = "N/A"

_model = None
_infer_lock = threading.Lock()  # models are shared globals; run one inference at a time


def load_metrics() -> dict:
    path = config.METRICS_PATH
    if not path.is_file():
        print(f"Warning: {path} not found — using config.THRESHOLD={config.THRESHOLD}")
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _accuracy_from(metrics: dict) -> tuple[str, str]:
    """Return (label, value). Reports held-out test accuracy when the metrics
    file has it, otherwise validation accuracy, labelled honestly."""
    acc = metrics.get("test_at_tuned_threshold", {}).get("accuracy")
    label = "TEST ACCURACY"
    if acc is None:
        acc = metrics.get("val_at_tuned_threshold", {}).get("accuracy")
        label = "VALIDATION ACC."
    if acc is None:
        return label, "N/A"
    return label, f"{acc * 100:.1f}%"


def load_model():
    global _model
    if _model is not None:
        return _model
    path = config.MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found at {path}.")
    _model = load_classifier(path, device=device)
    print(f"Loaded model K={_model.num_persons} from {path} on {device}")
    return _model


def warm_up() -> None:
    """Load YOLO/ResNet and run each once so the first real request is not slow."""
    model = load_model()
    get_yolo().predict(np.zeros((240, 320, 3), dtype=np.uint8), verbose=False, device=device)
    with torch.no_grad():
        get_cnn_encoder(device)(torch.zeros(1, 3, 224, 224, device=device))
        model(torch.zeros(1, config.NUM_FRAMES, config.INPUT_SIZE, device=device))


def init() -> None:
    """Read metrics, set the threshold, load models and warm up."""
    global threshold, accuracy_label, accuracy_display
    metrics = load_metrics()
    threshold = float(metrics.get("best_threshold", config.THRESHOLD))
    print(f"Using threshold={threshold:.4f}")
    accuracy_label, accuracy_display = _accuracy_from(metrics)
    warm_up()  # fail fast if weights are missing
    print(f"Warm-up done (torch threads={TORCH_THREADS})")


def check_video(path: Path) -> None:
    """Reject files OpenCV cannot read or that exceed the duration cap."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("The file could not be read as a video.")
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if frames <= 0:
            raise ValueError("The video has no readable frames.")
        if fps > 0 and frames / fps > MAX_VIDEO_SECONDS:
            raise ValueError(
                f"Video is {frames / fps:.0f}s long; the limit is {MAX_VIDEO_SECONDS:.0f}s."
            )
    finally:
        cap.release()


@torch.no_grad()
def predict_video(video_path: Path) -> dict:
    """Single whole-clip sample: NUM_FRAMES sampled across the entire video,
    one forward pass, one verdict."""
    model = load_model()
    with _infer_lock:
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
