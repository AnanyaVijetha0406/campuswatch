"""Shared constants — single source of truth for paths and settings.

Paths are relative to the project root so the same code works locally and
on Kaggle (only edit path values if your Kaggle input mount differs).
"""
from pathlib import Path

import os

try:
    import torch

    # CAMPUSWATCH_DEVICE forces a device (the ZeroGPU Space pins "cpu" because
    # CUDA is only emulated outside @spaces.GPU functions).
    DEVICE = os.environ.get("CAMPUSWATCH_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
except ImportError:
    DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Project / data layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_RAW_DIR = DATA_DIR / "train_raw"
SAMPLE_DIR = DATA_DIR / "sample"
SPLITS_DIR = DATA_DIR / "splits"

# Cached features: v2 is (NUM_FRAMES, PERSONS_PER_FRAME, INPUT_SIZE)
CACHE_DIR = DATA_DIR / "feature_cache"

VIOLENCE_CLASS = "Violence"
NONVIOLENCE_CLASS = "NonViolence"
CLASS_NAMES = (VIOLENCE_CLASS, NONVIOLENCE_CLASS)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# ---------------------------------------------------------------------------
# Model / feature constants
# ---------------------------------------------------------------------------
NUM_FRAMES = 16
# Top-K persons kept per frame (v1 was 1 = largest bbox only). Extra detections
# beyond K are dropped. Fewer than K are zero-padded and masked in person attention.
# Deployed model is K=1 (the K=3 closest-group ablation underperformed it on the
# real ragging-clip test -- same recall, ~2x the false-positive rate -- and a
# threshold sweep confirmed it wasn't just a calibration issue; see results/k3/).
PERSONS_PER_FRAME = 1
# How to pick which K people when YOLO sees more than K:
#   "closest_group" — tightest cluster of K box centers (interaction-centric)
#   "area"          — largest boxes (closest to camera; v1-style)
PERSON_SELECT = "area"
# ResNet18 CNN (512) + normalized bbox xyxy (4) — per person, then attention-pooled
# Scene context (reviewer ablation): append a 512-d whole-frame ResNet18 feature
# to the 516-d person feature. Off by default; the deployed model is 516-d.
SCENE_CONTEXT = os.environ.get("CAMPUSWATCH_SCENE", "0") == "1"
INPUT_SIZE = 516 + (512 if SCENE_CONTEXT else 0)
CNN_FEAT_DIM = 512
BBOX_DIM = 4

THRESHOLD = 0.6613  # default; app.py prefers best_threshold from metrics.json
RANDOM_SEED = int(os.environ.get("CAMPUSWATCH_SEED", "42"))
# Temporal head: "bilstm" (deployed model) or "transformer" (reviewer baseline).
TEMPORAL_HEAD = os.environ.get("CAMPUSWATCH_TEMPORAL", "bilstm")
WEIGHTS_DIR = PROJECT_ROOT / "weights"
YOLO_WEIGHTS = str(WEIGHTS_DIR / "yolov8n.pt")

# ---------------------------------------------------------------------------
# Training (Phase 2) — single source of truth
# ---------------------------------------------------------------------------
# Default output folder for new training runs (git-ignored).
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Deployed model: K=1 (largest person per frame). RLVS -> fine-tuned once on
# UBI-FIGHTS + self-collected ragging data, without UCF-Crime -- best real-world
# recall (5/6 on held-out ragging clips) among every checkpoint tested.
MODEL_PATH = WEIGHTS_DIR / "best_model_combined2_finetuned.pt"
METRICS_PATH = WEIGHTS_DIR / "metrics_combined2_finetuned.json"

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
DROPOUT = 0.3
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3

# Label mapping: Violence=1 (positive / ragging), NonViolence=0
LABEL_MAP = {VIOLENCE_CLASS: 1, NONVIOLENCE_CLASS: 0}
