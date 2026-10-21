"""
CLI batch inference: run a trained checkpoint over every .mp4 in a folder
and print per-clip probability/verdict plus a final catch-rate summary.
Used for quick screening of a folder of clips outside the formal eval
scripts (compare_yolo_detectors.py, eval_ucfcrime.py, etc.).

Usage:
  python batch_infer.py [folder] [model_path] [threshold]
"""
import glob
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from model import extract_video_features, load_classifier

FOLDER = sys.argv[1] if len(sys.argv) > 1 else r"data/ragging_clips"
MODEL_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/best_model_combined_finetuned.pt")
THRESHOLD = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7355

device = config.DEVICE
print(f"Device: {device}", flush=True)

net = load_classifier(MODEL_PATH, device=device)
print(f"Model loaded (K={net.num_persons}).", flush=True)

files = sorted(glob.glob(os.path.join(FOLDER, "*.mp4")))
print(f"Found {len(files)} files\n", flush=True)

results = []
start = time.perf_counter()
for i, f in enumerate(files, 1):
    name = os.path.basename(f)
    t0 = time.perf_counter()
    try:
        feats = extract_video_features(
            f, add_augmentation=False, device=device, num_persons=net.num_persons
        )
        with torch.no_grad():
            x = feats.unsqueeze(0).to(device)
            logit = net(x).squeeze().item()
            prob = float(torch.sigmoid(torch.tensor(logit)).item())
        verdict = "RAGGING" if prob >= THRESHOLD else "no violence"
        elapsed = time.perf_counter() - t0
        print(f"[{i}/{len(files)}] {name:45s} prob={prob:.4f} {verdict:12s} ({elapsed:.1f}s)", flush=True)
        results.append((name, prob, verdict))
    except Exception as exc:
        print(f"[{i}/{len(files)}] {name:45s} ERROR: {exc}", flush=True)
        results.append((name, None, "ERROR"))

total = time.perf_counter() - start
print(f"\nTotal: {total:.0f}s for {len(files)} files", flush=True)

caught = sum(1 for _, p, v in results if v == "RAGGING")
errors = sum(1 for _, p, v in results if v == "ERROR")
print(f"Caught: {caught}/{len(files) - errors} (errors: {errors})", flush=True)
