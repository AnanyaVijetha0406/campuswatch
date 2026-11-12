"""
Train BiLSTM+Attention on CACHED features (Phase 2).

Does NOT run YOLO/CNN inside the training loop — expects .pt files from
extract_features.py under config.CACHE_DIR/{train,val,test}/.

Writes:
  outputs/best_model.pt
  outputs/metrics.json  (includes tuned threshold from precision_recall_curve)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from model import ViolenceClassifier, expected_feature_shape


class CachedFeatureDataset(Dataset):
    """Load (NUM_FRAMES, K, INPUT_SIZE) tensors from cache_dir/{Violence,NonViolence}/*.pt"""

    def __init__(self, cache_dir: Path) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.expected_shape = expected_feature_shape(config.PERSONS_PER_FRAME)
        cache_dir = Path(cache_dir)
        for class_name, label in config.LABEL_MAP.items():
            class_dir = cache_dir / class_name
            if not class_dir.is_dir():
                class_dir = cache_dir / class_name.lower()
            if not class_dir.is_dir():
                continue
            for pt_path in sorted(class_dir.glob("*.pt")):
                self.samples.append((pt_path, label))
        if not self.samples:
            raise FileNotFoundError(
                f"No cached .pt files under {cache_dir}. "
                "Run extract_features.py for this split first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        feats = torch.load(path, map_location="cpu", weights_only=True).float()
        if tuple(feats.shape) != self.expected_shape:
            raise ValueError(
                f"{path.name}: expected {self.expected_shape} (Top-K v2 cache), "
                f"got {tuple(feats.shape)}. Re-run extract_features.py."
            )
        return feats, torch.tensor(label, dtype=torch.float32)


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def compute_pos_weight(dataset: CachedFeatureDataset) -> torch.Tensor:
    labels = [label for _, label in dataset.samples]
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = sum(1 for y in labels if y == 0)
    if n_pos == 0:
        return torch.tensor([1.0])
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


@torch.no_grad()
def collect_logits_labels(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).cpu().numpy()
        all_logits.extend(logits.tolist())
        all_labels.extend(y.numpy().astype(int).tolist())
    return np.array(all_logits), np.array(all_labels)


def tune_threshold(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Pick threshold maximizing F1 on validation logits (precision_recall_curve)."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    # thresholds has len = len(precisions) - 1
    if len(thresholds) == 0:
        return 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    f1s = []
    for i, thr in enumerate(thresholds):
        preds = (probs >= thr).astype(int)
        f1s.append(f1_score(labels, preds, zero_division=0))
    best_idx = int(np.argmax(f1s))
    best_thr = float(thresholds[best_idx])
    preds = (probs >= best_thr).astype(int)
    return best_thr, {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1s[best_idx]),
    }


def evaluate_at_threshold(
    logits: np.ndarray, labels: np.ndarray, threshold: float
) -> dict:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["nonviolence", "violence"],
        "n_samples": int(len(labels)),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def run_training(args: argparse.Namespace) -> dict:
    device = config.DEVICE
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    cache_root = Path(args.cache_dir or config.CACHE_DIR)
    train_ds = CachedFeatureDataset(cache_root / "train")
    val_ds = CachedFeatureDataset(cache_root / "val")
    test_ds = CachedFeatureDataset(cache_root / "test")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    pos_weight = compute_pos_weight(train_ds).to(device)
    print(
        f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)} "
        f"K={config.PERSONS_PER_FRAME} pos_weight={pos_weight.item():.3f} device={device}"
    )

    model = ViolenceClassifier(
        input_size=config.INPUT_SIZE,
        hidden_size=config.LSTM_HIDDEN,
        num_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT,
        num_persons=config.PERSONS_PER_FRAME,
        temporal=config.TEMPORAL_HEAD,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
    )

    out_dir = Path(args.output_dir or config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={lr:.2e}  ({elapsed:.1f}s)"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr}
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": config.INPUT_SIZE,
                    "num_frames": config.NUM_FRAMES,
                    "num_persons": config.PERSONS_PER_FRAME,
                    "temporal": config.TEMPORAL_HEAD,
                    "seed": config.RANDOM_SEED,
                    "person_select": config.PERSON_SELECT,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_path,
            )
            print(f"  -> saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop:
                print(f"Early stopping after {args.early_stop} epochs without improvement.")
                break

    # Load best weights for threshold tuning + test eval
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_logits, val_labels = collect_logits_labels(model, val_loader, device)
    best_threshold, val_tune = tune_threshold(val_logits, val_labels)
    print(f"Tuned threshold (val F1): {best_threshold:.4f}  {val_tune}")

    test_logits, test_labels = collect_logits_labels(model, test_loader, device)
    test_metrics = evaluate_at_threshold(test_logits, test_labels, best_threshold)
    test_metrics["split"] = "rlvs_test"
    test_metrics["tuned_on"] = "val"

    val_metrics = evaluate_at_threshold(val_logits, val_labels, best_threshold)
    val_metrics["split"] = "rlvs_val"

    metrics = {
        "device": device,
        "input_size": config.INPUT_SIZE,
        "num_frames": config.NUM_FRAMES,
        "num_persons": config.PERSONS_PER_FRAME,
        "temporal": config.TEMPORAL_HEAD,
        "seed": config.RANDOM_SEED,
        "person_select": config.PERSON_SELECT,
        "best_threshold": best_threshold,
        "best_val_loss": best_val_loss,
        "pos_weight": float(pos_weight.item()),
        "training": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs_run": len(history),
            "early_stop_patience": args.early_stop,
        },
        "history": history,
        "val_at_tuned_threshold": val_metrics,
        "test_at_tuned_threshold": test_metrics,
    }

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {metrics_path}")
    print(f"RLVS test: acc={test_metrics['accuracy']:.3f} "
          f"P={test_metrics['precision']:.3f} R={test_metrics['recall']:.3f} "
          f"F1={test_metrics['f1']:.3f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on cached video features.")
    parser.add_argument("--cache-dir", type=Path, default=config.CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--early-stop", type=int, default=config.EARLY_STOP_PATIENCE)
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
