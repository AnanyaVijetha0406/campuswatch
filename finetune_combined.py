"""
Fine-tune a Top-K (v2) RLVS checkpoint once, on UBI-FIGHTS + self-collected
ragging data. Do not stack another fine-tune on an already-finetuned head
(avoids compounding catastrophic forgetting).

v2 architecture change: features are (NUM_FRAMES, K, 516) with K=3 largest
persons per frame, then learned person-attention pooling. A v1 RLVS
checkpoint (K=1, no person_attn) cannot be loaded as --base-model unless
you pass --allow-partial-v1-init (LSTM/fc copied, person_attn random).
Agreed protocol: train v2 on RLVS first (extract_features.py + main.py),
then pass that checkpoint here. Omit UCF-Crime flags (deployed recipe).

self-ragging-dir / self-normal-dir add real staged ragging clips -- the
actual target domain.

Unlike finetune_ubifights.py, this script properly tunes a decision
threshold from the held-out validation set's precision-recall curve and
saves it as "best_threshold" in metrics.json.

Kaggle — step 1 (RLVS v2 baseline):
  python extract_features.py --split-dir .../train --cache-dir /kaggle/working/feature_cache/rlvs_k3/train
  python extract_features.py --split-dir .../val   --cache-dir /kaggle/working/feature_cache/rlvs_k3/val
  python extract_features.py --split-dir .../test  --cache-dir /kaggle/working/feature_cache/rlvs_k3/test
  python main.py --cache-dir /kaggle/working/feature_cache/rlvs_k3 --output-dir /kaggle/working/outputs_rlvs_k3

Kaggle — step 2 (UBI + self, no UCF):
  python finetune_combined.py
    --ubifights-dir "/kaggle/input/datasets/intissarziani/ubi-fightsall/UBI_FIGHTS/videos"
    --self-ragging-dir "/kaggle/input/datasets/<you>/<ragging-dataset>/ragging_clips/ragging_clips"
    --self-normal-dir "/kaggle/input/datasets/<you>/<ragging-dataset>/normal/normal"
    --base-model /kaggle/working/outputs_rlvs_k3/best_model.pt
    --cache-dir /kaggle/working/feature_cache/combined_k3
    --output-dir /kaggle/working/outputs_k3
    --epochs 60 --patience 8 --lr 1e-5
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from model import (
    ViolenceClassifier,
    checkpoint_num_persons,
    expected_feature_shape,
    extract_and_cache_features,
)


def collect_examples(
    ubifights_dir: Path,
    ucfcrime_abuse_dir: Path | None,
    ucfcrime_normal_dir: Path | None,
    limit_ucfcrime_normal: int,
    self_ragging_dir: Path | None = None,
    self_normal_dir: Path | None = None,
) -> list[tuple[Path, int, str]]:
    """Returns [(video_path, label, source_tag), ...]. label: 1=violence/abuse, 0=normal.

    self_ragging_dir / self_normal_dir: real staged ragging clips and their
    matched hard-negative normal clips -- the actual target domain, unlike
    RLVS/UBI-FIGHTS/UCF-Crime which are all violence-adjacent proxies.

    ucfcrime_abuse_dir / ucfcrime_normal_dir are optional so this can also run
    as an ablation without UCF-Crime, to check whether it's actually helping
    or just diluting the self-collected ragging signal.
    """
    examples: list[tuple[Path, int, str]] = []

    fight_dir = ubifights_dir / "fight"
    normal_dir = ubifights_dir / "normal"
    for v in sorted(fight_dir.glob("*.mp4")):
        examples.append((v, 1, "ubifights"))
    for v in sorted(normal_dir.glob("*.mp4")):
        examples.append((v, 0, "ubifights"))

    if ucfcrime_abuse_dir is not None:
        for v in sorted(ucfcrime_abuse_dir.glob("*.mp4")):
            examples.append((v, 1, "ucfcrime_abuse"))

    if ucfcrime_normal_dir is not None:
        normal_videos = sorted(ucfcrime_normal_dir.glob("*.mp4"))[:limit_ucfcrime_normal]
        for v in normal_videos:
            examples.append((v, 0, "ucfcrime_normal"))

    if self_ragging_dir is not None:
        for v in sorted(self_ragging_dir.glob("*.mp4")):
            examples.append((v, 1, "self_ragging"))
    if self_normal_dir is not None:
        for v in sorted(self_normal_dir.glob("*.mp4")):
            examples.append((v, 0, "self_normal"))

    return examples


class FeatureDataset(Dataset):
    def __init__(self, items: list[tuple[torch.Tensor, int, str]]) -> None:
        self.items = items
        self.expected_shape = expected_feature_shape(config.PERSONS_PER_FRAME)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        feats, label, _source = self.items[idx]
        if tuple(feats.shape) != self.expected_shape:
            raise ValueError(
                f"expected {self.expected_shape} (Top-K v2), got {tuple(feats.shape)}"
            )
        return feats, torch.tensor(label, dtype=torch.float32)


def extract_all(examples: list[tuple[Path, int, str]], cache_dir: Path, device: str) -> list[tuple[torch.Tensor, int, str]]:
    items = []
    start = time.perf_counter()
    for i, (video_path, label, source) in enumerate(examples, 1):
        try:
            feats = extract_and_cache_features(video_path, cache_dir, add_augmentation=False, device=device)
        except Exception as exc:
            print(f"  [skip] {video_path.name} ({source}): {exc}")
            continue
        items.append((feats, label, source))
        if i % 100 == 0 or i == len(examples):
            elapsed = time.perf_counter() - start
            print(f"  [extract {i}/{len(examples)}] {elapsed:.0f}s elapsed")
    return items


def load_base_model(
    model_path: Path,
    device: str,
    allow_partial_v1: bool = False,
) -> ViolenceClassifier:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    net = ViolenceClassifier(
        input_size=config.INPUT_SIZE,
        hidden_size=config.LSTM_HIDDEN,
        num_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT,
        num_persons=config.PERSONS_PER_FRAME,
        temporal=config.TEMPORAL_HEAD,
    ).to(device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ckpt_k = checkpoint_num_persons(state)
    if ckpt_k < config.PERSONS_PER_FRAME:
        if not allow_partial_v1:
            raise SystemExit(
                f"Base checkpoint is v1 (K={ckpt_k}, no person attention). "
                "Train v2 on RLVS first (extract_features.py + main.py) and pass "
                "that best_model.pt here. Or pass --allow-partial-v1-init to copy "
                "LSTM/temporal-attn/fc and randomly initialize person_attn."
            )
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(
            f"Partial v1→v2 init from {model_path.name}. "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )
        return net
    net.load_state_dict(state)
    return net


@torch.no_grad()
def evaluate(net: ViolenceClassifier, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    net.eval()
    y_true, y_prob = [], []
    for x, y in loader:
        x = x.to(device)
        logits = net(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true.extend(y.numpy().tolist())
        y_prob.extend(probs.tolist())
    return np.array(y_true), np.array(y_prob)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RLVS checkpoint on combined UBI-FIGHTS + UCF-Crime Abuse data.")
    parser.add_argument("--ubifights-dir", type=Path, required=True, help="Dir containing fight/ and normal/ subfolders")
    parser.add_argument("--ucfcrime-abuse-dir", type=Path, default=None, help="Omit to run without UCF-Crime (ablation)")
    parser.add_argument("--ucfcrime-normal-dir", type=Path, default=None, help="Omit to run without UCF-Crime (ablation)")
    parser.add_argument("--limit-ucfcrime-normal", type=int, default=50)
    parser.add_argument("--self-ragging-dir", type=Path, default=None, help="Real staged ragging clips (the actual target domain)")
    parser.add_argument("--self-normal-dir", type=Path, default=None, help="Matched hard-negative clips for the self-collected ragging set")
    parser.add_argument(
        "--base-model",
        type=Path,
        required=True,
        help="v2 RLVS best_model.pt from main.py (Top-K person attention)",
    )
    parser.add_argument(
        "--allow-partial-v1-init",
        action="store_true",
        help="Allow loading a v1 RLVS checkpoint: copy LSTM/fc, randomly init person_attn",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    device = config.DEVICE
    print(f"Device: {device}")

    examples = collect_examples(
        args.ubifights_dir,
        args.ucfcrime_abuse_dir,
        args.ucfcrime_normal_dir,
        args.limit_ucfcrime_normal,
        self_ragging_dir=args.self_ragging_dir,
        self_normal_dir=args.self_normal_dir,
    )
    n_pos = sum(1 for _, l, _ in examples if l == 1)
    n_neg = len(examples) - n_pos
    print(f"Collected {len(examples)} videos: {n_pos} positive (violence/abuse), {n_neg} negative (normal)")
    by_source: dict[str, int] = {}
    for _, _, source in examples:
        by_source[source] = by_source.get(source, 0) + 1
    print(f"By source: {by_source}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Extracting/caching Top-K={config.PERSONS_PER_FRAME} "
        f"PERSON_SELECT={config.PERSON_SELECT} features (YOLO + ResNet18)..."
    )
    items = extract_all(examples, args.cache_dir, device)

    # Stratify by (label, source), not just label -- the self-collected set is much
    # smaller than UBI-FIGHTS/UCF-Crime, so a label-only split could under- or
    # over-represent it in validation purely by chance. This guarantees each
    # source is split ~80/20 the same way, so the held-out set includes a
    # controlled, representative sample of the real ragging clips specifically.
    strata = [f"{label}_{source}" for _, label, source in items]
    train_items, val_items = train_test_split(
        items, test_size=0.2, stratify=strata, random_state=config.RANDOM_SEED
    )
    print(f"Train: {len(train_items)} ({sum(l for _, l, _ in train_items)} positive)")
    print(f"Val:   {len(val_items)} ({sum(l for _, l, _ in val_items)} positive)")
    val_by_source: dict[str, int] = {}
    for _, _, source in val_items:
        val_by_source[source] = val_by_source.get(source, 0) + 1
    print(f"Val by source: {val_by_source}")

    train_pos = sum(l for _, l, _ in train_items)
    train_neg = len(train_items) - train_pos
    pos_weight = torch.tensor([train_neg / max(train_pos, 1)], dtype=torch.float32).to(device)
    print(f"pos_weight = {pos_weight.item():.4f}")

    train_loader = DataLoader(FeatureDataset(train_items), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(FeatureDataset(val_items), batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = load_base_model(
        args.base_model, device, allow_partial_v1=args.allow_partial_v1_init
    )
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        net.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = net(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        net.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = net(x)
                val_losses.append(criterion(logits, y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch: {epoch - epochs_no_improve})")
                break

    net.load_state_dict(best_state)

    y_true, y_prob = evaluate(net, val_loader, device)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) > 0 else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5

    y_pred = (y_prob >= best_threshold).astype(int)
    val_metrics = {
        "threshold": best_threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "confusion_matrix_labels": ["normal", "violence/abuse"],
        "n_samples": int(len(y_true)),
    }
    print()
    print(f"Best threshold (tuned on val precision-recall curve): {best_threshold:.4f}")
    print(f"Val accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"Val precision: {val_metrics['precision']:.4f}")
    print(f"Val recall:    {val_metrics['recall']:.4f}")
    print(f"Val f1:        {val_metrics['f1']:.4f}")
    print(np.array(val_metrics["confusion_matrix"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "best_model_combined2_finetuned.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "input_size": config.INPUT_SIZE,
            "num_frames": config.NUM_FRAMES,
            "num_persons": config.PERSONS_PER_FRAME,
            "temporal": config.TEMPORAL_HEAD,
            "seed": config.RANDOM_SEED,
            "person_select": config.PERSON_SELECT,
            "val_loss": best_val_loss,
        },
        model_path,
    )
    metrics_path = args.output_dir / "metrics_combined2_finetuned.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "device": device,
                "input_size": config.INPUT_SIZE,
                "num_frames": config.NUM_FRAMES,
                "num_persons": config.PERSONS_PER_FRAME,
                "temporal": config.TEMPORAL_HEAD,
                "seed": config.RANDOM_SEED,
                "person_select": config.PERSON_SELECT,
                "best_threshold": best_threshold,
                "best_val_loss": best_val_loss,
                "pos_weight": pos_weight.item(),
                "training_sources": by_source,
                "val_sources": val_by_source,
                "n_train": len(train_items),
                "n_val": len(val_items),
                "history": history,
                "val_at_tuned_threshold": val_metrics,
                "note": (
                    "v2 Top-K person attention, fine-tuned from a v2 RLVS checkpoint "
                    "on UBI-FIGHTS + self-collected staged ragging/normal clips in one pass. "
                    "Split stratified by (label, source)."
                ),
            },
            f,
            indent=2,
        )

    print(f"\nSaved model -> {model_path}")
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
