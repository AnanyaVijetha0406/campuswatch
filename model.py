"""
Violence detection feature pipeline + BiLSTM+Attention classifier.

Deployed default is K=1: 16 frames -> YOLOv8n -> largest-area person crop
-> frozen ResNet18 (512-d) + normalized bbox (4-d) -> 516-d per frame
-> BiLSTM + temporal attention -> raw logit (BCEWithLogitsLoss). See
config.PERSONS_PER_FRAME / config.MODEL_PATH for the deployed checkpoint.

This module also supports an unreleased v2 Top-K path (K>1): keep the K
people whose box centers are nearest each other (closest-group, the
interaction cluster) instead of just the largest boxes, then pool them with
a learned PersonAttention layer before the same BiLSTM+attention head.
K=3 closest-group was evaluated but not deployed -- same real-clip recall
as K=1 with ~2x the false-positive rate (see results/k3/). K=1 always uses
largest-area regardless of PERSON_SELECT, so v1 checkpoints load unchanged.

Person attention is the only new trainable module for K>1; YOLO and
ResNet18 stay frozen either way.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

import config

# ImageNet normalization for ResNet18
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

CNN_FEAT_DIM = 512
# Seek to each sampled frame (instead of decoding through) only for clips longer
# than this many frames AND larger than this many pixels per frame.
SEQUENTIAL_READ_MAX_FRAMES = 450
SEQUENTIAL_READ_MAX_PIXELS = 640 * 480
BBOX_DIM = 4  # x1, y1, x2, y2 normalized to [0, 1]
FEATURE_DIM = CNN_FEAT_DIM + BBOX_DIM  # 516 — must match config.INPUT_SIZE


def _eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def _train_augment_transform() -> transforms.Compose:
    """Horizontal flip + mild color jitter — training crops only."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


class ResNet18Encoder(nn.Module):
    """Frozen ImageNet ResNet18 → 512-d embedding (no classifier head)."""

    def __init__(self) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # (B, 512, 1, 1)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        out = self.features(x)
        return out.flatten(1)  # (B, 512)


class Attention(nn.Module):
    """Simple temporal attention over BiLSTM outputs."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (B, T, H)
        weights = torch.softmax(self.score(lstm_out).squeeze(-1), dim=1)  # (B, T)
        return torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)  # (B, H)


class PersonAttention(nn.Module):
    """Learned attention pooling over K person vectors in a single frame.

    Zero-padded slots (no detection) are masked out. If a frame has no
    valid persons, the pooled vector is zeros.
    """

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(feat_dim, 1)

    def forward(self, persons: torch.Tensor) -> torch.Tensor:
        # persons: (B, T, K, D)
        valid = persons.abs().sum(dim=-1) > 0  # (B, T, K)
        logits = self.score(persons).squeeze(-1)  # (B, T, K)
        neg_large = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~valid, neg_large)
        all_invalid = ~valid.any(dim=-1)  # (B, T)
        weights = torch.softmax(logits, dim=-1)
        weights = weights.masked_fill(all_invalid.unsqueeze(-1), 0.0)
        return (persons * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, D)


class TemporalTransformer(nn.Module):
    """Transformer-encoder alternative to the BiLSTM (reviewer baseline).

    Projects each 516-d frame feature to d_model, adds a learned positional
    embedding, and runs a small pre-norm Transformer encoder. Output width is
    d_model so the same temporal attention + FC head follows.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, T, d_model)
        h = self.proj(x) + self.pos[:, : x.size(1)]
        return self.encoder(h)


class ViolenceClassifier(nn.Module):
    """BiLSTM + temporal attention binary classifier. Outputs raw logits.

    v2 (num_persons > 1): input is (B, T, K, D); person attention pools K
    people into one 516-d vector per frame, then the v1 temporal head runs.
    v1 (num_persons == 1): input is (B, T, D), same as the deployed checkpoint.
    """

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_persons: int = 1,
        temporal: str = "bilstm",
    ) -> None:
        super().__init__()
        self.num_persons = int(num_persons)
        self.temporal = temporal
        self.person_attn = (
            PersonAttention(input_size) if self.num_persons > 1 else None
        )
        if temporal == "transformer":
            self.temporal_enc = TemporalTransformer(input_size)
            out_dim = self.temporal_enc.out_dim
        elif temporal == "bilstm":
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.0 if num_layers == 1 else dropout,
            )
            out_dim = hidden_size * 2
        else:
            raise ValueError(f"Unknown temporal head: {temporal!r}")
        self.attn = Attention(out_dim)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) for v1, or (B, T, K, D) for v2
        if x.dim() == 4:
            if self.person_attn is not None:
                x = self.person_attn(x)
            else:
                # v1 head given Top-K features: keep largest person (slot 0)
                x = x[:, :, 0, :]
        if self.temporal == "transformer":
            seq_out = self.temporal_enc(x)
        else:
            seq_out, _ = self.lstm(x)
        context = self.attn(seq_out)
        return self.fc(context).squeeze(-1)  # (B,) logits


def expected_feature_shape(num_persons: int | None = None) -> tuple[int, ...]:
    k = int(num_persons if num_persons is not None else config.PERSONS_PER_FRAME)
    if k <= 1:
        return (config.NUM_FRAMES, config.INPUT_SIZE)
    return (config.NUM_FRAMES, k, FEATURE_DIM)


def checkpoint_num_persons(state: dict) -> int:
    """Infer K from a state dict: v2 has person_attn.*, v1 does not."""
    if any(key.startswith("person_attn") for key in state):
        return int(config.PERSONS_PER_FRAME)
    return 1


def load_classifier(
    model_path: str | Path,
    device: str | None = None,
    num_persons: int | None = None,
) -> ViolenceClassifier:
    """Build ViolenceClassifier matching a checkpoint (v1 or v2)."""
    device = device or config.DEVICE
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    if num_persons is None:
        if isinstance(ckpt, dict) and "num_persons" in ckpt:
            num_persons = int(ckpt["num_persons"])
        else:
            num_persons = checkpoint_num_persons(state)
    temporal = ckpt.get("temporal", "bilstm") if isinstance(ckpt, dict) else "bilstm"
    if any(key.startswith("temporal_enc") for key in state):
        temporal = "transformer"
    input_size = int(ckpt.get("input_size", config.INPUT_SIZE)) if isinstance(ckpt, dict) else config.INPUT_SIZE
    net = ViolenceClassifier(
        input_size=input_size,
        hidden_size=config.LSTM_HIDDEN,
        num_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT,
        num_persons=num_persons,
        temporal=temporal,
    ).to(device)
    net.load_state_dict(state)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Detection / feature extraction helpers
# ---------------------------------------------------------------------------

_yolo_model: YOLO | None = None
_cnn_encoder: ResNet18Encoder | None = None


def get_yolo(weights: str = config.YOLO_WEIGHTS) -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(weights)
    return _yolo_model


def get_cnn_encoder(device: str | None = None) -> ResNet18Encoder:
    global _cnn_encoder
    if _cnn_encoder is None:
        device = device or config.DEVICE
        _cnn_encoder = ResNet18Encoder().to(device)
    return _cnn_encoder


def _clip_bbox(
    xyxy: np.ndarray, frame_w: int, frame_h: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = xyxy.tolist()
    x1 = int(np.clip(x1, 0, frame_w - 1))
    x2 = int(np.clip(x2, 0, frame_w - 1))
    y1 = int(np.clip(y1, 0, frame_h - 1))
    y2 = int(np.clip(y2, 0, frame_h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def person_select_tag() -> str:
    """Short token for cache filenames so area and closest-group caches cannot mix."""
    method = getattr(config, "PERSON_SELECT", "closest_group")
    tag = {"closest_group": "cg", "area": "area"}.get(method, str(method)[:8])
    return tag + ("_scene" if getattr(config, "SCENE_CONTEXT", False) else "")


def _bbox_center(bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)


def _pairwise_distance_sum(centers: np.ndarray) -> float:
    cost = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            cost += float(np.linalg.norm(centers[i] - centers[j]))
    return cost


def _select_topk_bboxes(
    bboxes: list[tuple[int, int, int, int]], k: int
) -> list[tuple[int, int, int, int]]:
    """Keep up to K people. K=1 = largest area. K>1 uses config.PERSON_SELECT."""
    if not bboxes:
        return []
    if k <= 1:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        return [bboxes[int(np.argmax(areas))]]
    if len(bboxes) <= k:
        return bboxes

    method = getattr(config, "PERSON_SELECT", "closest_group")
    if method == "area":
        areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes])
        order = np.argsort(areas)[::-1][:k]
        return [bboxes[int(i)] for i in order]

    # closest_group: K boxes with minimum sum of pairwise center distances.
    # Tie-break: larger total area (prefer the nearer/more prominent cluster).
    centers = np.stack([_bbox_center(b) for b in bboxes], axis=0)
    areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes], dtype=np.float64)
    best_combo: tuple[int, ...] | None = None
    best_cost = float("inf")
    best_area = -1.0
    for combo in combinations(range(len(bboxes)), k):
        cost = _pairwise_distance_sum(centers[list(combo)])
        area = float(areas[list(combo)].sum())
        if cost < best_cost - 1e-9 or (abs(cost - best_cost) < 1e-9 and area > best_area):
            best_cost = cost
            best_area = area
            best_combo = combo
    assert best_combo is not None
    selected = [bboxes[i] for i in best_combo]
    selected.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return selected


def get_bboxes(
    frame_bgr: np.ndarray,
    k: int | None = None,
    yolo: YOLO | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    Return up to K person bboxes (xyxy, pixel coords).

    Default v2 rule (PERSON_SELECT=closest_group): keep the K people whose
    box centers are nearest each other. Extra detections are dropped.
    This is not identity tracking: the K people in frame t are not guaranteed
    to be the same K in frame t+1.
    """
    k = int(k if k is not None else config.PERSONS_PER_FRAME)
    yolo = yolo or get_yolo()
    results = yolo.predict(frame_bgr, verbose=False, classes=[0])  # person only
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    h, w = frame_bgr.shape[:2]
    clipped: list[tuple[int, int, int, int]] = []
    for row in xyxy:
        box = _clip_bbox(row, w, h)
        if box is not None:
            clipped.append(box)
    return _select_topk_bboxes(clipped, k)


def get_bbox(
    frame_bgr: np.ndarray, yolo: YOLO | None = None
) -> tuple[int, int, int, int] | None:
    """Largest person bbox only (v1 helper). Prefer get_bboxes() for v2."""
    boxes = get_bboxes(frame_bgr, k=1, yolo=yolo)
    return boxes[0] if boxes else None


def normalize_bbox(
    bbox: tuple[int, int, int, int], frame_w: int, frame_h: int
) -> np.ndarray:
    """Normalize xyxy to [0, 1] by frame width/height before concat with CNN feats."""
    x1, y1, x2, y2 = bbox
    return np.array(
        [x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h],
        dtype=np.float32,
    )


def sample_frame_indices(total_frames: int, num_frames: int = config.NUM_FRAMES) -> np.ndarray:
    """
    Cover the FULL video span (including the tail).

    Uses np.linspace instead of step = total // num_frames, which truncated
    longer videos and skipped the ending.
    """
    if total_frames <= 0:
        return np.zeros(num_frames, dtype=np.int64)
    if total_frames == 1:
        return np.zeros(num_frames, dtype=np.int64)
    idx = np.linspace(0, total_frames - 1, num_frames)
    return np.round(idx).astype(np.int64)


def _zero_persons(k: int) -> np.ndarray:
    return np.zeros((k, FEATURE_DIM), dtype=np.float32)


def encode_frame_persons(
    frame_bgr: np.ndarray,
    yolo: YOLO,
    cnn: ResNet18Encoder,
    tfm: transforms.Compose,
    device: str,
    k: int | None = None,
) -> np.ndarray:
    """YOLO Top-K → ResNet18 + bbox. Returns (K, FEATURE_DIM), zero-padded."""
    k = int(k if k is not None else config.PERSONS_PER_FRAME)
    h, w = frame_bgr.shape[:2]
    bboxes = get_bboxes(frame_bgr, k=k, yolo=yolo)
    out = _zero_persons(k)
    if not bboxes:
        return out

    crop_tensors: list[torch.Tensor] = []
    bbox_vecs: list[np.ndarray] = []
    slots: list[int] = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_tensors.append(tfm(crop_rgb))
        bbox_vecs.append(normalize_bbox(bbox, w, h))
        slots.append(i)

    if not crop_tensors:
        return out

    batch = torch.stack(crop_tensors, dim=0).to(device)
    with torch.no_grad():
        cnn_vecs = cnn(batch).cpu().numpy().astype(np.float32)
    for j, slot in enumerate(slots):
        out[slot] = np.concatenate([cnn_vecs[j], bbox_vecs[j]], axis=0)
    return out


def encode_frames_persons(
    frames_bgr: list[np.ndarray],
    yolo: YOLO,
    cnn: ResNet18Encoder,
    tfm: transforms.Compose,
    device: str,
    k: int | None = None,
) -> list[np.ndarray]:
    """Batched encode_frame_persons: one YOLO call over all frames and one
    ResNet18 call over all crops. Same per-frame output, far less overhead."""
    k = int(k if k is not None else config.PERSONS_PER_FRAME)
    outs = [_zero_persons(k) for _ in frames_bgr]
    if not frames_bgr:
        return outs

    results = yolo.predict(frames_bgr, verbose=False, classes=[0], device=device)

    crop_tensors: list[torch.Tensor] = []
    bbox_vecs: list[np.ndarray] = []
    owners: list[tuple[int, int]] = []  # (frame index, person slot)
    for fi, (frame, res) in enumerate(zip(frames_bgr, results)):
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        h, w = frame.shape[:2]
        clipped = []
        for row in boxes.xyxy.cpu().numpy():
            box = _clip_bbox(row, w, h)
            if box is not None:
                clipped.append(box)
        for slot, bbox in enumerate(_select_topk_bboxes(clipped, k)):
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_tensors.append(tfm(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            bbox_vecs.append(normalize_bbox(bbox, w, h))
            owners.append((fi, slot))

    if not crop_tensors:
        return outs

    batch = torch.stack(crop_tensors, dim=0).to(device)
    with torch.no_grad():
        cnn_vecs = cnn(batch).cpu().numpy().astype(np.float32)
    for j, (fi, slot) in enumerate(owners):
        outs[fi][slot] = np.concatenate([cnn_vecs[j], bbox_vecs[j]], axis=0)
    return outs


def read_sampled_frames(cap: cv2.VideoCapture, indices: np.ndarray) -> list[np.ndarray | None]:
    """Decode the requested frame indices in one forward pass.

    Seeking with CAP_PROP_POS_FRAMES re-decodes from the nearest keyframe for
    every sample; grab() through the stream is steadier. Falls back to seeking
    when the container does not report a frame count."""
    wanted = {int(i) for i in indices}
    found: dict[int, np.ndarray | None] = {}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    # Long, high-resolution clips (e.g. phone recordings at 1080p/4K) are cheaper
    # to seek through, since they have frequent keyframes, than to decode frame by
    # frame. Small clips are fastest and steadiest read sequentially.
    pixels = (cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) * (cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if total <= 0 or (total > SEQUENTIAL_READ_MAX_FRAMES and pixels > SEQUENTIAL_READ_MAX_PIXELS):
        for i in sorted(wanted):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            found[i] = frame if ok else None
    else:
        last = max(wanted)
        for pos in range(last + 1):
            if not cap.grab():
                break
            if pos in wanted:
                ok, frame = cap.retrieve()
                found[pos] = frame if ok else None
    return [found.get(int(i)) for i in indices]


def count_zero_frames(feats: torch.Tensor) -> int:
    """Count frames where no person was encoded (all K slots zero)."""
    if feats.dim() == 2:
        return int((feats.abs().sum(dim=1) == 0).sum().item())
    return int((feats.abs().sum(dim=(1, 2)) == 0).sum().item())


def extract_video_features(
    video_path: str | Path,
    add_augmentation: bool = False,
    device: str | None = None,
    yolo: YOLO | None = None,
    cnn: ResNet18Encoder | None = None,
    num_persons: int | None = None,
) -> torch.Tensor:
    """
    Run YOLO → Top-K crops → ResNet18 → concat normalized bbox
    for NUM_FRAMES uniformly sampled frames.

    Returns (NUM_FRAMES, K, FEATURE_DIM) when K > 1, else (NUM_FRAMES, FEATURE_DIM).
    Missing persons / failed frames are zero vectors so the gap stays explicit.
    """
    device = device or config.DEVICE
    k = int(num_persons if num_persons is not None else config.PERSONS_PER_FRAME)
    yolo = yolo or get_yolo()
    cnn = cnn or get_cnn_encoder(device)
    tfm = _train_augment_transform() if add_augmentation else _eval_transform()

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    indices = sample_frame_indices(total, config.NUM_FRAMES)
    feats: list[np.ndarray] = []

    frames = read_sampled_frames(cap, indices)
    cap.release()

    good = [i for i, f in enumerate(frames) if f is not None]
    encoded = encode_frames_persons([frames[i] for i in good], yolo, cnn, tfm, device, k=k)
    by_pos = dict(zip(good, encoded))
    for i in range(len(frames)):
        person_feats = by_pos.get(i)
        if person_feats is None:
            feats.append(_zero_persons(k) if k > 1 else np.zeros(FEATURE_DIM, dtype=np.float32))
        else:
            feats.append(person_feats if k > 1 else person_feats[0])

    arr = np.stack(feats, axis=0)
    if getattr(config, "SCENE_CONTEXT", False) and k <= 1:
        # Whole-frame ResNet18 embedding per sampled frame (zeros if unreadable).
        scene = np.zeros((len(frames), CNN_FEAT_DIM), dtype=np.float32)
        if good:
            batch = torch.stack(
                [_eval_transform()(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)) for i in good], dim=0
            ).to(device)
            with torch.no_grad():
                vecs = cnn(batch).cpu().numpy().astype(np.float32)
            for j, i in enumerate(good):
                scene[i] = vecs[j]
        arr = np.concatenate([arr, scene], axis=1)
    return torch.from_numpy(arr)


def cache_key_for_video(
    video_path: Path, num_persons: int | None = None
) -> str:
    """Stable cache filename keyed by video filename, K, and a short path hash."""
    k = int(num_persons if num_persons is not None else config.PERSONS_PER_FRAME)
    stem = video_path.stem
    digest = hashlib.md5(str(video_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{stem}__k{k}__{person_select_tag()}__{digest}.pt"


def extract_and_cache_features(
    video_path: str | Path,
    cache_dir: str | Path,
    add_augmentation: bool = False,
    force: bool = False,
    device: str | None = None,
    num_persons: int | None = None,
) -> torch.Tensor:
    """
    Run YOLO+Top-K+ResNet18+bbox-normalize ONCE per video and save to cache_dir.
    Skip if a cache with the expected v2 shape already exists.

    Augmentation is applied only when add_augmentation=True (training split).
    Val / test / ShanghaiTech must pass add_augmentation=False.
    """
    k = int(num_persons if num_persons is not None else config.PERSONS_PER_FRAME)
    video_path = Path(video_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_key_for_video(video_path, num_persons=k)
    expected = expected_feature_shape(k)

    if cache_path.exists() and not force:
        feats = torch.load(cache_path, map_location="cpu", weights_only=True)
        if tuple(feats.shape) == expected:
            return feats

    feats = extract_video_features(
        video_path,
        add_augmentation=add_augmentation,
        device=device,
        num_persons=k,
    )
    torch.save(feats, cache_path)
    return feats


# ---------------------------------------------------------------------------
# Convenience: batch-cache a split directory tree
# ---------------------------------------------------------------------------

def cache_split_tree(
    split_root: Path,
    cache_root: Path,
    add_augmentation: bool = False,
    force: bool = False,
    num_persons: int | None = None,
) -> list[tuple[str, str, tuple[int, ...]]]:
    """
    Walk split_root/{Violence,NonViolence}/*.mp4 and cache features under
    cache_root/<class>/. Returns list of (class, video_name, shape).
    """
    k = int(num_persons if num_persons is not None else config.PERSONS_PER_FRAME)
    results = []
    yolo = get_yolo()
    cnn = get_cnn_encoder()
    for class_name in config.CLASS_NAMES:
        class_dir = split_root / class_name
        if not class_dir.is_dir():
            class_dir = split_root / class_name.lower()
        if not class_dir.is_dir():
            continue
        out_dir = cache_root / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for video in sorted(class_dir.iterdir()):
            if video.suffix.lower() not in config.VIDEO_EXTENSIONS:
                continue
            cache_path = out_dir / f"{video.stem}__k{k}__{person_select_tag()}.pt"
            if cache_path.exists() and not force:
                feats = torch.load(cache_path, map_location="cpu", weights_only=True)
                if tuple(feats.shape) != expected_feature_shape(k):
                    feats = extract_video_features(
                        video,
                        add_augmentation=add_augmentation,
                        yolo=yolo,
                        cnn=cnn,
                        num_persons=k,
                    )
                    torch.save(feats, cache_path)
            else:
                feats = extract_video_features(
                    video,
                    add_augmentation=add_augmentation,
                    yolo=yolo,
                    cnn=cnn,
                    num_persons=k,
                )
                torch.save(feats, cache_path)
            results.append((class_name, video.name, tuple(feats.shape)))
            print(f"[cache] {class_name}/{video.name} -> {tuple(feats.shape)}")
    return results
