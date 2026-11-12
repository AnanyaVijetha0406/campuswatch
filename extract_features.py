"""
Extract and cache YOLO+ResNet18+bbox features for a split tree.

Usage (local sample sanity check):
  python extract_features.py --split-dir data/splits/train --cache-dir data/feature_cache/train --augment
  python extract_features.py --split-dir data/splits/val   --cache-dir data/feature_cache/val
  python extract_features.py --split-dir data/splits/test  --cache-dir data/feature_cache/test

On Kaggle, point --split-dir / --cache-dir at /kaggle/working/... paths
(or update config.SPLITS_DIR / config.CACHE_DIR).

Augmentation MUST be True only for the training split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from model import FEATURE_DIM, cache_split_tree, extract_and_cache_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache video features for training.")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Directory with Violence/ and NonViolence/ videos (e.g. data/splits/train).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where to write .pt feature tensors.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply train-time crop augmentations (ONLY for train split).",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild existing caches.")
    parser.add_argument(
        "--sanity-one-each",
        action="store_true",
        help="Only process one video per class and print shapes (Phase 1 check).",
    )
    args = parser.parse_args()

    print(
        f"DEVICE={config.DEVICE}  NUM_FRAMES={config.NUM_FRAMES}  "
        f"K={config.PERSONS_PER_FRAME}  PERSON_SELECT={config.PERSON_SELECT}  "
        f"FEATURE_DIM={FEATURE_DIM}"
    )
    assert FEATURE_DIM + (512 if config.SCENE_CONTEXT else 0) == config.INPUT_SIZE, (
        f"model.FEATURE_DIM={FEATURE_DIM} != config.INPUT_SIZE={config.INPUT_SIZE}"
    )

    if args.sanity_one_each:
        # Prefer already-split sample; fall back to data/sample
        source = args.split_dir or config.SPLITS_DIR / "train"
        if not source.is_dir() or not any(source.iterdir()):
            source = config.SAMPLE_DIR
        print(f"Sanity source: {source}")
        for class_name in config.CLASS_NAMES:
            class_dir = source / class_name
            if not class_dir.is_dir():
                class_dir = source / class_name.lower()
            videos = sorted(
                p
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in config.VIDEO_EXTENSIONS
            ) if class_dir.is_dir() else []
            if not videos:
                print(f"{class_name}: NO VIDEOS FOUND under {class_dir}")
                continue
            video = videos[0]
            cache_dir = args.cache_dir or (config.CACHE_DIR / "sanity")
            feats = extract_and_cache_features(
                video,
                cache_dir / class_name,
                add_augmentation=False,
                force=args.force,
            )
            print(f"{class_name}: {video.name}  shape={tuple(feats.shape)}  dtype={feats.dtype}")
        return

    split_dir = args.split_dir
    cache_dir = args.cache_dir
    if split_dir is None or cache_dir is None:
        raise SystemExit("Provide --split-dir and --cache-dir (or use --sanity-one-each).")

    results = cache_split_tree(
        split_dir.resolve(),
        cache_dir.resolve(),
        add_augmentation=args.augment,
        force=args.force,
    )
    print(f"\nCached {len(results)} videos under {cache_dir.resolve()}")
    shapes = {r[2] for r in results}
    print(f"Unique shapes: {shapes}")


if __name__ == "__main__":
    main()
