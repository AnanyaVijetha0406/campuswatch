"""
Split Real Life Violence Situations (or the local sample) into
train / val / test BY VIDEO FILE — never by frame.

Default ratio: 80% / 10% / 10%, stratified per class.
Works the same for:
  - local tiny sample:  --source data/sample
  - full RLVS on Kaggle: --source data/train_raw  (or /kaggle/input/... path)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Allow running as `python scripts/data/split_dataset.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config


def list_videos(class_dir: Path) -> list[Path]:
    if not class_dir.is_dir():
        return []
    files = [
        p
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in config.VIDEO_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def resolve_class_dir(source: Path, class_name: str) -> Path:
    """Accept Violence/NonViolence or violence/nonviolence folder names."""
    exact = source / class_name
    if exact.is_dir():
        return exact
    lower = source / class_name.lower()
    if lower.is_dir():
        return lower
    # RLVS sometimes nests one extra folder
    for child in source.iterdir() if source.is_dir() else []:
        if not child.is_dir():
            continue
        nested = child / class_name
        if nested.is_dir():
            return nested
        nested_l = child / class_name.lower()
        if nested_l.is_dir():
            return nested_l
    return exact


def stratified_split(
    files: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Deterministic 80/10/10 (or custom) split of video paths."""
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return [], [], []

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Give remainder to test so ratios sum to all files
    n_test = n - n_train - n_val

    # Tiny sets (e.g. 5–10 local clips): keep at least 1 in each split when possible
    if n >= 3 and (n_train == 0 or n_val == 0 or n_test == 0):
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n - n_train - n_val)
        # Rebalance if over-allocated
        while n_train + n_val + n_test > n and n_train > 1:
            n_train -= 1
        while n_train + n_val + n_test > n and n_val > 1:
            n_val -= 1

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def place_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        # Relative symlink so the tree is movable within the same machine
        rel = Path(os.path.relpath(src, start=dst.parent))
        dst.symlink_to(rel)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def print_counts(counts: dict[str, dict[str, int]]) -> None:
    print("\n=== Split counts (by video file) ===")
    header = f"{'split':<8} {'Violence':>10} {'NonViolence':>12} {'total':>8}"
    print(header)
    print("-" * len(header))
    for split in ("train", "val", "test"):
        v = counts[split].get(config.VIOLENCE_CLASS, 0)
        nv = counts[split].get(config.NONVIOLENCE_CLASS, 0)
        print(f"{split:<8} {v:>10} {nv:>12} {v + nv:>8}")
    print("-" * len(header))
    tv = sum(counts[s].get(config.VIOLENCE_CLASS, 0) for s in counts)
    tnv = sum(counts[s].get(config.NONVIOLENCE_CLASS, 0) for s in counts)
    print(f"{'ALL':<8} {tv:>10} {tnv:>12} {tv + tnv:>8}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split RLVS (or local sample) into train/val/test by video file."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=config.SAMPLE_DIR,
        help=(
            "Root with Violence/ and NonViolence/ folders. "
            f"Default: {config.SAMPLE_DIR} (local sanity sample). "
            f"Full dataset: {config.TRAIN_RAW_DIR} or a Kaggle /kaggle/input/... path."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.SPLITS_DIR,
        help=f"Where to write train/val/test trees. Default: {config.SPLITS_DIR}",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    # test_ratio is implied: 1 - train - val (default 0.1)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink", "manifest"),
        default="copy",
        help=(
            "copy: duplicate files into output (good for tiny local sample). "
            "symlink: link into output (saves disk on Kaggle/Linux). "
            "manifest: only write splits_manifest.json (no file copies)."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing train/val/test under --output before writing.",
    )
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < -1e-9:
        raise SystemExit("train_ratio + val_ratio must be <= 1.0")

    source = args.source.resolve()
    output = args.output.resolve()

    if not source.is_dir():
        raise SystemExit(
            f"Source not found: {source}\n"
            "Place videos under <source>/Violence and <source>/NonViolence, "
            "or pass --source to the correct folder."
        )

    if args.clean and output.exists() and args.mode != "manifest":
        for split in ("train", "val", "test"):
            split_dir = output / split
            if split_dir.exists():
                shutil.rmtree(split_dir)

    output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict[str, list[str]]] = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }
    counts: dict[str, dict[str, int]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    print(f"Source : {source}")
    print(f"Output : {output}")
    print(f"Mode   : {args.mode}")
    print(
        f"Ratios : train={args.train_ratio:.0%}  "
        f"val={args.val_ratio:.0%}  test={test_ratio:.0%}  seed={args.seed}"
    )

    for class_name in config.CLASS_NAMES:
        class_dir = resolve_class_dir(source, class_name)
        videos = list_videos(class_dir)
        print(f"\n{class_name}: found {len(videos)} videos in {class_dir}")

        # Per-class seed offset so both classes reshuffle independently but reproducibly
        class_seed = args.seed + (0 if class_name == config.VIOLENCE_CLASS else 1)
        train, val, test = stratified_split(
            videos, args.train_ratio, args.val_ratio, class_seed
        )

        for split_name, split_files in (
            ("train", train),
            ("val", val),
            ("test", test),
        ):
            counts[split_name][class_name] = len(split_files)
            for src in split_files:
                rel_key = f"{class_name}/{src.name}"
                manifest[split_name][class_name].append(str(src.resolve()))
                if args.mode != "manifest":
                    dst = output / split_name / class_name / src.name
                    place_file(src, dst, args.mode)

    # Serialize manifest (always written — useful on Kaggle + for audits)
    manifest_out = {
        split: {cls: paths for cls, paths in classes.items()}
        for split, classes in manifest.items()
    }
    manifest_path = output / "splits_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source": str(source),
                "seed": args.seed,
                "ratios": {
                    "train": args.train_ratio,
                    "val": args.val_ratio,
                    "test": round(test_ratio, 4),
                },
                "mode": args.mode,
                "splits": manifest_out,
                "counts": counts,
            },
            f,
            indent=2,
        )
    print(f"Wrote manifest: {manifest_path}")

    print_counts(counts)

    if sum(sum(c.values()) for c in counts.values()) == 0:
        print(
            "No videos found. Add clips under Violence/ and NonViolence/, then re-run.\n"
            "Local sanity sample target: ~5-10 videos per class under data/sample/."
        )


if __name__ == "__main__":
    main()
