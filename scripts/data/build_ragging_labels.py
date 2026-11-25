"""
Build labels.csv for the self-collected staged ragging clip set — real
ragging-specific data (unlike RLVS/UBI-FIGHTS/UCF-Crime, which are all
violence-adjacent proxies), used to close the gap those datasets couldn't:
non-fight-shaped coercion and humiliation (forced tasks, grabbing, mild
contact) that a model trained only on overt fights doesn't recognize.

Usage:
  python scripts/data/build_ragging_labels.py \
    --ragging-dir "path\\to\\ragging_clips" \
    --normal-dir "path\\to\\normal_clips" \
    --out ragging_labels.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labels.csv for the self-collected ragging clip set.")
    parser.add_argument("--ragging-dir", type=Path, required=True)
    parser.add_argument("--normal-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("ragging_labels.csv"))
    args = parser.parse_args()

    rows: list[tuple[str, int, str, str]] = []  # filepath, label, source, notes

    for v in sorted(args.ragging_dir.glob("*.mp4")):
        rows.append((str(v), 1, "self_ragging", ""))

    for v in sorted(args.normal_dir.glob("*.mp4")):
        rows.append((str(v), 0, "self_normal", ""))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "source", "notes"])
        writer.writerows(rows)

    n_pos = sum(1 for _, label, _, _ in rows if label == 1)
    n_neg = len(rows) - n_pos
    print(f"Wrote {len(rows)} rows to {args.out}  (ragging={n_pos}, normal={n_neg})")


if __name__ == "__main__":
    main()
