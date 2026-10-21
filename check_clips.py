"""
Sanity-check that every self-collected clip in a folder actually opens in
OpenCV and reports sane properties (frame count, resolution, fps) --
catches corrupted/unreadable files before they silently produce
zero-vector features downstream in extract_video_features().
"""
import cv2
import glob
import os
import sys

folder = r"data/ragging_clips"
files = sorted(glob.glob(os.path.join(folder, "Screen*.mp4")))
print(f"Found {len(files)} files", flush=True)

for f in files:
    print(f"Checking: {os.path.basename(f)} ...", flush=True)
    cap = cv2.VideoCapture(f)
    ok = cap.isOpened()
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if ok else 0
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) if ok else 0
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if ok else 0
    fps = cap.get(cv2.CAP_PROP_FPS) if ok else 0
    cap.release()
    print(f"  -> opened={ok} frames={frames:.0f} {w:.0f}x{h:.0f} fps={fps:.1f}", flush=True)
