"""
Flask dashboard — single-video inference using Kaggle-trained weights
and the tuned threshold from metrics.json (not hardcoded 0.5).

Deployment settings come from environment variables (see README):
  APP_USERNAME, APP_PASSWORD, SECRET_KEY, HOST, PORT,
  MAX_UPLOAD_MB, MAX_VIDEO_SECONDS, TORCH_THREADS
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import sys
import threading
import time
import uuid
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from model import extract_video_features, get_cnn_encoder, get_yolo, load_classifier

# Demo defaults so a local run works out of the box. Override both with
# environment variables (Space secrets / docker -e) for any shared deployment.
LOGIN_USER = os.environ.get("APP_USERNAME", "admin")
LOGIN_PASS = os.environ.get("APP_PASSWORD", "admin123")

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
MAX_VIDEO_SECONDS = float(os.environ.get("MAX_VIDEO_SECONDS", "60"))
UPLOAD_TTL_SECONDS = 15 * 60

# Oversubscribed CPU threads made inference time swing from ~4 s to ~50 s.
_threads = int(os.environ.get("TORCH_THREADS", str(min(4, os.cpu_count() or 1))))
torch.set_num_threads(_threads)
cv2.setNumThreads(1)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(16)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_DIR = config.PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_model = None
_threshold: float = float(config.THRESHOLD)
_device: str = config.DEVICE
_accuracy_display: str = "N/A"
_accuracy_label: str = "VALIDATION ACC."
_infer_lock = threading.Lock()  # models are shared globals; run one inference at a time


def load_metrics() -> dict:
    path = config.METRICS_PATH
    if not path.is_file():
        print(f"Warning: {path} not found — using config.THRESHOLD={config.THRESHOLD}")
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_threshold(metrics: dict) -> float:
    thr = float(metrics.get("best_threshold", config.THRESHOLD))
    print(f"Using threshold={thr:.4f}")
    return thr


def load_accuracy_display(metrics: dict) -> tuple[str, str]:
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
    _model = load_classifier(path, device=_device)
    print(f"Loaded model K={_model.num_persons} from {path} on {_device}")
    return _model


def warm_up() -> None:
    """Load YOLO/ResNet and run each once so the first real request is not slow."""
    model = load_model()
    get_yolo().predict(np.zeros((240, 320, 3), dtype=np.uint8), verbose=False)
    with torch.no_grad():
        get_cnn_encoder(_device)(torch.zeros(1, 3, 224, 224, device=_device))
        model(torch.zeros(1, config.NUM_FRAMES, config.INPUT_SIZE, device=_device))


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
            device=_device,
            num_persons=model.num_persons,
        )
        x = feats.unsqueeze(0).to(_device)
        logit = model(x).squeeze().item()
    prob = float(torch.sigmoid(torch.tensor(logit)).item())
    is_violence = prob >= _threshold
    confidence = prob if is_violence else (1.0 - prob)
    return {
        "is_violence": is_violence,
        "verdict": "Violence / ragging detected" if is_violence else "No violence detected",
        "probability": prob,
        "confidence": confidence,
        "threshold": _threshold,
    }


def _remove_upload(name: str | None) -> None:
    if not name:
        return
    try:
        (UPLOAD_DIR / name).unlink(missing_ok=True)
    except OSError:
        pass


def _purge_stale_uploads() -> None:
    cutoff = time.time() - UPLOAD_TTL_SECONDS
    for p in UPLOAD_DIR.iterdir():
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


@app.route("/health")
def health():
    return jsonify(status="ok", model_loaded=_model is not None)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        user = (request.form.get("username") or "").strip()
        pwd = request.form.get("password") or ""
        ok_user = hmac.compare_digest(user.encode(), LOGIN_USER.encode())
        ok_pass = hmac.compare_digest(pwd.encode(), LOGIN_PASS.encode())
        if ok_user and ok_pass:
            session["logged_in"] = True
            return redirect(url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    _remove_upload(session.get("last_upload"))
    session.clear()
    return redirect(url_for("login"))


@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename: str):
    """Serve only the clip this session uploaded, for the preview player."""
    if filename != session.get("last_upload"):
        abort(404)
    return send_from_directory(UPLOAD_DIR, filename)


@app.errorhandler(413)
def too_large(_exc):
    return (
        render_template(
            "index.html",
            **_page_context(error=f"File is too large. Limit is {MAX_UPLOAD_MB} MB."),
        ),
        413,
    )


def _page_context(**overrides) -> dict:
    ctx = dict(
        result=None,
        confidence=None,
        video_file=None,
        error=None,
        threshold=f"{_threshold:.4f}",
        frames=config.NUM_FRAMES,
        accuracy=_accuracy_display,
        accuracy_label=_accuracy_label,
        processing_time="—",
        max_upload_mb=MAX_UPLOAD_MB,
        max_seconds=int(MAX_VIDEO_SECONDS),
    )
    ctx.update(overrides)
    return ctx


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    ctx = _page_context()

    if request.method == "POST":
        file = request.files.get("video")
        if file is None or not file.filename:
            ctx["error"] = "Please choose a video file."
        else:
            ext = Path(secure_filename(file.filename)).suffix.lower()
            if ext not in ALLOWED_EXT:
                ctx["error"] = (
                    f"Unsupported format '{ext}'. Use: {', '.join(sorted(ALLOWED_EXT))}"
                )
            else:
                _purge_stale_uploads()
                _remove_upload(session.pop("last_upload", None))

                name = f"{uuid.uuid4().hex}{ext}"
                save_path = UPLOAD_DIR / name
                try:
                    file.save(save_path)
                    check_video(save_path)
                    start = time.perf_counter()
                    pred = predict_video(save_path)
                    ctx["processing_time"] = f"~{time.perf_counter() - start:.1f}s"
                    ctx["result"] = pred["verdict"]
                    ctx["confidence"] = f"{pred['confidence'] * 100:.1f}%"
                    ctx["video_file"] = name
                    session["last_upload"] = name
                except ValueError as exc:
                    ctx["error"] = str(exc)
                    _remove_upload(name)
                except Exception as exc:
                    ctx["error"] = f"Inference failed: {exc}"
                    _remove_upload(name)

    return render_template("index.html", **ctx)


def init_app() -> None:
    """Load metrics, models and warm up. Runs at import so gunicorn gets it too."""
    global _threshold, _accuracy_label, _accuracy_display
    metrics = load_metrics()
    _threshold = load_threshold(metrics)
    _accuracy_label, _accuracy_display = load_accuracy_display(metrics)
    t0 = time.perf_counter()
    warm_up()  # fail fast if weights are missing
    print(f"Warm-up done in {time.perf_counter() - t0:.1f}s (torch threads={_threads})")


init_app()


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print(f"Dashboard ready — http://{host}:{port}  (user={LOGIN_USER})")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
