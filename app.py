"""
Flask dashboard — single-video inference using Kaggle-trained weights
and the tuned threshold from metrics.json (not hardcoded 0.5).

Deployment settings come from environment variables (see README):
  APP_USERNAME, APP_PASSWORD, SECRET_KEY, HOST, PORT,
  MAX_UPLOAD_MB, MAX_VIDEO_SECONDS, TORCH_THREADS
"""
from __future__ import annotations

import hmac
import os
import secrets
import sys
import time
import uuid
from functools import wraps
from pathlib import Path

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
import inference

# Demo defaults so a local run works out of the box. Override both with
# environment variables (Space secrets / docker -e) for any shared deployment.
LOGIN_USER = os.environ.get("APP_USERNAME", "admin")
LOGIN_PASS = os.environ.get("APP_PASSWORD", "admin123")

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
UPLOAD_TTL_SECONDS = 15 * 60

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(16)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_DIR = config.PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
    return jsonify(status="ok", model_loaded=inference._model is not None)


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
        threshold=f"{inference.threshold:.4f}",
        frames=config.NUM_FRAMES,
        accuracy=inference.accuracy_display,
        accuracy_label=inference.accuracy_label,
        processing_time="—",
        max_upload_mb=MAX_UPLOAD_MB,
        max_seconds=int(inference.MAX_VIDEO_SECONDS),
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
                    inference.check_video(save_path)
                    start = time.perf_counter()
                    pred = inference.predict_video(save_path)
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


# Load models at import so gunicorn workers are ready (and fail fast) too.
inference.init()


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print(f"Dashboard ready — http://{host}:{port}  (user={LOGIN_USER})")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
