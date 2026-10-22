"""
Gradio front-end for Hugging Face Spaces (free CPU tier, no Docker needed).

Same model, thresholds and look as the Flask dashboard in app.py; the shared
logic lives in inference.py. Run locally with:  python gradio_app.py
"""
from __future__ import annotations

import html
import os
import time
from pathlib import Path

# Models stay on CPU. On a ZeroGPU Space CUDA is only emulated outside
# @spaces.GPU functions, so loading and running on CPU is the safe choice.
os.environ.setdefault("CAMPUSWATCH_DEVICE", "cpu")

import gradio as gr

import config
import inference

try:
    import spaces  # present on Hugging Face Spaces

    gpu_task = spaces.GPU(duration=10)
except ImportError:  # local run / Docker: no-op decorator

    def gpu_task(fn):
        return fn


@gpu_task
def _zerogpu_registration() -> None:
    """ZeroGPU refuses to start a Space with no @spaces.GPU function. The model
    runs on CPU, so this is never called and no GPU quota is spent per request."""


inference.init()

# Lets the result panel play the alarm: served at /gradio_api/file=static/alarm.wav
gr.set_static_paths(paths=[config.PROJECT_ROOT / "static"])
ALARM_URL = "/gradio_api/file=static/alarm.wav"

CSS = """
:root, .dark, .gradio-container {
    --body-background-fill: #0b1220;
    --background-fill-primary: #111827;
    --background-fill-secondary: #0b1220;
    --block-background-fill: #111827;
    --block-border-color: #1e293b;
    --border-color-primary: #1e293b;
    --body-text-color: #ffffff;
    --body-text-color-subdued: #aab4c3;
    --block-label-text-color: #aab4c3;
    --input-background-fill: #0b1220;
    --color-accent: #ff7a00;
    --button-primary-background-fill: linear-gradient(45deg, #ff7a00, #ff9a3c);
    --button-primary-background-fill-hover: linear-gradient(45deg, #ff7a00, #ff9a3c);
    --button-primary-text-color: #ffffff;
}
body, .gradio-container {
    background: #0b1220 !important;
    font-family: 'Segoe UI', sans-serif !important;
}
.gradio-container, .gradio-container main, .gradio-container .main, .gradio-container .wrap { max-width: 100% !important; }
.gradio-container { padding: 0 !important; }
footer { display: none !important; }

.cw-header {
    display: flex; justify-content: space-between; align-items: center;
    background: #111827; padding: 10px 25px; border-bottom: 2px solid #ff7a00;
}
.cw-logo { font-size: 22px; font-weight: bold; color: #ff7a00; letter-spacing: 1px; }
.cw-status { color: #22c55e; font-size: 14px; }

.cw-body { padding: 12px 20px; gap: 14px !important; flex-wrap: nowrap !important; }
.cw-body > div { min-width: 0 !important; }
@media (max-width: 900px) { .cw-body { flex-wrap: wrap !important; } .cw-body > div { min-width: 100% !important; } }
.cw-card {
    background: #111827 !important; border-radius: 12px !important;
    padding: 14px 18px !important; box-shadow: 0 0 20px rgba(255,122,0,0.08);
    border: none !important; gap: 8px !important;
}
.cw-card h3 { color: #ff7a00; margin: 0 0 4px; font-size: 1.1em; }
.cw-note { color: #cbd5e1; font-size: 13px; margin: 2px 0; }
.cw-warn { color: #8a94a6; font-size: 11px; margin-top: 6px; }

.cw-upload { border: 2px dashed #ff7a00 !important; border-radius: 10px !important;
             background: #0b1220 !important; min-height: 0 !important; }
.cw-upload .wrap, .cw-upload button { min-height: 0 !important; padding: 8px !important; }
.cw-upload { font-size: 13px !important; }
.cw-detect {
    background: linear-gradient(45deg, #ff7a00, #ff9a3c) !important; border: none !important;
    color: #fff !important; font-size: 16px !important; border-radius: 6px !important;
}

.cw-empty { color: #667; font-size: 13px; padding: 20px 10px; text-align: center; }
.cw-video video { max-height: 30vh; }
.cw-result { text-align: center; }
.cw-result h2 { margin: 6px 0; font-size: 1.3em; }
.cw-result p { margin: 3px 0; }
.cw-threat { padding: 6px 18px; border-radius: 20px; font-weight: bold;
             margin-bottom: 10px; display: inline-block; color: #fff; }
.cw-high { background: red; box-shadow: 0 0 15px red; animation: cw-pulse 1s infinite; }
.cw-low { background: green; box-shadow: 0 0 10px green; }
@keyframes cw-pulse { 50% { transform: scale(1.05); } }
.cw-progress { background: #1e293b; border-radius: 20px; overflow: hidden; height: 10px; margin: 6px 0; }
.cw-bar { height: 10px; background: linear-gradient(to right, #ff7a00, red); }
.cw-siren { font-size: 40px; animation: cw-blink 1s infinite; }
@keyframes cw-blink { 50% { opacity: 0.4; } }

.cw-stats {
    margin: 0 25px 14px; background: #111827; padding: 10px 15px; border-radius: 12px;
    display: flex; justify-content: space-between; border: 1px solid #1e293b;
    flex-wrap: wrap; gap: 10px;
}
.cw-stat { text-align: center; flex: 1; min-width: 90px; }
.cw-stat-label { font-size: 12px; color: #aaa; }
.cw-stat-value { font-size: 16px; font-weight: bold; margin-top: 3px; color: #fff; }
"""

HEADER_HTML = """
<div class="cw-header">
  <div class="cw-logo">CampusWatch</div>
  <div class="cw-status">● System Active</div>
</div>
"""

UPLOAD_NOTES_HTML = f"""
<h3>Upload Feed</h3>
<p class="cw-note">MP4, AVI, MOV, MKV, WEBM &middot; max {int(inference.MAX_VIDEO_SECONDS)} s</p>
<p class="cw-note">Frames sampled per clip: {config.NUM_FRAMES}</p>
<p class="cw-warn">Screening aid only: results can be wrong and must be reviewed by a person
before any action is taken.</p>
"""

EMPTY_RESULT_HTML = '<h3>Detection Result</h3><div class="cw-empty">Upload a clip to run detection.</div>'


def stats_html(processing: str = "—") -> str:
    items = [
        ("MODEL", '<span style="color:#22c55e;">ACTIVE</span>'),
        ("FRAMES", str(config.NUM_FRAMES)),
        ("PROCESSING", html.escape(processing)),
        (html.escape(inference.accuracy_label), html.escape(inference.accuracy_display)),
        ("THRESHOLD", f"{inference.threshold:.4f}"),
    ]
    cells = "".join(
        f'<div class="cw-stat"><div class="cw-stat-label">{label}</div>'
        f'<div class="cw-stat-value">{value}</div></div>'
        for label, value in items
    )
    return f'<div class="cw-stats">{cells}</div>'


def result_html(pred: dict) -> str:
    conf = f"{pred['confidence'] * 100:.1f}%"
    bar = f'<div class="cw-progress"><div class="cw-bar" style="width:{conf};"></div></div>'
    if pred["is_violence"]:
        return (
            '<div class="cw-result"><h3>Detection Result</h3>'
            '<div class="cw-threat cw-high">HIGH THREAT</div>'
            f'<h2 style="color:red;">{html.escape(pred["verdict"])}</h2>'
            f"{bar}<p>Confidence: {conf}</p><p>Status: Alert Triggered</p>"
            '<div class="cw-siren">🚨</div>'
            f'<audio autoplay><source src="{ALARM_URL}" type="audio/wav"></audio></div>'
        )
    return (
        '<div class="cw-result"><h3>Detection Result</h3>'
        '<div class="cw-threat cw-low">SAFE</div>'
        f'<h2 style="color:#22c55e;">{html.escape(pred["verdict"])}</h2>'
        f"{bar}<p>Confidence: {conf}</p><p>Status: Normal Activity</p></div>"
    )


def analyse(video_path: str | None):
    if not video_path:
        raise gr.Error("Please choose a video file first.")
    path = Path(video_path)
    try:
        inference.check_video(path)
    except ValueError as exc:
        raise gr.Error(str(exc))
    start = time.perf_counter()
    pred = inference.predict_video(path)
    elapsed = time.perf_counter() - start
    return result_html(pred), stats_html(f"~{elapsed:.1f}s")


with gr.Blocks(
    title="CampusWatch",
    theme=gr.themes.Base(primary_hue="orange", neutral_hue="slate"),
    fill_width=True,
    css=CSS,
    js="() => { document.body.classList.add('dark'); }",
) as demo:
    gr.HTML(HEADER_HTML)
    with gr.Row(elem_classes="cw-body"):
        with gr.Column(scale=2, elem_classes="cw-card"):
            gr.HTML(UPLOAD_NOTES_HTML)
            file_in = gr.File(
                label="Select Video",
                file_types=["video"],
                type="filepath",
                height=90,
                elem_classes="cw-upload",
            )
            run = gr.Button("Detect", elem_classes="cw-detect")
        with gr.Column(scale=3, elem_classes="cw-card"):
            gr.HTML("<h3>Uploaded Clip</h3>")
            preview = gr.Video(
                show_label=False,
                interactive=False,
                container=False,
                height=240,
                show_share_button=False,
                show_download_button=False,
                elem_classes="cw-video",
            )
        with gr.Column(scale=2, elem_classes="cw-card"):
            result = gr.HTML(EMPTY_RESULT_HTML)
    stats = gr.HTML(stats_html())

    file_in.change(lambda p: p, inputs=file_in, outputs=preview, show_progress="hidden")
    run.click(analyse, inputs=file_in, outputs=[result, stats], api_name="analyse")

# One inference at a time: the models are shared and CPU-bound.
demo.queue(max_size=8, default_concurrency_limit=1)

if __name__ == "__main__":
    # ssr_mode=False: Spaces enable experimental SSR by default, which re-renders
    # the whole page after load and makes it flicker.
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
