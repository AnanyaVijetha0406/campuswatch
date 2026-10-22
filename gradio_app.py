"""
Gradio front-end for Hugging Face Spaces (free CPU tier, no Docker needed).

Same model and thresholds as the Flask dashboard in app.py; the shared logic
lives in inference.py. Run locally with:  python gradio_app.py
"""
from __future__ import annotations

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

TITLE = "CampusWatch — campus ragging / violence screening"
DESCRIPTION = (
    f"Upload a short clip (up to {int(inference.MAX_VIDEO_SECONDS)} s). The model samples "
    f"{config.NUM_FRAMES} frames, finds the main person with YOLOv8n, embeds the crop with "
    "ResNet18 and classifies the sequence with a BiLSTM + temporal attention head.\n\n"
    "**Screening aid only.** Results can be wrong and must be reviewed by a person "
    "before any action is taken."
)
FOOTER = (
    f"Decision threshold {inference.threshold:.4f} (tuned on validation). "
    f"{inference.accuracy_label.title()}: {inference.accuracy_display}. "
    "See the README for datasets, results and limitations."
)


def analyse(video_path: str | None):
    if not video_path:
        raise gr.Error("Please upload or record a video first.")
    path = Path(video_path)
    try:
        inference.check_video(path)
    except ValueError as exc:
        raise gr.Error(str(exc))
    start = time.perf_counter()
    pred = inference.predict_video(path)
    elapsed = time.perf_counter() - start
    p = pred["probability"]
    scores = {"Violence / ragging": p, "Normal": 1.0 - p}
    flag = "🚨 HIGH THREAT — " if pred["is_violence"] else "✅ "
    summary = (
        f"### {flag}{pred['verdict']}\n"
        f"Confidence {pred['confidence'] * 100:.1f}% · violence probability {p * 100:.1f}% "
        f"(alert threshold {pred['threshold'] * 100:.1f}%) · processed in {elapsed:.1f}s"
    )
    return summary, scores


with gr.Blocks(title="CampusWatch") as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Video clip", sources=["upload"])
            run = gr.Button("Detect", variant="primary")
        with gr.Column():
            summary = gr.Markdown()
            scores = gr.Label(label="Class probabilities", num_top_classes=2)
    gr.Markdown(FOOTER)
    run.click(analyse, inputs=video, outputs=[summary, scores])

# One inference at a time: the models are shared and CPU-bound.
demo.queue(max_size=8, default_concurrency_limit=1)

if __name__ == "__main__":
    # ssr_mode=False: Spaces enable experimental SSR by default, which re-renders
    # the whole page after load and makes it flicker.
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
