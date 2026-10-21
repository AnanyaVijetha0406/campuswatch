# CampusWatch — violence / ragging screening dashboard (CPU inference).
# Everything the app needs (code, weights, dependencies) is baked into the image;
# nothing is downloaded when the container starts.
#
#   docker build -t campuswatch .
#   docker run -p 7860:7860 campuswatch
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    TORCH_HOME=/app/.torch \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app

# Shared libraries the OpenCV wheel links against (missing from the slim base image).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image small (no CUDA libraries).
COPY requirements.txt .
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Bake the ImageNet ResNet18 weights into the image so first start needs no network.
RUN python -c "from torchvision import models; models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)"

COPY config.py model.py inference.py app.py ./
COPY templates ./templates
COPY static ./static
COPY yolov8n.pt ./
COPY outputs_no_UCF/best_model_combined2_finetuned.pt outputs_no_UCF/metrics_combined2_finetuned.json ./outputs_no_UCF/

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/uploads \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','7860'), timeout=4)"

# One worker: the models are loaded once per process and inference is serialized
# anyway. Threads let the login page and health check answer during a request.
CMD ["sh", "-c", "exec gunicorn --bind ${HOST}:${PORT} --workers 1 --threads 4 --timeout 180 app:app"]
