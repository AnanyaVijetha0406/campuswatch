# CampusWatch — Campus Ragging / Violence Screening

A web dashboard that takes a short video clip and says whether it shows violence
or ragging-like physical aggression, with a confidence score.

**Live demo:** https://huggingface.co/spaces/AnanyaVij/campuswatch (free ZeroGPU Space; the first request after idle can take a minute while it wakes)
**Login:** none for the Gradio Space. The Flask dashboard (`app.py`, used by the Docker image) asks for `APP_USERNAME` / `APP_PASSWORD` (local default: `admin` / `admin123`).

> **This is a screening aid, not an evidence tool.** It can miss incidents and
> raise false alarms (see [Limitations](#limitations)). A person must review every
> alert before any action is taken.

## How it works

```
video ──► 16 frames, evenly spaced over the whole clip
      ──► YOLOv8n: find the largest person in each frame
      ──► crop ──► ResNet18 (frozen, ImageNet) ──► 512-d appearance vector
      ──► + normalised box position (4-d) = 516-d per frame
      ──► BiLSTM (128 hidden, bidirectional) ──► attention over time
      ──► Linear ──► sigmoid ──► probability ──► compare with tuned threshold
```

Only the BiLSTM + attention head is trained; YOLOv8n and ResNet18 stay frozen.
The decision threshold (0.6613) was tuned on a validation set, not fixed at 0.5.

## Project structure

```
.
├── app.py                 # Flask dashboard with login (what the Docker image runs)
├── gradio_app.py          # Gradio app (what the Hugging Face Space runs)
├── inference.py           # Shared inference core used by both apps
├── model.py               # Feature pipeline (YOLO + ResNet18) and BiLSTM + attention classifier
├── config.py              # Paths, frame count, threshold, model settings
├── templates/, static/    # Flask HTML templates, alarm sound
├── weights/               # yolov8n.pt + the deployed classifier and its metrics
├── results/               # Metrics JSON of the other training runs (k1 = largest person, k3 = top-3 ablation)
├── scripts/
│   ├── data/              # split_dataset, build_ragging_labels, check_clips, extract_features
│   ├── train/             # train.py (RLVS base model), finetune_combined.py (deployed recipe)
│   ├── eval/              # eval_ucfcrime, compare_yolo_detectors
│   └── infer/             # infer (one clip), batch_infer (a folder), sliding_window_infer (long video)
├── Dockerfile, requirements.txt, requirements-train.txt, packages.txt
```

Run every script from the repository root, for example `python scripts/infer/infer.py clip.mp4`.

## Run it

### Option 1 — Docker (nothing else to install)

```bash
docker build -t campuswatch .
```

```bash
docker run -p 7860:7860 campuswatch
```

Open http://localhost:7860. To set your own login:

```bash
docker run -p 7860:7860 -e APP_USERNAME=me -e APP_PASSWORD=choose-a-password -e SECRET_KEY=any-long-random-string campuswatch
```

All weights (YOLOv8n, the trained classifier, ResNet18 ImageNet) are inside the
image, so the container does not download anything when it starts.

### Option 2 — Python

Python 3.12 recommended.

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. (`PORT=8080 python app.py` to change the port.)

### Command line

```bash
python scripts/infer/infer.py path/to/clip.mp4
python scripts/infer/batch_infer.py path/to/folder
python scripts/infer/sliding_window_infer.py path/to/long_video.mp4
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `APP_USERNAME` / `APP_PASSWORD` | `admin` / `admin123` | Dashboard login. **Set your own for any shared deployment.** |
| `SECRET_KEY` | random per start | Session signing key. Set it so logins survive restarts. |
| `HOST` / `PORT` | `127.0.0.1` / `5000` (`0.0.0.0` / `7860` in Docker) | Where the server listens |
| `MAX_UPLOAD_MB` | `100` | Largest accepted upload |
| `TORCH_THREADS` | `min(4, cores)` | CPU threads for inference |

Uploaded clips are stored under a random name, visible only to the session that
uploaded them, and deleted on the next upload, on logout, or after 15 minutes.

## Deploy to Hugging Face Spaces

The live demo runs `gradio_app.py` (same model and thresholds as the Flask dashboard)
on a Gradio Space, which is free. The Docker SDK is a paid Spaces feature, so the Docker
image is for running the project yourself, not for the Space.

1. Create a Space at https://huggingface.co/new-space with the **Gradio** SDK (Blank template). If only ZeroGPU hardware is offered on your account, pick it; `gradio_app.py` keeps the models on CPU and wraps the request in `@spaces.GPU`, as ZeroGPU requires.
2. Add this settings block as the very first lines of `README.md` (Hugging Face reads it to pick the Gradio version, Python version and entry file), commit, then push the repository to the Space:

   ```yaml
   ---
   title: CampusWatch
   emoji: 🎥
   colorFrom: yellow
   colorTo: red
   sdk: gradio
   sdk_version: 5.50.0
   python_version: "3.12.12"
   app_file: gradio_app.py
   pinned: false
   ---
   ```

   ```bash
   git remote add space https://huggingface.co/spaces/<you>/<name>
   git push space main --force
   ```

   (`--force` replaces the starter files the Space was created with.)
3. Wait for the build to finish and open the Space URL.

The first request after the Space wakes from sleep is slower. Vercel is not suitable:
PyTorch and OpenCV exceed its function size limit. The Gradio Space has no login page;
the Flask dashboard (`app.py`, with login) is what the Docker image runs.

## Results

All results are on a held-out test set of **12 staged ragging clips and 17 normal clips**
that were never used for training. The set is small, so the numbers are indicative only.

### Training-data ablation

Each row adds one training source to the one above it.

| Training data | Ragging caught (of 12) | False alarms (of 17) |
|---|---|---|
| RLVS | 41.7% (5) | 0.0% (0) |
| + UBI-Fights | 75.0% (9) | 23.5% (4) |
| + UCF-Crime | 66.7% (8) | 23.5% (4) |
| + Staged ragging clips (with UCF-Crime) | 75.0% (9) | 29.4% (5) |
| **+ Staged ragging clips, no UCF-Crime (final model)** | **91.7% (11)** | 35.3% (6) |

- **Validation accuracy is not ragging detection.** The RLVS-only model reached 98.5%
  validation accuracy but caught only 41.7% of the held-out ragging clips.
- **More data is not always better.** Adding UCF-Crime lowered ragging recall from 75.0%
  to 66.7%, so it was left out of the final model.
- **Final model: 11 of 12 ragging clips caught** (indicative only, McNemar p = 0.5).

### Model comparison (3 seeds each)

| Model | Trained params | Validation F1 (%) | Ragging recall, 12 clips (%) | False alarms, 17 clips (%) |
|---|---|---|---|---|
| **BiLSTM + attention (this repo)** | 662K | 72.1 ± 2.6 | **83.3 ± 8.3** | 35.3 ± 10.2 |
| Transformer + attention | 340K | 79.8 ± 3.3 | 77.8 ± 4.8 | 52.9 ± 10.2 |
| VideoMAE-B (end-to-end) | 86.2M | 86.6 ± 4.7 | 66.7 ± 0.0 | 25.5 ± 3.4 |

- Higher source-domain scores did not give better ragging detection.
- The BiLSTM had the best ragging recall with 130× fewer trained parameters than VideoMAE.
- It runs on a laptop CPU at 4.7 s per 16-frame clip (Intel Core i7-1355U).

## Data

Training ran in two stages. None of the videos are redistributed here.

1. **Stage 1:** a base violence model trained on RLVS.
2. **Stage 2:** adapted to surveillance footage and to ragging with UBI-Fights and the
   staged clips. The decision threshold (τ = 0.661) was set at maximum F1 on validation.

| Dataset | Clips | Role | Link |
|---|---|---|---|
| RLVS (Real Life Violence Situations) | ~2,000 violence / non-violence | Stage 1: base violence model | kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset |
| UBI-Fights | 1,000 surveillance fight / normal | Stage 2: domain adaptation | kaggle.com/datasets/intissarziani/ubi-fightsall |
| UCF-Crime (Abuse + Normal) | 100 | Ablation only, excluded from the final model | kaggle.com/datasets/minmints/ufc-crime-full-dataset |
| Staged ragging + normal | 27 + 12 | Stage 2: target-domain adaptation | Not published |
| Held-out test sets | 12 ragging + 17 normal | Never used for training | Not published |

### Dataset licences

- **No videos are included in this repository.** Download each dataset from its source
  and follow the terms its creators set. None of them publishes a clear open licence,
  so treat them as research-only unless their creators say otherwise.
- **The trained weights in `weights/` were learned from these datasets.** They are shared
  for research and education. If you plan any commercial use, check each dataset's terms
  with its creators first.
- The AGPL-3.0 licence of this repository covers the code only, not the datasets.
- If you use these datasets, please cite their authors:
  - RLVS: M. M. Soliman et al., "Violence Recognition from Videos using Deep Learning
    Techniques," ICICIS, 2019.
  - UBI-Fights: B. Degardin and H. Proença, "Human Activity Analysis: Iterative
    Weak/Self-Supervised Learning Frameworks for Detecting Abnormal Events," IJCB, 2020.
  - UCF-Crime: W. Sultani, C. Chen and M. Shah, "Real-World Anomaly Detection in
    Surveillance Videos," CVPR, 2018.

`scripts/data/split_dataset.py` splits a dataset by video file (never by frame). Training was run on Kaggle with `scripts/train/train.py` (stage 1) and then `scripts/train/finetune_combined.py` (stage 2); install `requirements-train.txt` to run the training and evaluation scripts yourself.

## Limitations

- **Small, staged data.** The model was adapted to ragging with only 27 staged ragging
  and 12 normal clips, and tested on 12 held-out ragging and 17 normal clips. With a
  test set this small the confidence intervals are wide, so the results are indicative
  only (McNemar p = 0.5 between the main variants).
- **Misses non-contact humiliation.** Low-motion, one-sided ragging (humiliation without
  physical contact) is the hardest case. End-to-end VideoMAE missed the same clips in
  every seed.
- **Not yet tested on real campus CCTV.** All evaluation used staged or public clips, not
  real multi-camera campus footage.
- **A screening aid, not an evidence tool.** Run it on-premises, have a person review
  every flag, and blur faces where possible. Do not use it to accuse anyone.

## Licence and credits

Uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0) and
torchvision's ImageNet ResNet18. Datasets keep their own licences.

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE),
the same licence as YOLOv8.
