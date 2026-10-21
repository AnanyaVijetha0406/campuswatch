---
title: CampusWatch
emoji: 🎥
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# CampusWatch — Campus Ragging / Violence Screening

A web dashboard that takes a short video clip and says whether it shows violence
or ragging-like physical aggression, with a confidence score.

**Live demo:** _add the Hugging Face Space URL here after deploying_
**Demo login:** _add the username/password you set in the Space secrets_ (local default: `admin` / `admin123`)

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

| File | Purpose |
|---|---|
| `app.py` | Flask dashboard (login, upload, verdict) |
| `model.py` | Feature extraction pipeline and the BiLSTM + attention classifier |
| `config.py` | Paths, frame count, threshold, model location |
| `infer.py`, `batch_infer.py` | Command-line inference on one video / a folder |
| `finetune_combined.py` | Training script (run on Kaggle) |
| `eval_shanghaitech.py`, `eval_ucfcrime.py` | Cross-dataset evaluation scripts |
| `outputs_no_UCF/` | The deployed checkpoint and its metrics |

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
python infer.py path/to/clip.mp4
python batch_infer.py path/to/folder
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `APP_USERNAME` / `APP_PASSWORD` | `admin` / `admin123` | Dashboard login. **Set your own for any shared deployment.** |
| `SECRET_KEY` | random per start | Session signing key. Set it so logins survive restarts. |
| `HOST` / `PORT` | `127.0.0.1` / `5000` (`0.0.0.0` / `7860` in Docker) | Where the server listens |
| `MAX_UPLOAD_MB` | `100` | Largest accepted upload |
| `MAX_VIDEO_SECONDS` | `60` | Longest accepted clip |
| `TORCH_THREADS` | `min(4, cores)` | CPU threads for inference |

Uploaded clips are stored under a random name, visible only to the session that
uploaded them, and deleted on the next upload, on logout, or after 15 minutes.

## Deploy to Hugging Face Spaces

1. Create a Space at https://huggingface.co/new-space, choose **Docker** as the SDK.
2. Push this repository to the Space (`git remote add space https://huggingface.co/spaces/<you>/<name>` then `git push space main`). The YAML block at the top of this README configures the port.
3. In the Space **Settings → Variables and secrets**, add `APP_USERNAME`, `APP_PASSWORD` and `SECRET_KEY` as secrets.
4. Wait for the build to finish, open the Space URL, and put it at the top of this README.

The free CPU tier works. The first request after the Space wakes from sleep is slower.
Vercel is not suitable: PyTorch and OpenCV exceed its function size limit.

## Results

Metrics of the deployed model on its **validation split** (208 clips, threshold tuned on this same split, so treat as optimistic):

| Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| 0.875 | 0.695 | 0.837 | 0.759 |

Confusion matrix (rows = truth): normal → 141 correct / 18 false alarms; violence → 41 caught / 8 missed.

A separate hand-picked check on 6 ragging clips never used in training found 5 of 6.
That is a very small sample and should be read as an indication only.
Evaluation on other datasets was weaker (ShanghaiTech F1 0.375, UCF-Crime Abuse recall 0.30).
There is no separate untouched test split yet.

## Data

Training combined these sources; none of the videos are redistributed here.

| Dataset | Use | Link |
|---|---|---|
| RLVS (Real Life Violence Situations) | Base training | kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset |
| UBI-FIGHTS | Fine-tuning | kaggle.com/datasets/intissarziani/ubi-fightsall |
| Self-collected staged ragging / hard-negative clips (39) | Fine-tuning | Not published (third-party video) |
| ShanghaiTech Campus (`nikanvasei/shanghaitech-campus-dataset-test`), UCF-Crime Abuse/Normal (`minmints/ufc-crime-full-dataset`) | Evaluation (UCF-Crime was also used in one ablation run) | Kaggle |

`split_dataset.py` splits a dataset by video file (never by frame). Training was run on Kaggle with `finetune_combined.py`; install `requirements-train.txt` to run the training and evaluation scripts yourself.

## Limitations

- **It detects violence-like motion and appearance, not ragging as such.** Non-contact
  humiliation (for example pouring water, forced feeding) was missed by every checkpoint.
- **Only the largest person per frame is analysed**, and there is no tracking, so group
  interactions are only partly seen.
- **16 frames per clip**: a short incident inside a long video can be diluted. Clips are
  limited to 60 s for this reason and for speed.
- **False alarms are common** (precision 0.695 on validation, and real-world precision
  would be lower because real incidents are rare).
- Training ragging data is small (39 clips) and staged, so behaviour on real CCTV
  (low resolution, night, crowded) is untested.
- Roughly 30% of sampled frames can contain no detected person, which the model sees as empty.
- Analysing video of students raises consent and privacy questions that this demo does
  not address. Do not use it to accuse anyone.

## Licence and credits

Uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0) and
torchvision's ImageNet ResNet18. Datasets keep their own licences. Add a `LICENSE`
file for this repository's own code before publishing.
