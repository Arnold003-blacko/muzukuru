# Muzukuru: Temporal Fall-Classification Training Pipeline

Training-only stack for the 5-class temporal model (MediaPipe Pose → engineered features → Bidirectional GRU + attention). This is not the full Muzukuru response chain (rules engine, GSM voice, offline path). Those come later.

## Classes

| ID | Name |
|----|------|
| 0 | normal |
| 1 | fall |
| 2 | collapse |
| 3 | prolonged_immobility |
| 4 | abnormal_repetitive_movement |

## Pipeline

```
raw video  →  MediaPipe Pose (33×4)
           →  normalize (hip-center, torso scale)
           →  features (keypoints, vel/acc, torso angle, bbox aspect, CoM height)
           →  sliding windows (90 frames, 50% overlap train)
           →  BiGRU+attention classifier
           →  metrics (per-class F1, confusion matrix, latency, false alarms/hour)
```

## Setup

```bash
cd "FACIAL RECOGNITION"
# Prefer Python 3.10–3.12 (MediaPipe / PyTorch wheels). Avoid 3.14 for now.
py -3.12 -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data layout

1. Put videos under `data/raw/...`
2. Copy `data/metadata.example.csv` → `data/metadata.csv` and fill rows:

```text
video_path,clip_id,subject_id,session_id,label
```

`label` may be a class name or integer id. Splits are by **subject** (not by clip), so the same person never appears in both train and test.

## Demo interface

```bash
.\.venv\Scripts\activate
streamlit run app.py
```

Opens in the browser (usually http://localhost:8501). Upload a video or classify a clip from `data/windows`.

## Public datasets

See [docs/DATASETS.md](docs/DATASETS.md). Quick start (UP-Fall from Zenodo):

```bash
python scripts/prepare_datasets.py --upfall
python scripts/train.py
```

## Commands

```bash
# 1) Pose extraction
python scripts/extract_poses.py

# 2) Feature sequences (one .npz per clip under data/windows/)
python scripts/build_windows.py

# 3) Train
python scripts/train.py

# 4) Evaluate best checkpoint
python scripts/evaluate.py --split test
```

Smoke-test without real videos:

```bash
python scripts/make_synthetic_data.py
python scripts/train.py
python scripts/evaluate.py
```

Transfer fine-tune (after a binary pretrain checkpoint exists):

```bash
python scripts/train.py --pretrained checkpoints/pretrain_binary.pt
```

Set `train.freeze_backbone_epochs` in `configs/default.yaml` to freeze CNN + early GRU for the first N epochs.

## Model

Input shape: `(batch, 90, F)` where `F ≈ 114` with MediaPipe xyz + derived features (see `feature_dim_from_cfg`).

Architecture: 1D-CNN (k=3) → Bidirectional GRU (2×128) → temporal attention → FC 128→64→5.

Training: AdamW, cosine LR, class-weighted cross-entropy (or focal loss), subject-level 70/15/15 split, skeleton augmentations (flip, rotate, temporal jitter, Gaussian noise, speed variation).

## Outputs

| Path | Content |
|------|---------|
| `checkpoints/best.pt` | Best val-accuracy weights + config |
| `reports/test_metrics.json` | Per-class P/R/F1, confusion matrix, latency, FAR/hour |
| `reports/confusion_matrix.png` | Confusion matrix figure |
| `reports/history.json` | Per-epoch train/val curves |

## Config

All knobs live in [`configs/default.yaml`](configs/default.yaml).
