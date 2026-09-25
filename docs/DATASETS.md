# Public training datasets

## Classes

| ID | Name | Notes |
|----|------|--------|
| 0 | normal_activity | walking, standing, sitting, voluntary laying |
| 1 | fall | rapid descent falls |
| 2 | collapse | empty-chair / syncope-like (UP-Fall A5) |
| 3 | prolonged_immobility | rare in public sets; mostly self-recorded |
| 4 | abnormal_repetitive_movement | rare in public sets; mostly self-recorded |

Sitting/laying map to `normal_activity` unless a **rapid hip-center descent** is detected in the clip.

## Sources

### 1. UP-Fall improved 3D skeletons (primary)
- Zenodo: https://zenodo.org/records/12773013
- Already MediaPipe BlazePose 33 joints (x,y,z); visibility synthesized if missing
- Zenodo release contains fall activities A1–A5 only (no separate ADL A6–A11 files)
- Ingest splits each CSV using the impact `LABEL` column:
  - pre-impact → `normal_activity`
  - impact neighborhood → `fall` or `collapse` (A5)
  - post-impact (when long enough) → `prolonged_immobility`
- Auto-downloaded by `scripts/prepare_datasets.py --upfall`

### 2. UR Fall Detection Dataset
- Official RGB: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
- Roboflow Universe (optional): set `ROBOFLOW_API_KEY` and use `--roboflow`
  (project slug varies; default workspace `fall-detection-w7nxl` / `ur-fall`)
- Videos/frames → MediaPipe Pose extraction

### 3. Le2i Fall Detection Dataset
- Official (~9 GB): https://search-data.ubfc.fr/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html
- Place under `data/external/le2i/` (e.g. `Home_01/`, `Coffee_room_01/` with `Videos/` + `Annotation_files/`)
- Then: `python scripts/prepare_datasets.py --le2i`

## Prepare + train

```powershell
.\.venv\Scripts\activate
python scripts/prepare_datasets.py --upfall
# optional supplements:
python scripts/prepare_datasets.py --urfall --urfall-falls 15 --urfall-adls 15 --keep-existing
python scripts/prepare_datasets.py --le2i --keep-existing

python scripts/train.py
```

Or one shot: `python scripts/prepare_datasets.py --all` (Le2i skipped gracefully if absent).

Windows are built at train time from `data/windows/*.npz` (length 90, 50% overlap via `stride_train: 30`).
