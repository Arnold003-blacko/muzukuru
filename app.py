"""
Muzukuru demo interface.

Run:
  .\\.venv\\Scripts\\streamlit.exe run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzukuru.config import CLASS_NAMES, load_config
from muzukuru.infer import classify_npz_clip, classify_video, load_classifier

st.set_page_config(
    page_title="Muzukuru",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

:root {
  --bg: #0f1c1a;
  --panel: #162824;
  --ink: #e8f0ee;
  --muted: #9bb0aa;
  --accent: #3d9b84;
  --warn: #e0a45a;
  --danger: #d4675a;
}

html, body, [class*="css"] {
  font-family: "DM Sans", sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, #1d3a34 0%, transparent 55%),
    radial-gradient(900px 500px at 100% 0%, #24352f 0%, transparent 50%),
    linear-gradient(180deg, #0f1c1a 0%, #12201d 100%);
}

h1, h2, h3, .brand {
  font-family: "Fraunces", Georgia, serif !important;
  letter-spacing: -0.02em;
}

.brand {
  font-size: 3rem;
  margin: 0;
  color: var(--ink);
}

.tagline {
  color: var(--muted);
  font-size: 1.05rem;
  max-width: 36rem;
  margin: 0.4rem 0 1.5rem;
}

.panel {
  background: rgba(22, 40, 36, 0.85);
  border: 1px solid rgba(61, 155, 132, 0.25);
  border-radius: 12px;
  padding: 1.1rem 1.25rem;
}

.status-ok { color: var(--accent); font-weight: 700; }
.status-bad { color: var(--danger); font-weight: 700; }
.status-warn { color: var(--warn); font-weight: 700; }

div[data-testid="stMetricValue"] {
  font-family: "Fraunces", Georgia, serif;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def _cached_model(ckpt: str, cfg_path: str):
    return load_classifier(ckpt, cfg_path)


def _label_style(label: str) -> str:
    if label in ("fall", "collapse"):
        return "status-bad"
    if label in ("prolonged_immobility", "abnormal_repetitive_movement"):
        return "status-warn"
    return "status-ok"


def main() -> None:
    cfg = load_config()
    ckpt = Path(cfg["paths"]["checkpoints"]) / "best.pt"

    st.markdown('<p class="brand">Muzukuru</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="tagline">Temporal movement classifier for older adults living alone. '
        "Distinguishes real danger from everyday floor-level activity.</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Model")
        if ckpt.exists():
            st.markdown('<span class="status-ok">Checkpoint loaded</span>', unsafe_allow_html=True)
            st.caption(str(ckpt))
            try:
                _cached_model(str(ckpt), str(ROOT / "configs" / "default.yaml"))
                st.caption("BiGRU + attention ready")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
        else:
            st.markdown('<span class="status-bad">No checkpoint</span>', unsafe_allow_html=True)
            st.caption("Run training first: python scripts/train.py")

        st.subheader("Classes")
        for name in CLASS_NAMES:
            st.write(f"- `{name}`")

        max_frames = st.slider("Max video frames", 90, 600, 300, 30)

    tab_video, tab_clip, tab_overview = st.tabs(
        ["Classify video", "Classify saved clip", "System overview"]
    )

    with tab_video:
        st.markdown("### Upload a clip")
        st.caption("MediaPipe Pose runs on each frame, then the temporal model scores ~3s windows.")
        upload = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv", "webm"])
        if upload is not None and ckpt.exists():
            tmp = ROOT / "data" / "raw" / "_ui_upload" / upload.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(upload.getbuffer())
            with st.spinner("Extracting pose and classifying..."):
                try:
                    result = classify_video(
                        tmp,
                        checkpoint=ckpt,
                        max_frames=max_frames,
                    )
                except Exception as e:
                    st.error(f"Classification failed: {e}")
                    result = None
            if result:
                _render_result(result, cfg)

    with tab_clip:
        st.markdown("### Pick a feature clip from `data/windows`")
        clips = sorted(Path(cfg["paths"]["windows"]).glob("*.npz"))
        if not clips:
            st.info("No `.npz` clips found. Generate synthetic data or run the pose pipeline.")
        else:
            choice = st.selectbox("Clip", clips, format_func=lambda p: p.name)
            if st.button("Classify clip", type="primary") and ckpt.exists():
                with st.spinner("Running model..."):
                    result = classify_npz_clip(choice, checkpoint=ckpt)
                _render_result(result, cfg, show_truth=True)

    with tab_overview:
        st.markdown("### What this screen is")
        st.write(
            "This is the **classifier demo** for Muzukuru: pose dynamics in, class label out. "
            "The full product chain (rules engine, GSM voice call, offline fallback) is not wired here yet."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Window", f"{cfg['windows']['length']} frames")
        c2.metric("Pose backend", "MediaPipe")
        c3.metric("Model", "BiGRU + attention")


def _render_result(result: dict, cfg: dict, show_truth: bool = False) -> None:
    label = result.get("summary_label", "unknown")
    css = _label_style(label)
    st.markdown(
        f'<div class="panel"><h2>Decision: '
        f'<span class="{css}">{label.replace("_", " ")}</span></h2></div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cols[0].metric("Frames", result.get("frame_count", "—"))
    cols[1].metric("Windows", len(result.get("windows", [])))
    if "pose_valid_ratio" in result:
        cols[2].metric("Pose coverage", f"{result['pose_valid_ratio'] * 100:.0f}%")
    if show_truth and result.get("true_label") is not None:
        truth = cfg["class_names"][int(result["true_label"])]
        cols[3].metric("True label", truth.replace("_", " "))

    windows = result.get("windows", [])
    if not windows:
        st.warning("No valid windows (need enough consecutive frames with a detected pose).")
        return

    st.markdown("### Window scores")
    import pandas as pd

    rows = []
    for w in windows:
        row = {
            "start": w["start_frame"],
            "end": w["end_frame"],
            "label": w["label"],
            "confidence": round(w["confidence"], 3),
        }
        row.update({k: round(v, 3) for k, v in w["probs"].items()})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Average probability bar chart across windows
    avg = {name: float(np.mean([w["probs"][name] for w in windows])) for name in cfg["class_names"]}
    st.bar_chart(avg)


if __name__ == "__main__":
    main()
