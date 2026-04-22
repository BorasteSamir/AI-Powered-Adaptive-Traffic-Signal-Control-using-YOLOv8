# app.py — Main Streamlit entry point.
# Run: streamlit run app.py

import os
import sys
import time
import tempfile
from collections import deque

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from config import NUM_SIGNALS, MODEL_PATH
from model_loader import load_model
from detection import count_vehicles
from zones import split_into_zones, assign_to_zones
from signal_controller import SignalController
from per_signal_input import render_image_upload_panel, apply_image_results_to_controller
from dashboard import (
    draw_dashboard,
    render_full_signal_panel,
    render_metric_bar,
    render_analytics_sidebar,
    render_control_panel,
    update_history,
    render_chart,
    apply_custom_css,
    traffic_badge_html,
)


# ── Live countdown fragment ───────────────────────────────────────────────────

@st.fragment(run_every=1)
def _live_signal_panel() -> None:
    """Ticks every 1s — advances timer and redraws signal cards + control panel."""
    controller: SignalController = st.session_state.get("controller")
    if controller is None:
        return
    if not controller.manual_mode:
        controller.update_signals([sig.detections for sig in controller.signals])

    render_full_signal_panel(controller.signals, controller)
    st.markdown("---")
    action = render_control_panel(controller)
    if action["mode"] == "AUTO":
        controller.set_auto()
        st.session_state.control_mode = "AUTO"
        st.rerun()
    elif action["mode"] == "MANUAL":
        controller.set_manual()
        st.session_state.control_mode = "MANUAL"
        st.rerun()
    if action["set_signal"] is not None:
        controller.set_signal(action["set_signal"])
        st.rerun()
    if action["force_next"]:
        controller.force_next()
        st.rerun()


# ── Per-frame pipeline (video/webcam) ─────────────────────────────────────────

def process_frame(
    model,
    frame: np.ndarray,
    controller: SignalController,
    conf: float,
) -> np.ndarray:
    """Detect on full frame → split zones → update controller → annotate."""
    zones           = split_into_zones(frame)
    dets, _         = count_vehicles(model, frame, conf) if model else ([], [])
    zone_dets       = assign_to_zones(dets, zones)
    controller.update_signals(zone_dets)
    return draw_dashboard(frame, controller.signals, zones, controller.stats)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:

    st.set_page_config(
        page_title="Smart Traffic Management System",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_css()

    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:4px;">
          <span style="font-size:2.4rem;">🚦</span>
          <div>
            <h1 style="margin:0;font-size:2rem;
                       background:linear-gradient(90deg,#60a5fa,#34d399);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
              Smart Traffic Management System
            </h1>
            <p style="margin:0;color:#64748b;font-size:0.85rem;">
              YOLOv8 · Indian Driving Dataset · 4-Signal Corridor Simulation
            </p>
          </div>
        </div>
        <hr style="margin:10px 0 20px;">
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        model_path = st.text_input("Model path", value=MODEL_PATH)

        st.markdown("---")
        st.markdown("### 📹 Live Feed Source")
        st.markdown(
            "<small style='color:#64748b;'>Optional: upload a video or use webcam "
            "for the background feed. Signal timing is driven by the images below.</small>",
            unsafe_allow_html=True,
        )
        input_mode = st.radio("Source", ["None (Images Only)", "Upload Video", "Webcam"], index=0)

        uploaded_file = None
        use_webcam    = False
        if input_mode == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload a video file", type=["mp4", "avi", "mov", "mkv"]
            )
        elif input_mode == "Webcam":
            use_webcam = True

        st.markdown("---")
        st.markdown("### 🎛️ Detection Settings")
        conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)

        st.markdown("---")
        if input_mode != "None (Images Only)":
            st.markdown("### 💾 Recording")
            save_video = st.checkbox("Save processed video", value=False)
        else:
            save_video = False

        st.markdown("---")
        st.markdown("<small style='color:#475569;'>Trained on IDD · ultralytics YOLOv8</small>",
                    unsafe_allow_html=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(model_path)
    if model is None:
        st.warning(f"⚠️ Model **{model_path}** not found.")
        st.info("Signal simulation will run without live detection.")

    # ── Session state ─────────────────────────────────────────────────────────
    if "controller"   not in st.session_state:
        st.session_state.controller   = SignalController()
    if "history"      not in st.session_state:
        st.session_state.history      = {f"S{i+1}": [] for i in range(NUM_SIGNALS)}
    if "running"      not in st.session_state:
        st.session_state.running      = False
    if "fps_times"    not in st.session_state:
        st.session_state.fps_times    = deque(maxlen=30)
    if "control_mode" not in st.session_state:
        st.session_state.control_mode = "AUTO"
    if "img_results"  not in st.session_state:
        st.session_state.img_results  = {}

    controller: SignalController = st.session_state.controller

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 1 — 4-Image Upload Panel (always visible at top)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("#### 📸 Traffic Image Analysis — Per Signal")
    img_results = render_image_upload_panel(model, conf_threshold)
    st.session_state.img_results = img_results

    # Push image detections into controller whenever any image is uploaded
    any_uploaded = any(r["uploaded"] for r in img_results.values())
    if any_uploaded:
        apply_image_results_to_controller(img_results, controller)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 2 — Live Signal Panel (auto-ticking countdown)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("#### 🚦 Signal Control Panel")
    _live_signal_panel()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 3 — Video Feed (optional)
    # ══════════════════════════════════════════════════════════════════════════
    if input_mode != "None (Images Only)":
        st.markdown("#### 📹 Live Video Feed")
        video_placeholder = st.empty()

        info_col1, info_col2 = st.columns([2, 1])
        with info_col1:
            metrics_placeholder = st.empty()
        with info_col2:
            st.markdown("#### 📊 Analytics")
            analytics_placeholder = st.empty()

        st.markdown("#### 📈 Vehicle Count History")
        chart_placeholder = st.empty()

        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            if st.button("▶ Start", type="primary"):
                st.session_state.running = True
        with c2:
            if st.button("⏹ Stop"):
                st.session_state.running = False

        if st.session_state.running:
            cap      = None
            tmp_path = None
            writer   = None

            if use_webcam:
                cap = cv2.VideoCapture(0)
            elif uploaded_file is not None:
                suffix = os.path.splitext(uploaded_file.name)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                cap = cv2.VideoCapture(tmp_path)
            else:
                st.warning("Please upload a video or enable webcam.")
                st.session_state.running = False
                st.stop()

            if not cap.isOpened():
                st.error("❌ Could not open video source.")
                st.session_state.running = False
                st.stop()

            if save_video and tmp_path:
                out_path = tmp_path.replace(os.path.splitext(tmp_path)[-1], "_processed.mp4")
                writer   = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    cap.get(cv2.CAP_PROP_FPS) or 25,
                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                )

            frame_idx = 0
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    controller.stats.cycle_count += 1
                    continue

                t0        = time.time()
                annotated = process_frame(model, frame, controller, conf_threshold)

                st.session_state.fps_times.append(1.0 / max(time.time() - t0, 1e-6))
                controller.stats.fps         = float(np.mean(st.session_state.fps_times))
                controller.stats.frame_count += 1

                if writer:
                    writer.write(annotated)

                video_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_container_width=True,
                )
                st.session_state.history = update_history(
                    st.session_state.history, controller.signals
                )
                with metrics_placeholder.container():
                    render_metric_bar(controller.stats)
                with analytics_placeholder.container():
                    render_analytics_sidebar(controller.signals)
                if frame_idx % 5 == 0:
                    render_chart(chart_placeholder, st.session_state.history)

                frame_idx += 1
                time.sleep(0.01)

            cap.release()
            if writer:
                writer.release()
                st.success(f"✅ Saved to `{out_path}`")
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        else:
            # Idle video placeholder
            video_placeholder.markdown(
                """
                <div style="background:#0f172a;border:2px dashed #1e293b;
                            border-radius:12px;padding:60px 20px;
                            text-align:center;color:#475569;">
                  <div style="font-size:3rem;margin-bottom:12px;">🎥</div>
                  <div>Click <b>▶ Start</b> to begin the live feed.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with metrics_placeholder.container():
                render_metric_bar(controller.stats)

    else:
        # Images-only mode — show metrics below the signal panel
        st.markdown("#### 📊 System Metrics")
        render_metric_bar(controller.stats)

        st.markdown("#### 📈 Vehicle Count History")
        chart_placeholder = st.empty()
        render_chart(chart_placeholder, st.session_state.history)

        # Update history from image results
        if any_uploaded:
            st.session_state.history = update_history(
                st.session_state.history, controller.signals
            )


if __name__ == "__main__":
    main()
