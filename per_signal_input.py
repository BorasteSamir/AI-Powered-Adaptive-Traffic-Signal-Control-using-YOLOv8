# per_signal_input.py
# Per-signal image upload panel rendered on the MAIN PAGE.
# Each of the 4 signals gets one image uploader.
# After upload, YOLOv8 runs on each image and returns per-signal detections.

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from config import NUM_SIGNALS, GREEN_BASE_SEC, GREEN_PER_VEHICLE, GREEN_MAX_SEC, GREEN_MIN_SEC
from detection import count_vehicles


def _calc_green(weighted_score: float) -> float:
    return max(GREEN_MIN_SEC,
               min(GREEN_MAX_SEC, GREEN_BASE_SEC + weighted_score * GREEN_PER_VEHICLE))


def render_image_upload_panel(model, conf: float) -> dict:
    """
    Render a 4-column image upload panel on the main page.
    Runs YOLOv8 on each uploaded image immediately.

    Returns:
        results: dict keyed 0-3:
            {
              "frame":      np.ndarray BGR | None,
              "detections": list[dict],
              "count":      int,
              "score":      float,
              "green_time": float,
              "uploaded":   bool,
              "name":       str,
            }
    """
    st.markdown(
        """
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:14px;
                    padding:16px 20px;margin-bottom:16px;">
          <div style="font-size:1rem;font-weight:700;color:#60a5fa;margin-bottom:4px;">
            📸 Upload Traffic Images — One Per Signal
          </div>
          <div style="font-size:0.78rem;color:#64748b;">
            Upload one image per signal zone. YOLOv8 analyses each image instantly
            and adjusts the green time for that signal based on detected vehicle count.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols    = st.columns(NUM_SIGNALS, gap="small")
    results = {}

    for i, col in enumerate(cols):
        with col:
            st.markdown(
                f"<div style='text-align:center;font-weight:700;color:#94a3b8;"
                f"margin-bottom:6px;'>🚦 Signal S{i+1}</div>",
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                f"S{i+1} image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key=f"sig_img_{i}",
                label_visibility="collapsed",
            )

            if uploaded is not None:
                pil   = Image.open(uploaded).convert("RGB")
                frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

                # Run detection
                dets, _ = count_vehicles(model, frame, conf) if model else ([], [])
                score   = sum(d["weight"] for d in dets)
                green_t = _calc_green(score)

                # Draw bboxes on preview
                preview = frame.copy()
                for d in dets:
                    x1, y1, x2, y2 = d["bbox"]
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (34, 197, 94), 2)
                    lbl = f"{d['class_name']} {d['conf']:.2f}"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(preview, (x1, y1 - th - 4), (x1 + tw + 2, y1),
                                  (34, 197, 94), -1)
                    cv2.putText(preview, lbl, (x1 + 1, y1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

                # Show annotated preview
                st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                         use_container_width=True)

                # Stats card
                level = ("🟢 LOW" if len(dets) <= 5 else
                         "🟡 MEDIUM" if len(dets) <= 12 else "🔴 HIGH")
                st.markdown(
                    f"""
                    <div style="background:#0c1a2e;border:1px solid #1e3a5f;
                                border-radius:10px;padding:8px;margin-top:4px;
                                font-size:0.72rem;">
                      <div style="color:#94a3b8;">Vehicles: <b style="color:#f1f5f9;">
                        {len(dets)}</b></div>
                      <div style="color:#94a3b8;">Traffic: <b>{level}</b></div>
                      <div style="color:#94a3b8;">Green time:
                        <b style="color:#22c55e;">{green_t:.1f}s</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                results[i] = {
                    "frame":      frame,
                    "detections": dets,
                    "count":      len(dets),
                    "score":      score,
                    "green_time": green_t,
                    "uploaded":   True,
                    "name":       uploaded.name,
                }
            else:
                # Placeholder when no image uploaded
                st.markdown(
                    f"""
                    <div style="background:#0f172a;border:2px dashed #1e293b;
                                border-radius:10px;padding:30px 8px;
                                text-align:center;color:#475569;font-size:0.75rem;">
                      <div style="font-size:1.8rem;">📷</div>
                      <div>Upload image<br>for S{i+1}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                results[i] = {
                    "frame":      None,
                    "detections": [],
                    "count":      0,
                    "score":      0.0,
                    "green_time": GREEN_BASE_SEC,
                    "uploaded":   False,
                    "name":       "",
                }

    return results


def apply_image_results_to_controller(results: dict, controller) -> None:
    """
    Push per-signal detection results into the controller so green times
    are set from the uploaded images before the signal cycle starts.
    """
    zone_detections = [results[i]["detections"] for i in range(NUM_SIGNALS)]

    # Manually set weighted scores so _calc_green uses image data
    for i, dets in enumerate(zone_detections):
        sig                = controller.signals[i]
        sig.detections     = dets
        sig.vehicle_count  = len(dets)
        sig.weighted_score = sum(d["weight"] for d in dets)
        sig.input_source   = "image"
        sig.source_name    = results[i]["name"]

    # Recalculate green time for the active signal
    active = controller.signals[controller.active_idx]
    from config import GREEN_BASE_SEC, GREEN_PER_VEHICLE, GREEN_MAX_SEC, GREEN_MIN_SEC
    controller.green_duration = max(
        GREEN_MIN_SEC,
        min(GREEN_MAX_SEC, GREEN_BASE_SEC + active.weighted_score * GREEN_PER_VEHICLE)
    )
    active.green_time = controller.green_duration
