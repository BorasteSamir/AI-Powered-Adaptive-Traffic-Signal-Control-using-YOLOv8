"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         SMART TRAFFIC MANAGEMENT SYSTEM — YOLOv8 + Streamlit Dashboard      ║
║         Trained on Indian Driving Dataset (IDD)                              ║
║         Run: streamlit run smart_traffic_app.py                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import threading

# ─── Ultralytics / YOLOv8 ────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    st.error("❌ ultralytics not installed. Run: pip install ultralytics")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTS & CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# IDD class names that map to "vehicle" categories we care about.
# Adjust these names to exactly match the class names in YOUR best.pt.
VEHICLE_CLASSES = {
    "car":          1.0,   # weight for traffic score
    "bus":          2.5,
    "truck":        2.5,
    "motorcycle":   0.8,
    "autorickshaw": 1.2,
    # Common COCO fallbacks (if model uses COCO labels)
    "motorbike":    0.8,
    "auto":         1.2,
    "van":          1.5,
}

NUM_SIGNALS = 5               # S1 … S5
THRESHOLD_LOW    = 5          # vehicles ≤ this → LOW traffic
THRESHOLD_MEDIUM = 12         # vehicles ≤ this → MEDIUM traffic (else HIGH)
GREEN_BASE_SEC   = 5.0        # minimum green duration (seconds)
GREEN_PER_VEHICLE= 0.6        # extra seconds added per weighted vehicle
GREEN_MAX_SEC    = 25.0       # cap green time
YELLOW_SEC       = 1.0        # yellow transition duration
MODEL_PATH       = "best.pt"  # path to your trained YOLOv8 model

# Dashboard colour palette (used in annotations & UI)
COLORS = {
    "green":  (34,  197,  94),   # traffic green
    "yellow": (250, 204,  21),   # amber
    "red":    (239,  68,  68),   # traffic red
    "blue":   ( 59, 130, 246),   # accent
    "white":  (255, 255, 255),
    "black":  (  0,   0,   0),
    "dark":   ( 15,  23,  42),   # near-black panel
    "gray":   (100, 116, 139),
}

SIGNAL_STATE_COLOR = {
    "GREEN":  COLORS["green"],
    "YELLOW": COLORS["yellow"],
    "RED":    COLORS["red"],
}

# ──────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalState:
    """Holds the live state for one traffic signal."""
    id: int                          # 1-based index
    name: str                        # "S1" … "S5"
    state: str = "RED"               # "GREEN" | "YELLOW" | "RED"
    vehicle_count: int    = 0
    weighted_score: float = 0.0
    green_time: float     = 0.0      # allocated green duration
    traffic_level: str    = "LOW"    # LOW / MEDIUM / HIGH
    detections: list      = field(default_factory=list)  # bbox list this frame


@dataclass
class SystemStats:
    """Global stats updated each frame."""
    total_vehicles: int   = 0
    fps: float            = 0.0
    frame_count: int      = 0
    active_signal: int    = 1        # 1-based
    active_state: str     = "GREEN"  # state of active signal
    cycle_count: int      = 0        # how many full S1→S5 cycles completed


# ──────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> Optional[YOLO]:
    """
    Load the YOLOv8 model from disk.
    Cached so it loads only once per Streamlit session.
    """
    if not os.path.exists(model_path):
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  ZONE SPLITTING
# ──────────────────────────────────────────────────────────────────────────────

def split_into_zones(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Divide a frame into NUM_SIGNALS equal vertical strips.
    Returns a list of (x1, y1, x2, y2) bounding boxes, one per zone.
    """
    h, w = frame.shape[:2]
    zone_w = w // NUM_SIGNALS
    zones = []
    for i in range(NUM_SIGNALS):
        x1 = i * zone_w
        x2 = x1 + zone_w if i < NUM_SIGNALS - 1 else w  # last zone gets remainder
        zones.append((x1, 0, x2, h))
    return zones


# ──────────────────────────────────────────────────────────────────────────────
#  VEHICLE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def count_vehicles(
    model: YOLO,
    frame: np.ndarray,
    conf_threshold: float = 0.35,
) -> tuple[list[dict], list[str]]:
    """
    Run YOLOv8 inference on a frame.

    Returns:
        detections : list of dicts with keys: bbox, class_name, conf, weight
        class_names: full list of class names from the model
    """
    results = model.predict(
        frame,
        conf=conf_threshold,
        verbose=False,
        stream=False,
    )

    detections = []
    class_names = model.names  # dict {idx: name}

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id   = int(box.cls[0])
            cls_name = class_names.get(cls_id, "").lower()
            conf     = float(box.conf[0])

            # Only keep vehicle classes
            weight = VEHICLE_CLASSES.get(cls_name)
            if weight is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox":       (x1, y1, x2, y2),
                "class_name": cls_name,
                "conf":       conf,
                "weight":     weight,
            })

    return detections, list(class_names.values())


# ──────────────────────────────────────────────────────────────────────────────
#  ASSIGN DETECTIONS → ZONES
# ──────────────────────────────────────────────────────────────────────────────

def assign_to_zones(
    detections: list[dict],
    zones: list[tuple],
) -> list[list[dict]]:
    """
    Map each detection to the zone whose centre-x it falls within.
    Returns a list of lists: zone_detections[i] = detections in zone i.
    """
    zone_detections = [[] for _ in range(NUM_SIGNALS)]
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) // 2
        for i, (zx1, _, zx2, _) in enumerate(zones):
            if zx1 <= cx < zx2:
                zone_detections[i].append(det)
                break
    return zone_detections


# ──────────────────────────────────────────────────────────────────────────────
#  TRAFFIC LEVEL HELPER
# ──────────────────────────────────────────────────────────────────────────────

def traffic_level(count: int) -> str:
    if count <= THRESHOLD_LOW:
        return "LOW"
    elif count <= THRESHOLD_MEDIUM:
        return "MEDIUM"
    return "HIGH"


# ──────────────────────────────────────────────────────────────────────────────
#  SIGNAL CONTROLLER
# ──────────────────────────────────────────────────────────────────────────────

class SignalController:
    """
    Coordinates 5 traffic signals in a sequential round-robin fashion.
    Only one signal is GREEN at a time; others are RED.
    Uses wall-clock time for realistic transitions.
    """

    def __init__(self):
        self.signals: list[SignalState] = [
            SignalState(id=i + 1, name=f"S{i + 1}") for i in range(NUM_SIGNALS)
        ]
        self.active_idx: int   = 0       # index of the currently active signal
        self.phase_start: float = time.time()
        self.phase: str         = "GREEN" # "GREEN" or "YELLOW"
        self.green_duration: float = GREEN_BASE_SEC
        self.stats = SystemStats()

        # Initialise first signal as GREEN
        self.signals[0].state = "GREEN"

    # ── update signal states every frame ─────────────────────────────────────

    def update_signals(self, zone_detections: list[list[dict]]) -> None:
        """
        Update vehicle counts from latest detection results and
        advance the signal cycle based on elapsed time.
        """
        now = time.time()
        elapsed = now - self.phase_start

        # --- Update counts for ALL zones every frame -------------------------
        for i, dets in enumerate(zone_detections):
            sig = self.signals[i]
            sig.detections    = dets
            sig.vehicle_count = len(dets)
            sig.weighted_score = sum(d["weight"] for d in dets)
            sig.traffic_level  = traffic_level(sig.vehicle_count)

        active_sig = self.signals[self.active_idx]

        # --- Compute dynamic green time for the active signal ----------------
        self.green_duration = min(
            GREEN_MAX_SEC,
            GREEN_BASE_SEC + active_sig.weighted_score * GREEN_PER_VEHICLE,
        )
        active_sig.green_time = self.green_duration

        # --- State machine: GREEN → YELLOW → (next) GREEN -------------------
        if self.phase == "GREEN":
            # Check if green time expired OR traffic cleared
            cleared = active_sig.vehicle_count <= THRESHOLD_LOW
            if elapsed >= self.green_duration or (cleared and elapsed >= GREEN_BASE_SEC * 0.5):
                # Transition to YELLOW
                active_sig.state = "YELLOW"
                self.phase       = "YELLOW"
                self.phase_start = now

        elif self.phase == "YELLOW":
            if elapsed >= YELLOW_SEC:
                # Move to next signal
                active_sig.state    = "RED"
                self.active_idx     = (self.active_idx + 1) % NUM_SIGNALS
                next_sig            = self.signals[self.active_idx]
                next_sig.state      = "GREEN"
                self.phase          = "GREEN"
                self.phase_start    = now
                if self.active_idx == 0:
                    self.stats.cycle_count += 1

        # --- Sync stats -------------------------------------------------------
        self.stats.active_signal = self.signals[self.active_idx].id
        self.stats.active_state  = self.signals[self.active_idx].state
        self.stats.total_vehicles = sum(s.vehicle_count for s in self.signals)

    # ── force advance (manual override) ──────────────────────────────────────

    def force_next(self) -> None:
        """Immediately advance to next signal (for testing / manual control)."""
        self.signals[self.active_idx].state = "RED"
        self.active_idx = (self.active_idx + 1) % NUM_SIGNALS
        self.signals[self.active_idx].state = "GREEN"
        self.phase      = "GREEN"
        self.phase_start = time.time()


# ──────────────────────────────────────────────────────────────────────────────
#  FRAME ANNOTATION (OpenCV)
# ──────────────────────────────────────────────────────────────────────────────

def draw_dashboard(
    frame: np.ndarray,
    signals: list[SignalState],
    zones:   list[tuple],
    stats:   SystemStats,
) -> np.ndarray:
    """
    Annotate the video frame with:
      • Zone dividers + signal state overlays
      • Bounding boxes per detection
      • Per-zone vehicle count badge
      • FPS counter
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    # ── Draw zone overlays ───────────────────────────────────────────────────
    for i, (zx1, zy1, zx2, zy2) in enumerate(zones):
        sig   = signals[i]
        color = SIGNAL_STATE_COLOR[sig.state]  # (R,G,B) → need BGR for cv2
        bgr   = (color[2], color[1], color[0])

        # Transparent tinted overlay
        alpha = 0.08 if sig.state == "RED" else 0.18
        overlay = vis.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), bgr, -1)
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

        # Zone border
        thickness = 3 if sig.state != "RED" else 1
        cv2.rectangle(vis, (zx1, zy1), (zx2, zy2), bgr, thickness)

        # ── Signal label pill at top of zone ─────────────────────────────────
        label    = sig.name
        pill_x   = zx1 + 6
        pill_y   = 6
        pill_w   = 52
        pill_h   = 28
        cv2.rectangle(vis, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), bgr, -1)
        cv2.putText(vis, label, (pill_x + 6, pill_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

        # ── Vehicle count badge ───────────────────────────────────────────────
        count_str = str(sig.vehicle_count)
        badge_x   = zx1 + 6
        badge_y   = zy2 - 40
        cv2.rectangle(vis, (badge_x, badge_y), (badge_x + 44, badge_y + 28),
                       (20, 20, 30), -1)
        cv2.putText(vis, count_str, (badge_x + 6, badge_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, bgr, 2, cv2.LINE_AA)

        # ── Draw detections in this zone ──────────────────────────────────────
        for det in sig.detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), bgr, 2)
            label_text = f"{det['class_name']} {det['conf']:.2f}"
            # Small label above bbox
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), bgr, -1)
            cv2.putText(vis, label_text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # ── FPS overlay ──────────────────────────────────────────────────────────
    fps_text = f"FPS: {stats.fps:.1f}  |  Vehicles: {stats.total_vehicles}  |  Cycle: {stats.cycle_count}"
    cv2.rectangle(vis, (0, h - 34), (w, h), (10, 10, 20), -1)
    cv2.putText(vis, fps_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 1, cv2.LINE_AA)

    return vis


# ──────────────────────────────────────────────────────────────────────────────
#  STREAMLIT UI HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def signal_light_html(state: str, size: int = 32) -> str:
    """Return an HTML circle for the given signal state."""
    palette = {"GREEN": "#22c55e", "YELLOW": "#facc15", "RED": "#ef4444"}
    dim     = {"GREEN": "#14532d", "YELLOW": "#713f12", "RED": "#7f1d1d"}
    c       = palette.get(state, "#334155")
    d       = dim.get(state, "#334155")
    glow    = f"box-shadow: 0 0 14px 4px {c};" if state != "RED" else ""
    return (
        f'<span style="display:inline-block;width:{size}px;height:{size}px;'
        f'border-radius:50%;background:{c};{glow}'
        f'border:2px solid {d};vertical-align:middle;"></span>'
    )


def traffic_badge_html(level: str) -> str:
    colors = {
        "LOW":    ("#bbf7d0", "#166534"),
        "MEDIUM": ("#fef9c3", "#854d0e"),
        "HIGH":   ("#fecaca", "#991b1b"),
    }
    bg, fg = colors.get(level, ("#e2e8f0", "#334155"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:12px;font-size:0.75rem;font-weight:700;">{level}</span>'
    )


def render_signal_panel(signals: list[SignalState]) -> None:
    """Render the 5-signal panel using Streamlit columns."""
    cols = st.columns(NUM_SIGNALS)
    for i, col in enumerate(cols):
        sig = signals[i]
        with col:
            # Signal light
            st.markdown(
                f"""
                <div style="
                    background:#0f172a;border-radius:14px;padding:14px 10px;
                    text-align:center;border:1px solid #1e293b;
                ">
                  <div style="font-size:1rem;font-weight:700;
                              color:#94a3b8;margin-bottom:8px;">{sig.name}</div>
                  <div style="margin-bottom:8px;">
                    {signal_light_html("RED"    if sig.state != "GREEN"  else "GREEN",  36)}
                  </div>
                  <div style="margin-bottom:4px;">
                    {signal_light_html("YELLOW" if sig.state == "YELLOW" else "YELLOW", 28) if sig.state == "YELLOW"
                     else signal_light_html("dim",28).replace("dim","#334155")}
                  </div>
                  <div style="margin-bottom:10px;">
                    {signal_light_html("RED" if sig.state != "GREEN" else "dim", 36).replace("dim","#334155")}
                  </div>
                  <div style="font-size:1.4rem;font-weight:800;color:#f1f5f9;">
                    {sig.vehicle_count}
                  </div>
                  <div style="font-size:0.7rem;color:#64748b;margin-bottom:6px;">vehicles</div>
                  {traffic_badge_html(sig.traffic_level)}
                  {'<div style="font-size:0.65rem;color:#22c55e;margin-top:6px;">⏱ ' + f'{sig.green_time:.0f}s' + '</div>' if sig.state == "GREEN" else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metric_bar(stats: SystemStats) -> None:
    """Top metric row."""
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🚦 Active Signal", f"S{stats.active_signal}", stats.active_state)
    m2.metric("🚗 Total Vehicles", stats.total_vehicles)
    m3.metric("📡 FPS", f"{stats.fps:.1f}")
    m4.metric("🔄 Cycles", stats.cycle_count)


# ──────────────────────────────────────────────────────────────────────────────
#  FULL SIGNAL PANEL (with proper traffic light bulbs)
# ──────────────────────────────────────────────────────────────────────────────

def render_full_signal_panel(signals: list[SignalState]) -> None:
    """
    Render a realistic 3-bulb traffic light for each signal.
    Each bulb (R/Y/G) is lit or dimmed based on the current state.
    """
    col_list = st.columns(NUM_SIGNALS)
    for i, col in enumerate(col_list):
        sig = signals[i]
        is_green  = sig.state == "GREEN"
        is_yellow = sig.state == "YELLOW"
        is_red    = sig.state == "RED"

        red_clr    = "#ef4444" if is_red    else "#3b0d0d"
        yellow_clr = "#facc15" if is_yellow else "#3b2f00"
        green_clr  = "#22c55e" if is_green  else "#0a2e18"

        red_glow    = "box-shadow:0 0 16px 6px #ef4444;" if is_red    else ""
        yellow_glow = "box-shadow:0 0 16px 6px #facc15;" if is_yellow else ""
        green_glow  = "box-shadow:0 0 16px 6px #22c55e;" if is_green  else ""

        green_time_info = (
            f'<div style="color:#22c55e;font-size:0.7rem;margin-top:6px;">⏱ {sig.green_time:.0f}s green</div>'
            if is_green else ""
        )

        with col:
            st.markdown(
                f"""
                <div style="background:#0f172a;border-radius:16px;padding:16px 8px;
                            text-align:center;border:1px solid #1e293b;min-height:230px;">
                  <div style="font-size:1rem;font-weight:700;color:#94a3b8;margin-bottom:10px;">{sig.name}</div>

                  <div style="display:inline-block;background:#1e293b;border-radius:12px;
                              padding:10px 14px;border:2px solid #334155;">
                    <!-- RED -->
                    <div style="width:34px;height:34px;border-radius:50%;
                                background:{red_clr};{red_glow}margin:0 auto 8px;"></div>
                    <!-- YELLOW -->
                    <div style="width:34px;height:34px;border-radius:50%;
                                background:{yellow_clr};{yellow_glow}margin:0 auto 8px;"></div>
                    <!-- GREEN -->
                    <div style="width:34px;height:34px;border-radius:50%;
                                background:{green_clr};{green_glow}margin:0 auto;"></div>
                  </div>

                  <div style="margin-top:12px;font-size:1.5rem;font-weight:800;color:#f1f5f9;">
                    {sig.vehicle_count}
                  </div>
                  <div style="font-size:0.68rem;color:#64748b;">vehicles</div>
                  <div style="margin-top:6px;">{traffic_badge_html(sig.traffic_level)}</div>
                  {green_time_info}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
#  HISTORY CHART DATA
# ──────────────────────────────────────────────────────────────────────────────

def update_history(
    history: dict,
    signals: list[SignalState],
    max_len: int = 60,
) -> dict:
    """Append current counts to rolling history for sparklines."""
    for sig in signals:
        history[sig.name].append(sig.vehicle_count)
        if len(history[sig.name]) > max_len:
            history[sig.name].pop(0)
    return history


# ──────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE FRAME  (core pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def process_frame(
    model: YOLO,
    frame: np.ndarray,
    controller: SignalController,
    conf_threshold: float,
) -> np.ndarray:
    """
    Full pipeline for a single frame:
      1. Detect vehicles
      2. Assign to zones
      3. Update signal controller
      4. Draw annotated frame
    """
    zones             = split_into_zones(frame)
    detections, _     = count_vehicles(model, frame, conf_threshold)
    zone_detections   = assign_to_zones(detections, zones)
    controller.update_signals(zone_detections)
    annotated         = draw_dashboard(frame, controller.signals, zones, controller.stats)
    return annotated


# ──────────────────────────────────────────────────────────────────────────────
#  STREAMLIT APP  (main entry point)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── Page config ──────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Smart Traffic Management System",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;600&display=swap');

          html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0a0f1e;
            color: #e2e8f0;
          }
          h1, h2, h3 { font-family: 'Rajdhani', sans-serif; letter-spacing:0.5px; }

          /* Sidebar */
          section[data-testid="stSidebar"] { background:#0f172a; }
          section[data-testid="stSidebar"] .stMarkdown { color:#94a3b8; }

          /* Metric cards */
          div[data-testid="metric-container"] {
            background:#0f172a;border:1px solid #1e293b;
            border-radius:12px;padding:12px;
          }
          div[data-testid="metric-container"] label { color:#64748b !important; }
          div[data-testid="metric-container"] div[data-testid="metric-value"] {
            color:#f1f5f9 !important; font-family:'Rajdhani',sans-serif;font-size:1.8rem;
          }

          /* Buttons */
          .stButton > button {
            background:#1e3a5f;color:#60a5fa;border:1px solid #1d4ed8;
            border-radius:8px;font-weight:600;
          }
          .stButton > button:hover { background:#1d4ed8;color:#fff; }

          /* Sliders */
          .stSlider label { color:#94a3b8; }

          /* Divider */
          hr { border-color:#1e293b; }

          /* Video frame */
          img { border-radius:10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:4px;">
          <span style="font-size:2.4rem;">🚦</span>
          <div>
            <h1 style="margin:0;font-size:2rem;background:linear-gradient(90deg,#60a5fa,#34d399);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
              Smart Traffic Management System
            </h1>
            <p style="margin:0;color:#64748b;font-size:0.85rem;">
              YOLOv8 · Indian Driving Dataset · 5-Signal Simulation
            </p>
          </div>
        </div>
        <hr style="margin:10px 0 20px;">
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar: configuration ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        model_path = st.text_input("Model path", value=MODEL_PATH)

        st.markdown("---")
        st.markdown("### 📥 Input Source")
        input_mode = st.radio("Source", ["Upload Video", "Webcam"], index=0)

        uploaded_file = None
        use_webcam    = False

        if input_mode == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload a video file",
                type=["mp4", "avi", "mov", "mkv"],
            )
        else:
            use_webcam = True

        st.markdown("---")
        st.markdown("### 🎛️ Detection Settings")
        conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)

        st.markdown("---")
        st.markdown("### 💾 Recording")
        save_video = st.checkbox("Save processed video", value=False)

        st.markdown("---")
        st.markdown("### 🔧 Manual Control")
        force_next_btn = st.button("⏭ Force Next Signal")

        st.markdown("---")
        st.markdown(
            "<small style='color:#475569;'>Trained on IDD · ultralytics YOLOv8</small>",
            unsafe_allow_html=True,
        )

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(model_path)
    if model is None:
        st.warning(
            f"⚠️ Model file **{model_path}** not found. "
            "Place your `best.pt` in the same directory as this script, "
            "or change the path in the sidebar."
        )
        st.info("The dashboard will show signal simulation without live detection if no model is loaded.")

    # ── Session state ─────────────────────────────────────────────────────────
    if "controller" not in st.session_state:
        st.session_state.controller = SignalController()
    if "history" not in st.session_state:
        st.session_state.history = {f"S{i+1}": [] for i in range(NUM_SIGNALS)}
    if "running" not in st.session_state:
        st.session_state.running = False
    if "fps_times" not in st.session_state:
        st.session_state.fps_times = deque(maxlen=30)

    controller: SignalController = st.session_state.controller

    if force_next_btn:
        controller.force_next()
        st.rerun()

    # ── Layout: video | signal panel ─────────────────────────────────────────
    video_col, info_col = st.columns([3, 1], gap="medium")

    with video_col:
        st.markdown("#### 📹 Live Feed")
        video_placeholder = st.empty()

    with info_col:
        st.markdown("#### 📊 Traffic Analytics")
        analytics_placeholder = st.empty()

    st.markdown("---")

    # ── Signal panel ─────────────────────────────────────────────────────────
    st.markdown("#### 🚦 Signal Control Panel")
    signal_placeholder = st.empty()

    # ── Metrics bar ──────────────────────────────────────────────────────────
    st.markdown("---")
    metrics_placeholder = st.empty()

    # ── Chart ─────────────────────────────────────────────────────────────────
    st.markdown("#### 📈 Vehicle Count History (last 60 frames)")
    chart_placeholder = st.empty()

    # ── Start/Stop controls ───────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, _ = st.columns([1, 1, 4])
    with ctrl_col1:
        start_btn = st.button("▶ Start", type="primary")
    with ctrl_col2:
        stop_btn  = st.button("⏹ Stop")

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    # ── Processing loop ───────────────────────────────────────────────────────
    if st.session_state.running:

        # Determine video source
        cap = None
        tmp_path = None
        writer = None

        if use_webcam:
            cap = cv2.VideoCapture(0)
        elif uploaded_file is not None:
            # Write to temp file so OpenCV can read it
            suffix = os.path.splitext(uploaded_file.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
        else:
            st.warning("Please upload a video or enable webcam first.")
            st.session_state.running = False
            st.stop()

        if not cap.isOpened():
            st.error("❌ Could not open video source.")
            st.session_state.running = False
            st.stop()

        # Optional: VideoWriter
        if save_video and tmp_path:
            out_path = tmp_path.replace(os.path.splitext(tmp_path)[-1], "_processed.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_out = cap.get(cv2.CAP_PROP_FPS) or 25
            width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer  = cv2.VideoWriter(out_path, fourcc, fps_out, (width, height))

        frame_idx = 0

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                # Video ended → loop or stop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                controller.stats.cycle_count += 1
                continue

            t0 = time.time()

            # ── Process frame ────────────────────────────────────────────────
            if model is not None:
                annotated = process_frame(model, frame, controller, conf_threshold)
            else:
                # No model: just split zones + simulate signal advancement
                zones           = split_into_zones(frame)
                zone_detections = [[] for _ in range(NUM_SIGNALS)]
                controller.update_signals(zone_detections)
                annotated = draw_dashboard(frame, controller.signals, zones, controller.stats)

            # ── FPS ──────────────────────────────────────────────────────────
            t1 = time.time()
            st.session_state.fps_times.append(1.0 / max(t1 - t0, 1e-6))
            controller.stats.fps = float(np.mean(st.session_state.fps_times))
            controller.stats.frame_count += 1

            # ── Write frame ───────────────────────────────────────────────────
            if writer is not None:
                writer.write(annotated)

            # ── Convert to RGB for Streamlit ──────────────────────────────────
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_container_width=True)

            # ── Update history ────────────────────────────────────────────────
            st.session_state.history = update_history(
                st.session_state.history, controller.signals
            )

            # ── Render signal panel ───────────────────────────────────────────
            with signal_placeholder.container():
                render_full_signal_panel(controller.signals)

            # ── Render metrics ────────────────────────────────────────────────
            with metrics_placeholder.container():
                render_metric_bar(controller.stats)

            # ── Side analytics ────────────────────────────────────────────────
            with analytics_placeholder.container():
                for sig in controller.signals:
                    icon = "🟢" if sig.state == "GREEN" else ("🟡" if sig.state == "YELLOW" else "🔴")
                    st.markdown(
                        f"{icon} **{sig.name}** — {sig.vehicle_count} vehicles "
                        f"| {traffic_badge_html(sig.traffic_level)}",
                        unsafe_allow_html=True,
                    )

            # ── Chart ─────────────────────────────────────────────────────────
            if frame_idx % 5 == 0:  # update chart every 5 frames to reduce load
                chart_data = {
                    k: v for k, v in st.session_state.history.items()
                    if len(v) > 0
                }
                if chart_data:
                    import pandas as pd
                    max_len = max(len(v) for v in chart_data.values())
                    padded  = {k: ([0] * (max_len - len(v))) + v for k, v in chart_data.items()}
                    df      = pd.DataFrame(padded)
                    chart_placeholder.line_chart(df, height=160)

            frame_idx += 1

            # ── Small sleep to prevent UI overload ────────────────────────────
            time.sleep(0.01)

        # ── Cleanup ───────────────────────────────────────────────────────────
        cap.release()
        if writer is not None:
            writer.release()
            st.success(f"✅ Processed video saved to `{out_path}`")
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    else:
        # ── Static placeholder when not running ───────────────────────────────
        with signal_placeholder.container():
            render_full_signal_panel(controller.signals)
        with metrics_placeholder.container():
            render_metric_bar(controller.stats)
        video_placeholder.markdown(
            """
            <div style="background:#0f172a;border:2px dashed #1e293b;border-radius:12px;
                        padding:80px 20px;text-align:center;color:#475569;">
              <div style="font-size:3rem;margin-bottom:12px;">🎥</div>
              <div style="font-size:1rem;">Upload a video and click <b>▶ Start</b> to begin</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
