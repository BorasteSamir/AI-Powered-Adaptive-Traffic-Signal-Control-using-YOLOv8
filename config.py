# config.py — All constants and shared configuration for the traffic system.

import os
from dataclasses import dataclass, field

# Absolute path to the directory where this file lives.
# All relative paths are resolved from here, regardless of where
# `streamlit run` is invoked from.
_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Vehicle classes with weighted traffic scores ──────────────────────────────
# Keys must exactly match class names in your best.pt model.
VEHICLE_CLASSES: dict[str, float] = {
    "car":          1.0,
    "bus":          2.5,
    "truck":        2.5,
    "motorcycle":   0.8,
    "autorickshaw": 1.2,
    "motorbike":    0.8,   # COCO fallback
    "auto":         1.2,   # COCO fallback
    "van":          1.5,
}

# ── Signal system ─────────────────────────────────────────────────────────────
NUM_SIGNALS      = 4
THRESHOLD_LOW    = 5     # vehicles ≤ this  → LOW
THRESHOLD_MEDIUM = 12    # vehicles ≤ this  → MEDIUM  (else HIGH)
GREEN_BASE_SEC   = 0.0   # base green (vehicle count drives the time)
GREEN_PER_VEHICLE= 2.0   # 2s per weighted vehicle unit
GREEN_MAX_SEC    = 120.0 # maximum green capped at 2 minutes
GREEN_MIN_SEC    = 8.0   # floor — never less than 8s even if zone is empty
YELLOW_SEC       = 3.0   # yellow transition duration
ALL_RED_SEC      = 1.0   # all-red clearance interval between signals

# ── Coordination (Green Wave) ─────────────────────────────────────────────────
COORDINATION_ENABLED = True   # Enable signal coordination
OFFSET_SEC           = 2.0    # Time offset between adjacent signals (green wave)
SKIP_EMPTY_SIGNALS   = True   # Skip signals with 0 vehicles to reduce congestion

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(_HERE, "best.pt")

# ── Colour palette  (R, G, B) — converted to BGR where needed ─────────────────
COLORS: dict[str, tuple[int, int, int]] = {
    "green":  ( 34, 197,  94),
    "yellow": (250, 204,  21),
    "red":    (239,  68,  68),
    "blue":   ( 59, 130, 246),
    "white":  (255, 255, 255),
    "black":  (  0,   0,   0),
    "dark":   ( 15,  23,  42),
    "gray":   (100, 116, 139),
}

SIGNAL_STATE_COLOR: dict[str, tuple[int, int, int]] = {
    "GREEN":  COLORS["green"],
    "YELLOW": COLORS["yellow"],
    "RED":    COLORS["red"],
}

# ── Data classes shared across modules ───────────────────────────────────────

@dataclass
class SignalState:
    """Live state for one traffic signal on the corridor."""
    id: int
    name: str
    state: str            = "RED"
    vehicle_count: int    = 0
    weighted_score: float = 0.0
    green_time: float     = 0.0
    traffic_level: str    = "LOW"
    detections: list      = field(default_factory=list)
    skipped: bool         = False
    wait_time: float      = 0.0
    input_source: str     = "none"   # "none" | "image" | "video" | "webcam"
    source_name: str      = ""       # uploaded filename for display


@dataclass
class SystemStats:
    """Global stats updated each processed frame."""
    total_vehicles: int    = 0
    fps: float             = 0.0
    frame_count: int       = 0
    active_signal: int     = 1
    active_state: str      = "GREEN"
    cycle_count: int       = 0
    skipped_count: int     = 0       # total signals skipped this session
    avg_wait_time: float   = 0.0     # average wait across all RED signals
    green_wave_active: bool = False  # True when coordination offset is running
