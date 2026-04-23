
# Smart Traffic Management System — Solution Document

> **Project:** AI-Powered Adaptive Traffic Signal Control using YOLOv8  
> **Dataset:** Indian Driving Dataset (IDD)  
> **Stack:** Python · YOLOv8 · Streamlit · OpenCV  
> **Run:** `streamlit run app.py`

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Proposed Solution](#2-proposed-solution)
3. [System Architecture](#3-system-architecture)
4. [Module Breakdown](#4-module-breakdown)
5. [Core Concepts Used](#5-core-concepts-used)
6. [Signal Timing Logic](#6-signal-timing-logic)
7. [Corridor Coordination Features](#7-corridor-coordination-features)
8. [Dashboard & UI Design](#8-dashboard--ui-design)
9. [Current Gaps & Limitations](#9-current-gaps--limitations)
10. [Future Improvements](#10-future-improvements)
11. [Tech Stack & Dependencies](#11-tech-stack--dependencies)
12. [Project File Structure](#12-project-file-structure)

---

## 1. Problem Statement

### Background

Urban traffic congestion is one of the most critical infrastructure challenges in Indian cities. Traditional traffic signal systems operate on **fixed pre-timed cycles** — every signal gets the same green duration regardless of actual traffic conditions. This leads to:

- Vehicles waiting at a red signal while the green lane is empty
- Emergency vehicles stuck in congestion due to rigid signal cycles
- Fuel wastage and increased carbon emissions from unnecessary idling
- No real-time adaptation to sudden traffic surges (accidents, events, rush hours)

### Specific Problem

On a **single arterial road corridor** with 4 intersections (signals S1–S4):

```
[S1] ──── road ──── [S2] ──── road ──── [S3] ──── road ──── [S4]
```

- All 4 signals operate independently with fixed timers
- No communication between signals
- No awareness of vehicle density at each intersection
- A signal with 0 vehicles holds green for the same duration as one with 30 vehicles
- Drivers experience unpredictable and unfair wait times

### Goal

Build a **real-time adaptive traffic management system** that:
1. Detects and counts vehicles at each signal using computer vision
2. Allocates green time proportional to vehicle density
3. Coordinates all 4 signals to minimise total corridor wait time
4. Provides a live dashboard for monitoring and manual override

---

## 2. Proposed Solution

### Approach

Replace fixed-timer signals with an **AI-driven adaptive signal controller** that:

1. Uses a **YOLOv8 model** (trained on IDD) to detect vehicles in each zone
2. Computes a **weighted traffic score** per zone (heavy vehicles count more)
3. Allocates **dynamic green time** based on the score: `green = vehicles × 2.0s`
4. Runs a **coordinated round-robin cycle** — only one signal is GREEN at a time
5. Implements **corridor coordination** (green wave, skip-empty, starvation prevention)
6. Displays a **live Streamlit dashboard** with real-time countdowns, signal states, and analytics

### Key Innovation

Unlike simple vehicle-count systems, this solution uses:
- **Weighted scoring** — a bus or truck contributes 2.5× more to green time than a car
- **Per-signal image input** — operators can upload a traffic photo for each signal to pre-configure timing before the cycle starts
- **Live countdown timers** — every signal card shows a draining progress bar so operators always know exactly when the next switch will happen
- **Starvation prevention** — no signal waits more than 60 seconds regardless of vehicle count

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE (Streamlit)                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  📸 Per-Signal Image Upload (S1 S2 S3 S4)                   │   │
│  │  YOLOv8 runs on each image → vehicle count → green time     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  🚦 Signal Panel  (auto-ticking every 1s via st.fragment)   │   │
│  │  S1[🟢 14.2s] S2[🔴 wait 8s] S3[🔴 wait 22s] S4[🔴 wait]  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  🎛️ Control Panel  AUTO ↔ MANUAL  |  ⏭️ Force Next          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐   │
│  │  📹 Live Video Feed  │  │  📊 Analytics + 📈 History Chart │   │
│  └──────────────────────┘  └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────▼───────────────┐
              │       SignalController         │
              │  STARTUP → GREEN → YELLOW →   │
              │  ALL_RED → next GREEN          │
              │  (wall-clock timer, strict)    │
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │         YOLOv8 (best.pt)       │
              │  Detects: car, bus, truck,     │
              │  motorcycle, autorickshaw      │
              └───────────────────────────────┘
```

---

## 4. Module Breakdown

### `config.py` — Configuration & Data Classes

**Responsibility:** Single source of truth for all constants and shared data structures.

**Key contents:**
- `VEHICLE_CLASSES` — dict mapping class names to traffic weight scores
- Signal timing constants (`GREEN_MIN_SEC=8`, `GREEN_PER_VEHICLE=2.0`, `GREEN_MAX_SEC=120`)
- `SignalState` dataclass — holds live state for one signal (state, vehicle_count, weighted_score, green_time, wait_time, skipped, input_source)
- `SystemStats` dataclass — global stats (total_vehicles, fps, cycle_count, avg_wait_time, skipped_count)
- Coordination flags (`COORDINATION_ENABLED`, `SKIP_EMPTY_SIGNALS`, `OFFSET_SEC`)

**Design decision:** All modules import from `config.py` only — no circular imports, no magic numbers scattered across files.

---

### `model_loader.py` — YOLOv8 Model Loading

**Responsibility:** Load and cache the trained YOLOv8 model.

**Key function:**
```python
@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> Optional[YOLO]
```

**Design decision:** `@st.cache_resource` ensures the model (≈50MB) is loaded only once per Streamlit session and reused across all reruns. Without this, the model would reload on every user interaction.

---

### `detection.py` — Vehicle Detection

**Responsibility:** Run YOLOv8 inference on a frame and return only vehicle detections.

**Key function:**
```python
def count_vehicles(model, frame, conf_threshold) -> (detections, class_names)
```

**Output per detection:**
```python
{
  "bbox":       (x1, y1, x2, y2),
  "class_name": "car",
  "conf":       0.87,
  "weight":     1.0,   # from VEHICLE_CLASSES
}
```

**Filtering:** Only classes present in `VEHICLE_CLASSES` are returned. Pedestrians, animals, and other objects are ignored.

---

### `zones.py` — Frame Zone Splitting

**Responsibility:** Divide a video frame into 4 equal vertical strips, one per signal.

**Key functions:**
```python
def split_into_zones(frame) -> list of (x1, y1, x2, y2)
def assign_to_zones(detections, zones) -> list of lists
```

**Zone assignment logic:** Each detection is assigned to the zone that contains the horizontal centre of its bounding box (`cx = (x1 + x2) // 2`). This handles vehicles that span zone boundaries correctly.

---

### `signal_controller.py` — Signal State Machine

**Responsibility:** The core intelligence — manages all 4 signals, enforces timing rules, implements coordination.

**Phase sequence (AUTO mode):**
```
STARTUP (1s all-red settle)
    ↓
GREEN  (adaptive duration: 8s–120s based on vehicle weight)
    ↓
YELLOW (3s fixed)
    ↓
ALL_RED (1s safety clearance)
    ↓
next signal GREEN
```

**Key methods:**
| Method | Purpose |
|---|---|
| `update_signals(zone_detections)` | Called every frame/second — refreshes counts, advances state machine |
| `set_auto()` | Switch to AUTO mode, restart timer |
| `set_manual()` | Freeze timer, operator controls |
| `set_signal(idx)` | MANUAL: make signal idx GREEN immediately |
| `force_next()` | Skip current signal, advance to next (both modes) |
| `_next_idx(current)` | Choose next signal with starvation prevention + empty-skip |
| `_start_green(now)` | Begin GREEN phase, lock green duration from current vehicle weight |
| `_calc_green(score)` | `max(8, min(120, score × 2.0))` |

**Public properties for UI:**
| Property | Returns |
|---|---|
| `remaining` | Seconds left in current GREEN phase |
| `yellow_remaining` | Seconds left in YELLOW phase |
| `progress` | 0.0→1.0 drain fraction for progress bar |
| `elapsed` | Seconds elapsed in current phase |

---

### `per_signal_input.py` — Per-Signal Image Upload

**Responsibility:** Render a 4-column image upload panel on the main page. Run YOLOv8 on each uploaded image and return per-signal detection results.

**Key functions:**
```python
def render_image_upload_panel(model, conf) -> dict
def apply_image_results_to_controller(results, controller) -> None
```

**Flow:**
1. User uploads one image per signal (S1–S4)
2. YOLOv8 runs immediately on each image
3. Bounding boxes drawn on preview thumbnail
4. Vehicle count, traffic level, and calculated green time shown per signal
5. `apply_image_results_to_controller()` pushes detections into the controller
6. Signal cycle starts using image-derived green times

---

### `dashboard.py` — Visualisation & UI Components

**Responsibility:** All visual output — OpenCV frame annotation and Streamlit HTML components.

**Key functions:**
| Function | Purpose |
|---|---|
| `draw_dashboard()` | Annotate video frame with zone overlays, bboxes, count badges, FPS bar |
| `render_full_signal_panel()` | 4 traffic light cards with live countdown bars |
| `render_control_panel()` | AUTO/MANUAL toggle + timing info + Force Next button |
| `render_metric_bar()` | 6 metric cards (active signal, vehicles, FPS, cycles, skipped, avg wait) |
| `render_analytics_sidebar()` | Per-signal icon + count + traffic badge list |
| `render_chart()` | Rolling 60-frame vehicle count line chart |
| `update_history()` | Append current counts to rolling history dict |
| `apply_custom_css()` | Inject dark-theme CSS (fonts, metric cards, buttons) |
| `traffic_badge_html()` | Coloured LOW/MEDIUM/HIGH pill badge |

**Signal card states:**
| State | Visual |
|---|---|
| 🟢 GREEN | Draining progress bar, colour shifts green→yellow→red as time runs out |
| 🟡 YELLOW | Yellow draining bar with countdown |
| ⬛ ALL_RED | Grey flash with "CLEARING" text |
| 🔴 RED | Climbing wait timer in red |

---

### `app.py` — Main Streamlit Entry Point

**Responsibility:** Wire all modules together, manage session state, run the processing loop.

**Key sections:**
1. `@st.fragment(run_every=1)` — `_live_signal_panel()` auto-ticks every second for live countdown
2. Per-signal image upload panel (Section 1 — always visible)
3. Live signal panel with countdown (Section 2 — auto-ticking)
4. Optional video/webcam feed (Section 3 — Start/Stop controlled)

**Session state keys:**
| Key | Type | Purpose |
|---|---|---|
| `controller` | `SignalController` | Persists signal state across reruns |
| `history` | `dict[str, list]` | Rolling vehicle count history per signal |
| `running` | `bool` | Video loop on/off |
| `fps_times` | `deque(30)` | Rolling FPS calculation |
| `control_mode` | `str` | "AUTO" or "MANUAL" |
| `img_results` | `dict` | Last per-signal image detection results |

---

## 5. Core Concepts Used

### Computer Vision
- **Object Detection** — YOLOv8 (You Only Look Once v8) single-pass detector
- **Bounding Box Regression** — predicts (x1, y1, x2, y2) for each detected object
- **Confidence Thresholding** — detections below `conf_threshold` (default 0.35) are discarded
- **Non-Maximum Suppression** — handled internally by ultralytics to remove duplicate boxes
- **BGR/RGB conversion** — OpenCV uses BGR; Streamlit and PIL use RGB

### Deep Learning
- **Transfer Learning** — YOLOv8 pre-trained on COCO, fine-tuned on IDD
- **Inference mode** — `model.predict()` with `verbose=False` for silent inference
- **Model caching** — `@st.cache_resource` prevents repeated model loading

### Traffic Engineering
- **Adaptive Signal Control** — green time proportional to traffic demand
- **Weighted Vehicle Scoring** — heavy vehicles (bus, truck) contribute more to congestion
- **Round-Robin Scheduling** — fair sequential service across all signals
- **Green Wave Coordination** — offset timing so a platoon released by S(n) arrives at S(n+1) on green
- **Starvation Prevention** — any signal waiting >60s is force-promoted
- **All-Red Clearance Interval** — 1s safety gap between every signal change
- **Yellow Transition** — 3s amber phase warns drivers before red

### Software Engineering
- **Dataclasses** — `SignalState` and `SystemStats` for typed, structured state
- **State Machine** — explicit phase transitions (STARTUP → GREEN → YELLOW → ALL_RED)
- **Wall-Clock Timing** — `time.time()` for real-world accurate phase durations
- **Modular Architecture** — 7 single-responsibility modules with clean imports
- **Session State** — Streamlit `st.session_state` for persistence across reruns
- **Fragment Reruns** — `@st.fragment(run_every=1)` for independent 1-second UI ticks

---

## 6. Signal Timing Logic

### Green Time Formula

```
green_time = max(GREEN_MIN_SEC, min(GREEN_MAX_SEC, weighted_score × GREEN_PER_VEHICLE))

where:
  GREEN_MIN_SEC    = 8.0   seconds  (floor — always at least 8s)
  GREEN_PER_VEHICLE= 2.0   seconds per weighted vehicle unit
  GREEN_MAX_SEC    = 120.0 seconds  (cap — never more than 2 minutes)
```

### Vehicle Weights

| Vehicle Type | Weight | Reason |
|---|---|---|
| car | 1.0 | Standard reference |
| bus | 2.5 | Occupies 2.5× road space |
| truck | 2.5 | Same as bus |
| autorickshaw | 1.2 | Slightly more than car |
| van | 1.5 | Larger than car |
| motorcycle | 0.8 | Smaller, less congestion impact |

### Timing Table

| Vehicles (all cars) | Weighted Score | Green Time |
|---|---|---|
| 0–3 | 0–3 | 8s (minimum) |
| 5 | 5.0 | 10s |
| 7 | 7.0 | **14s** |
| 10 | 10.0 | 20s |
| 15 | 15.0 | **30s** |
| 20 | 20.0 | 40s |
| 30 | 30.0 | 60s |
| 40 | 40.0 | 80s |
| 60+ | 60+ | **120s (2 min cap)** |

### Phase Durations

| Phase | Duration | Purpose |
|---|---|---|
| STARTUP | 1s | All-red settle before first green |
| GREEN | 8s–120s | Adaptive, based on vehicle weight |
| YELLOW | 3s | Amber warning to drivers |
| ALL_RED | 1s | Safety clearance between signals |

---

## 7. Corridor Coordination Features

### 1. Skip Empty Signals (`SKIP_EMPTY_SIGNALS = True`)
If a zone has 0 vehicles, it is skipped in the round-robin. The cycle moves to the next non-empty signal. This prevents unnecessary waiting when a lane is clear.

### 2. Starvation Prevention (`MAX_WAIT_SEC = 60`)
Any signal that has been RED for more than 60 seconds is force-promoted to GREEN regardless of vehicle count. This ensures fairness and prevents indefinite waiting.

### 3. Green Wave Offset (`OFFSET_SEC = 2.0`)
When S(n) turns GREEN, S(n+1) gets a 2-second head-start on its phase clock. This means a platoon of vehicles released by S1 will arrive at S2 just as it turns GREEN — reducing stop-and-go behaviour along the corridor.

### 4. Downstream Congestion Trimming
If the next signal (S(n+1)) has HIGH traffic density, the current signal's green time is trimmed by 20%. This slows the release of vehicles into an already congested zone.

### 5. All-Red Clearance
A 1-second all-red interval between every signal change ensures vehicles from the previous green phase have cleared the intersection before the next signal turns green.

---

## 8. Dashboard & UI Design

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  📸 Traffic Image Analysis — Per Signal                     │
│  [S1 upload] [S2 upload] [S3 upload] [S4 upload]           │
│  [preview]   [preview]   [preview]   [preview]              │
│  7 veh/14s   2 veh/8s    15 veh/30s  0 veh/8s              │
├─────────────────────────────────────────────────────────────┤
│  🚦 Signal Control Panel                                    │
│  [S1 🟢 14.2s▓▓▓▓▓░░] [S2 🔴 wait 8s] [S3 🔴] [S4 🔴]    │
│  ─────────────────────────────────────────────────────────  │
│  🎛️ AUTO MODE  [🤖 Auto] [🕹️ Manual]  [⏭️ Force Next]      │
│  Active: S1 | Phase: GREEN | Allotted: 14s | Remaining: 9s │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
├─────────────────────────────────────────────────────────────┤
│  🚦 S1  🚗 18  📡 12.4  🔄 2  ⏭️ 3  ⏱️ 14.2s              │
├─────────────────────────────────────────────────────────────┤
│  📈 Vehicle Count History (last 60 frames)                  │
│  [line chart — S1 S2 S3 S4]                                 │
└─────────────────────────────────────────────────────────────┘
```

### Live Countdown Mechanism

The signal panel uses `@st.fragment(run_every=1)` — a Streamlit feature that rerenders only the fragment every 1 second without triggering a full page reload. This gives smooth live countdowns without performance overhead.

### Dark Theme

Custom CSS injects a dark navy theme (`#0a0f1e` background) with the Rajdhani and Inter fonts, styled metric cards, and glowing signal bulbs.

---

## 9. Current Gaps & Limitations

### Detection Gaps

| Gap | Description |
|---|---|
| Static image analysis | Per-signal images are uploaded once; they do not update dynamically as traffic changes |
| No vehicle tracking | Vehicles are counted per frame but not tracked across frames — the same vehicle may be counted multiple times |
| Zone boundary ambiguity | Vehicles at zone edges may be assigned to the wrong signal |
| Night/low-light performance | YOLOv8 accuracy degrades in poor lighting without specific training data |
| Occlusion handling | Heavily occluded vehicles (behind trucks) may not be detected |

### Signal Logic Gaps

| Gap | Description |
|---|---|
| No real sensor integration | The system simulates signals; it does not connect to actual traffic light hardware |
| Fixed corridor direction | The system assumes a single-direction corridor; bidirectional or multi-lane roads are not modelled |
| No emergency vehicle priority | Ambulances and fire trucks are not given automatic signal preemption |
| No pedestrian phase | There is no dedicated pedestrian crossing phase in the signal cycle |
| Weather adaptation | Rain, fog, or glare affecting camera feeds is not handled |

### System Gaps

| Gap | Description |
|---|---|
| Single camera assumption | Each signal zone is a vertical slice of one camera frame; real intersections need separate cameras |
| No historical learning | The system does not learn from past traffic patterns to predict future demand |
| No database logging | Signal events, vehicle counts, and wait times are not persisted to a database |
| No alert system | There is no notification when a signal is stuck, a zone is overloaded, or the system detects an anomaly |

---

## 10. Future Improvements

### Short-Term (Next Sprint)

- **Vehicle tracking with DeepSORT** — track vehicles across frames to avoid double-counting and measure queue length more accurately
- **Database logging** — store per-cycle signal events and vehicle counts in SQLite for historical analysis
- **Alert system** — notify operator when avg wait time exceeds threshold or a signal has been skipped too many times
- **Webcam per signal** — replace single-frame images with live webcam feeds per signal zone

### Medium-Term

- **Emergency vehicle preemption** — detect ambulance/fire truck class and immediately grant green to that signal
- **Pedestrian phase** — add a dedicated all-red pedestrian crossing interval after every N cycles
- **Multi-lane support** — split each zone into lanes and count per-lane density
- **REST API** — expose signal states and vehicle counts via FastAPI for integration with city dashboards

### Long-Term

- **Reinforcement Learning controller** — replace the rule-based state machine with a trained RL agent that optimises total corridor throughput
- **Hardware integration** — connect to real traffic light controllers via MQTT or RS-485 serial protocol
- **Federated learning** — share anonymised traffic patterns across multiple intersections to improve prediction
- **Digital twin** — simulate the full corridor in a physics engine (SUMO) before deploying timing changes to real signals

---

## 11. Tech Stack & Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | ≥8.4.0 | YOLOv8 inference engine |
| `torch` | ≥2.0.0 | PyTorch backend for YOLOv8 |
| `torchvision` | ≥0.15.0 | Image transforms used by ultralytics |
| `streamlit` | ≥1.32.0 | Web dashboard framework |
| `opencv-python` | ≥4.8.0 | Frame reading, annotation, VideoWriter |
| `numpy` | ≥1.24.0 | Array operations on frames |
| `pandas` | ≥2.0.0 | DataFrame for rolling chart history |
| `Pillow` | ≥10.0.0 | Image decoding for uploaded files |
| `PyYAML` | ≥6.0 | Model config parsing by ultralytics |
| `requests` | ≥2.28.0 | Model download fallback |
| `tqdm` | ≥4.64.0 | Progress bars during model load |
| `psutil` | ≥5.9.0 | System monitoring by ultralytics |

Install all:
```bash
pip install -r requirements.txt
```

---

## 12. Project File Structure

```
traffic_system/
│
├── app.py                  ← Streamlit entry point & main processing loop
├── config.py               ← All constants, timing params, SignalState, SystemStats
├── model_loader.py         ← @st.cache_resource YOLOv8 loader
├── detection.py            ← YOLOv8 inference + vehicle-class filtering
├── zones.py                ← Frame splitting + detection-to-zone assignment
├── signal_controller.py    ← 4-signal adaptive state machine with coordination
├── dashboard.py            ← OpenCV annotations + all Streamlit UI components
├── per_signal_input.py     ← Per-signal image upload panel + detection runner
├── best.pt                 ← Trained YOLOv8 model weights (IDD)
├── requirements.txt        ← Python dependencies
├── README.md               ← Setup and usage guide
├── solution.md             ← This document
└── __init__.py
```

### Data Flow Summary

```
User uploads image for S1
        ↓
per_signal_input.py → count_vehicles() → detections list
        ↓
apply_image_results_to_controller() → controller.signals[0].detections
        ↓
_live_signal_panel() ticks every 1s
        ↓
controller.update_signals() → _calc_green(weighted_score) → green_duration
        ↓
Phase: GREEN (14s) → YELLOW (3s) → ALL_RED (1s) → next signal GREEN
        ↓
render_full_signal_panel() → live countdown bar drains on screen
```

---

*Document version: 1.0 — reflects the current state of the traffic_system codebase.*
