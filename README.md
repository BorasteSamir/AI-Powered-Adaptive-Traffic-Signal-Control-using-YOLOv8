<<<<<<< HEAD
=======

>>>>>>> 2fe9df0a77a6cf97ab886fafb0e95231b49a13ae
# 🚦 Smart Traffic Management System

> **Project:** AI-Powered Adaptive Traffic Signal Control using YOLOv8  
> **Dataset:** Indian Driving Dataset (IDD)  
> **Stack:** Python · YOLOv8 · Streamlit · OpenCV  
> **Run:** `streamlit run app.py`

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.32+-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.8+-green?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-IDD-orange?style=for-the-badge"/>
</p>

<p align="center">
  An AI-powered adaptive traffic signal control system that uses <strong>YOLOv8</strong> trained on the
  <strong>Indian Driving Dataset (IDD)</strong> to detect vehicles in real time and dynamically
  adjust signal green times across a 4-signal road corridor.
</p>

---

## 📋 Table of Contents

- [Demo](#-demo)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Getting the Model](#-getting-the-model)
- [Running the App](#-running-the-app)
- [Usage Guide](#-usage-guide)
- [Signal Timing Logic](#-signal-timing-logic)
- [Configuration](#-configuration)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)

---

##  Demo

```
 Upload one traffic image per signal → YOLOv8 detects vehicles instantly
 Signal panel shows live countdown timers for all 4 signals
 AUTO mode: green time adapts to vehicle count (7 vehicles = 14s, 15 = 30s)
 MANUAL mode: operator picks which signal is GREEN
 Force Next: skip current signal at any time
```

> **Dashboard preview:**
> ```
> ┌──────────┬──────────┬──────────┬──────────┐
> │  S1 📷   │  S2 📷   │  S3 📷   │  S4 📷   │
> │ 7 veh    │ 2 veh    │ 15 veh   │ 0 veh    │
> │ 🟡 MED   │ 🟢 LOW   │ 🔴 HIGH  │ 🟢 LOW   │
> │ 14.0s ✅ │ 8.0s     │ 30.0s    │ 8.0s     │
> └──────────┴──────────┴──────────┴──────────┘
>
> 🚦 Signal Panel
> [S1 🟢 ▓▓▓▓▓▓░░ 9.2s] [S2 🔴 wait 14s] [S3 🔴 wait 28s] [S4 🔴 wait 42s]
> ```

---

##  Features

| Feature | Description |
|---|---|
|  **YOLOv8 Detection** | Detects cars, buses, trucks, motorcycles, autorickshaws |
|  **Weighted Scoring** | Heavy vehicles (bus/truck) count 2.5× more than cars |
|  **Adaptive Green Time** | `green = vehicles × 2.0s`, capped at 2 minutes |
|  **Per-Signal Images** | Upload one image per signal — timing adjusts instantly |
|  **Video / Webcam** | Optional live feed with real-time detection overlay |
|  **Realistic Phases** | GREEN → YELLOW (3s) → ALL_RED (1s) → next GREEN |
|  **Live Countdown** | Draining progress bar on every signal card, updates every second |
|  **AUTO Mode** | Timer-driven round-robin with coordination |
|  **MANUAL Mode** | Operator selects any signal to turn GREEN |
|  **Force Next** | Skip current signal instantly (both modes) |
|  **Green Wave** | 2s offset so vehicle platoons hit the next signal on green |
|  **Skip Empty** | Signals with 0 vehicles are skipped to reduce waiting |
|  **Starvation Guard** | No signal waits more than 60s regardless of traffic |
|  **Live Dashboard** | 6 metric cards, rolling history chart, analytics panel |

---

##  How It Works

```
Step 1 — Upload Images
  User uploads one traffic photo per signal (S1–S4)
  YOLOv8 runs on each image immediately

Step 2 — Vehicle Detection
  Each detected vehicle is assigned a weight:
    car=1.0, bus=2.5, truck=2.5, motorcycle=0.8, autorickshaw=1.2

Step 3 — Green Time Calculation
  weighted_score = sum of weights of all detected vehicles
  green_time = max(8s, min(120s, weighted_score × 2.0))

Step 4 — Signal Cycle (AUTO)
  STARTUP (1s all-red)
      ↓
  S1 GREEN (adaptive) → YELLOW (3s) → ALL_RED (1s)
      ↓
  S2 GREEN (adaptive) → YELLOW (3s) → ALL_RED (1s)
      ↓
  S3 → S4 → S1 → ... (round-robin, skipping empty zones)

Step 5 — Live Dashboard
  Signal panel rerenders every 1 second
  Countdown bar drains in real time
  Operator can switch to MANUAL or force-advance at any time
```

---

##  Project Structure

```
traffic_system/
│
├── app.py                  ← Streamlit entry point — run this file
├── config.py               ← All constants, timing params, data classes
├── model_loader.py         ← YOLOv8 model loading (cached)
├── detection.py            ← YOLOv8 inference + vehicle filtering
├── zones.py                ← Frame splitting into 4 zones
├── signal_controller.py    ← 4-signal adaptive state machine
├── dashboard.py            ← OpenCV annotations + Streamlit UI components
├── per_signal_input.py     ← Per-signal image upload panel
│
├── best.pt                 ←  Trained model weights (NOT in repo — see below)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── solution.md             ← Detailed technical solution document
└── __init__.py
```

---

##  Prerequisites

- **Python 3.10 or higher** (uses modern type hints)
- **pip** package manager
- **Git** (to clone the repo)
- A trained `best.pt` YOLOv8 model file (see [Getting the Model](#-getting-the-model))

> **GPU (optional but recommended):** If you have an NVIDIA GPU with CUDA, PyTorch will use it automatically for faster inference. CPU works fine for image-mode usage.

---

##  Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/smart-traffic-management.git
cd smart-traffic-management/traffic_system
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `ultralytics` (YOLOv8)
- `streamlit`
- `opencv-python`
- `torch` + `torchvision`
- `numpy`, `pandas`, `Pillow`

> **Note:** `torch` installation may take a few minutes. If you have a CUDA GPU, install the CUDA version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

##  Getting the Model

The trained `best.pt` file (~50MB) is **not included in this repository** due to GitHub's file size limits.

### Option A — Use your own trained model

If you have trained a YOLOv8 model on IDD or any traffic dataset:

```bash
# Copy your model into the traffic_system folder
cp /path/to/your/best.pt traffic_system/best.pt
```

### Option B — Train from scratch

```bash
pip install ultralytics
yolo train data=idd.yaml model=yolov8n.pt epochs=50 imgsz=640
# Copy runs/detect/train/weights/best.pt → traffic_system/best.pt
```

### Option C — Use a COCO-pretrained model (quick test)

The app works with any YOLOv8 model. For a quick test with COCO classes:

```bash
# The app will auto-download yolov8n.pt on first run
# Change MODEL_PATH in config.py:
MODEL_PATH = "yolov8n.pt"
```

> **Note:** COCO-trained models detect `car`, `bus`, `truck`, `motorcycle` but not `autorickshaw`. The system will still work — autorickshaw detections will simply be 0.

### Option D — Download via Git LFS (if configured)

```bash
git lfs install
git lfs pull
```

---

##  Running the App

Make sure you are inside the `traffic_system` folder with your virtual environment activated:

```bash
cd traffic_system
streamlit run app.py
```

The dashboard opens automatically at:
```
http://localhost:8501
```

### Run on a custom port

```bash
streamlit run app.py --server.port 8080
```

### Run accessible on your local network

```bash
streamlit run app.py --server.address 0.0.0.0
```

---

##  Usage Guide

### Mode 1 — Image Analysis (Recommended for first run)

1. Open the app at `http://localhost:8501`
2. In the **" Traffic Image Analysis"** section, upload one traffic image per signal (S1–S4)
3. YOLOv8 detects vehicles in each image immediately — you'll see bounding boxes and green time
4. The **Signal Control Panel** starts the countdown automatically
5. Watch the live draining progress bar on the active GREEN signal
6. Use ** Auto Mode** to let the system cycle automatically
7. Use ** Manual Mode** to pick which signal is GREEN yourself
8. Click ** Force Next Signal** to skip the current signal at any time

### Mode 2 — Live Video Feed

1. In the sidebar, select **"Upload Video"** or **"Webcam"**
2. Upload a traffic video file (`.mp4`, `.avi`, `.mov`, `.mkv`)
3. Click **▶ Start** to begin processing
4. The video frame is split into 4 vertical zones (S1–S4)
5. YOLOv8 detects vehicles in each zone every frame
6. Signal timing updates in real time based on live vehicle counts
7. Click **⏹ Stop** to end the session
8. Optionally check **"Save processed video"** to export the annotated output

### Adjusting Detection Sensitivity

Use the **Confidence threshold** slider in the sidebar:
- **Lower (0.20)** — detects more vehicles, including distant/partial ones
- **Higher (0.60)** — only high-confidence detections, fewer false positives
- **Default (0.35)** — balanced for most traffic scenes

### Switching Control Modes

| Mode | When to use |
|---|---|
|  **AUTO** | Normal operation — system manages timing automatically |
|  **MANUAL** | Emergency override, testing, or when you need to hold a specific signal |
|  **Force Next** | Quickly advance past a signal that has been green too long |

---

##  Signal Timing Logic

### Green Time Formula

```
green_time = max(8s, min(120s, weighted_score × 2.0))
```

### Vehicle Weights

| Vehicle | Weight | Example: 5 of these = |
|---|---|---|
| car | 1.0 | score 5.0 → 10s green |
| bus | 2.5 | score 12.5 → 25s green |
| truck | 2.5 | score 12.5 → 25s green |
| autorickshaw | 1.2 | score 6.0 → 12s green |
| motorcycle | 0.8 | score 4.0 → 8s green (minimum) |

### Quick Reference Table

| Vehicles | Green Time |
|---|---|
| 0–3 | 8s (minimum) |
| 5 | 10s |
| 7 | **14s** |
| 10 | 20s |
| 15 | **30s** |
| 20 | 40s |
| 30 | 60s |
| 60+ | **120s (2 min max)** |

### Phase Sequence

```
🟢 GREEN  (8s – 120s, adaptive)
    ↓
🟡 YELLOW (3s fixed)
    ↓
⬛ ALL_RED (1s safety clearance)
    ↓
🟢 next signal GREEN
```

---

##  Configuration

All tunable parameters are in `config.py`:

```python
# Signal count
NUM_SIGNALS       = 4

# Traffic density thresholds
THRESHOLD_LOW     = 5    # ≤5 vehicles → LOW
THRESHOLD_MEDIUM  = 12   # ≤12 vehicles → MEDIUM (else HIGH)

# Green time
GREEN_MIN_SEC     = 8.0   # minimum green (floor)
GREEN_PER_VEHICLE = 2.0   # seconds per weighted vehicle unit
GREEN_MAX_SEC     = 120.0 # maximum green (2 minutes)

# Phase durations
YELLOW_SEC        = 3.0   # amber warning
ALL_RED_SEC       = 1.0   # safety clearance

# Coordination
COORDINATION_ENABLED = True   # green wave offset
OFFSET_SEC           = 2.0    # head-start for next signal
SKIP_EMPTY_SIGNALS   = True   # skip zones with 0 vehicles

# Model
MODEL_PATH = "best.pt"   # relative to this file's directory
```

To change any value, edit `config.py` — no other file needs to be touched.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Training Dataset | [Indian Driving Dataset (IDD)](https://idd.insaan.iiit.ac.in/) |
| Dashboard | [Streamlit](https://streamlit.io/) |
| Frame Annotation | [OpenCV](https://opencv.org/) |
| Deep Learning | [PyTorch](https://pytorch.org/) |
| Numerics | [NumPy](https://numpy.org/) |
| Charts | [Pandas](https://pandas.pydata.org/) |
| Language | Python 3.10+ |

---

##  Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is for academic and research purposes. The YOLOv8 model is subject to the [Ultralytics AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

---

<p align="center">
  Built with ❤️ using YOLOv8 + Streamlit · Indian Driving Dataset
</p>
