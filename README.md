# Lowke — Multi-Camera Vehicle Tracking & Re-Identification System

A unified computer vision and simulation suite for sequential multi-camera vehicle tracking, cross-camera re-identification (Re-ID), and speed/kinematics estimation across blind-spot highway corridors.

---

## Repository Structure

```
Lowke/
├── real/                      # Real-world video Re-ID pipeline & HUD telemetry
│   ├── config.json            # Camera registry, virtual start timestamps, and distance matrix
│   ├── multi_cam_reid.py      # Multi-camera YOLOv8 Re-ID tracking engine
│   ├── requirements.txt       # Dependencies (ultralytics, opencv, torch, etc.)
│   ├── run.sh                 # Linux / macOS runner script
│   ├── run.bat                # Windows runner script
│   ├── yolov8n.pt             # YOLOv8 Nano weights
│   ├── yolov8s.pt             # YOLOv8 Small weights
│   └── README.md              # Pipeline documentation & mathematical formulas
│
└── sim/                       # Real-time highway corridor CCTV simulation & CV tracker
    ├── cv_tracker.py          # 5-Parameter AI target dossier & cross-camera Re-ID
    ├── hud_overlay.py         # Pelco/Traffic-Pro CCTV workstation UI & HUD overlay
    ├── main.py                # Main simulation loop and event dispatcher
    ├── sim_world.py           # Highway corridor physics, camera views, and traffic generator
    ├── tiny_car_net.py        # PyTorch convolutional neural network for vehicle feature extraction
    ├── requirements.txt       # Dependencies (pygame-ce, opencv, torch, numpy)
    ├── run.sh                 # Interactive workstation launcher
    └── README.md              # Simulation architecture & control reference
```

---

## 1. Real Pipeline (`real/`)

The real-world pipeline processes multiple sequential CCTV feeds along a highway corridor:
* **Detection & Re-ID**: Powered by YOLOv8 with bounding box feature matching and spatial-temporal constraints.
* **Corridor Telemetry**: Accurately computes frame-to-global timestamps and average section speeds:
  $$\text{Timestamp}_{\text{frame}} = \text{VirtualStartTime} + \left(\frac{\text{frame\_idx}}{\text{fps}}\right) \times 1\text{s}$$
  $$\text{Speed } (v) = \frac{\text{Distance}_{A \to B}}{\Delta t} \times 3.6 \quad (\text{km/h})$$
* **Quick Start**:
  ```bash
  cd real
  ./run.sh
  ```

---

## 2. Highway CCTV Simulation (`sim/`)

A high-performance surveillance simulation and computer vision workstation built with Pygame, OpenCV, and NumPy, tuned to run at 30–60 FPS:
* **Corridor Geometry**: 2000m multi-lane corridor with 4 sequential cameras (`CAM 01` to `CAM 04`) and 250m unmonitored blind spots.
* **5-Parameter AI Re-ID**:
  1. Visual Reference Photo Match (NCC & HSV correlation)
  2. Outer Chassis Color Histogram
  3. Driver Windshield Avatar Inspection
  4. Speed Vector / Optical Displacement Radar
  5. Position Prediction (In-camera Kalman Filter + Blind-spot transit countdown)
* **Quick Start**:
  ```bash
  cd sim
  ./run.sh
  ```

---

## Requirements

- Python 3.8+
- PyTorch
- OpenCV (`opencv-python`)
- Pygame Community Edition (`pygame-ce`)
- Ultralytics (`ultralytics`)
- NumPy
