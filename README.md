# Lowke — Multi-Camera Vehicle Tracking & Re-Identification System

A unified computer vision and simulation suite for sequential multi-camera vehicle tracking, cross-camera re-identification (Re-ID), and speed/kinematics estimation across blind-spot highway corridors.

---

## Repository Structure

```
Lowke/
├── Unreal/                    # Video Re-ID pipeline & HUD telemetry
│   ├── config.json            # Camera registry, virtual start timestamps, and distance matrix
│   ├── configs/               # Configuration settings
│   ├── models/                # YOLOv8 weights (yolov8n.pt, yolov8s.pt)
│   ├── multi_cam_reid.py      # Multi-camera YOLOv8 Re-ID tracking engine
│   ├── output/                # Processed output showcase videos
│   ├── requirements.txt       # Dependencies (ultralytics, opencv, torch, etc.)
│   ├── run.bat                # Windows runner script
│   ├── SAMPLE OUTPUTS/        # Showcase sample outputs
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

## 1. Unreal Pipeline (`Unreal/`)

The multi-camera pipeline processes sequential CCTV feeds along a highway corridor:
* **Detection & Re-ID**: Powered by YOLOv8 with bounding box feature matching and spatial-temporal constraints.
* **Corridor Telemetry**: Accurately computes frame-to-global timestamps and average section speeds:
  $$\text{Timestamp}_{\text{frame}} = \text{VirtualStartTime} + \left(\frac{\text{frame\_idx}}{\text{fps}}\right) \times 1\text{s}$$
  $$\text{Speed } (v) = \frac{\text{Distance}_{A \to B}}{\Delta t} \times 3.6 \quad (\text{km/h})$$
* **Quick Start**:
  * **Linux / macOS**:
    ```bash
    ./run_unreal.sh
    # or: cd Unreal && ./run.sh
    ```
  * **Windows**:
    ```cmd
    run_unreal.bat
    :: or: cd Unreal && run.bat
    ```

---

## 2. Highway CCTV Simulation (`sim/`)

A high-performance surveillance simulation and computer vision workstation built with Pygame, OpenCV, and NumPy, tuned to run at 30–60 FPS:
* **Corridor Geometry**: 2000m multi-lane corridor with 4 sequential cameras (`CAM 01` to `CAM 04`) and 250m unmonitored blind spots.
* **Camera Controls**: Disable or enable any camera on the fly with keyboard keys `1`, `2`, `3`, `4` or clicking the monitor headers without breaking target tracking continuity.
* **5-Parameter AI Re-ID**:
  1. Visual Reference Photo Match (NCC & HSV correlation)
  2. Outer Chassis Color Histogram
  3. Driver Windshield Avatar Inspection
  4. Speed Vector / Optical Displacement Radar
  5. Position Prediction (In-camera Kalman Filter + Blind-spot transit countdown)
* **Quick Start**:
  * **Linux / macOS**:
    ```bash
    ./run_sim.sh
    # or: cd sim && ./run.sh
    ```
  * **Windows**:
    ```cmd
    run_sim.bat
    :: or: cd sim && run.bat
    ```

---

## Requirements

- Python 3.8+
- PyTorch
- OpenCV (`opencv-python`)
- Pygame Community Edition (`pygame-ce`)
- Ultralytics (`ultralytics`)
- NumPy
