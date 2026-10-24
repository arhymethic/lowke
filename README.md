# Lowke: Multi-Camera Vehicle Tracking, Re-Identification & Telemetry Suite

[![](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/Object%20Detection-YOLOv8-green.svg)](https://ultralytics.com/)
[![](https://img.shields.io/badge/Deep%20Learning-PyTorch%20%7C%20NumPy-orange.svg)](https://pytorch.org/)
[![](https://img.shields.io/badge/Computer%20Vision-OpenCV%20%7C%20Pygame-red.svg)](https://opencv.org/)
[![](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()
[![](https://img.shields.io/badge/License-MIT-blue.svg)]()

A unified computer vision and simulation suite for sequential multi-camera vehicle tracking, cross-camera re-identification (Re-ID), and speed/kinematics estimation across blind-spot highway corridors.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Methodological Approach](#methodological-approach)
  - [1. Spatial-Temporal Corridor Formulation](#1-spatial-temporal-corridor-formulation)
  - [2. Multi-Stage Detection & Tracking Pipeline](#2-multi-stage-detection--tracking-pipeline)
  - [3. 5-Parameter AI Dossier & Feature Fusion](#3-5-parameter-ai-dossier--feature-fusion)
  - [4. Fault-Tolerant Camera Outage & Impostor Rejection](#4-fault-tolerant-camera-outage--impostor-rejection)
  - [5. Privacy-Preserving Telemetry](#5-privacy-preserving-telemetry)
- [Models & Feature Extractors](#models--feature-extractors)
- [Mathematical Formulations](#mathematical-formulations)
- [Repository Structure](#repository-structure)
- [Installation & Quick Start](#installation--quick-start)
  - [Highway CCTV Simulator (sim/)](#highway-cctv-simulator-sim)
  - [Unreal Re-ID Telemetry Pipeline (Unreal/)](#unreal-re-id-telemetry-pipeline-unreal)
- [Interactive Controls & Keybindings](#interactive-controls--keybindings)
- [Performance & Benchmarks](#performance--benchmarks)

---

## Overview

The Lowke platform addresses a critical challenge in intelligent transportation systems (ITS): continuous tracking and re-identification of vehicles moving through sequential highway surveillance zones separated by extensive unmonitored blind spots (250m+).

The repository provides two complementary environments:

1. **Unreal Multi-Camera Pipeline (`Unreal/`)**: An enterprise-grade video processing engine that consumes pre-recorded or live multi-stream CCTV footage, performs YOLOv8 vehicle detection, deep appearance feature extraction, cross-camera Re-ID matching, and renders a 1080p Mission Control Telemetry HUD.
2. **Highway CCTV Workstation Simulator (`sim/`)**: A real-time physics and computer vision simulation operating at 60 FPS. It models a 2000m linear corridor across 4 CCTV cameras, enabling interactive experimentation with environmental degradation (night lighting, rain, contrast shifts), traffic density, camera outages, and multi-attribute classification.

```
===================================================================================================
HIGHWAY CORRIDOR TOPOLOGY (West -> East: 0m to 2000m)
  [ CAM 01 ] ----(250m Blind Spot)----> [ CAM 02 ] ----(250m Blind Spot)----> [ CAM 03 ] ----(250m Blind Spot)----> [ CAM 04 ]
  (0m - 250m)                           (500m - 750m)                         (1000m - 1250m)                       (1500m - 1750m)
===================================================================================================
```

---

## System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                       INPUT VIDEO STREAMS                                         |
|                 CAM_01 (Inlet)         CAM_02 (Midway)         CAM_03 (Outlet)                    |
+-----------------------------------------------+---------------------------------------------------+
                                                |
                                                v
+---------------------------------------------------------------------------------------------------+
|                                STAGE 1: DETECTION & LOCAL TRACKING                                |
|   - YOLOv8 (640px Backbone) -> High-sensitivity bounding box localization (Classes 1, 2, 3, 5, 7)  |
|   - ByteTrack / Kalman Trajectory -> Intra-camera frame-to-frame vehicle association              |
+-----------------------------------------------+---------------------------------------------------+
                                                |
                                                v
+---------------------------------------------------------------------------------------------------+
|                               STAGE 2: MULTI-MODAL FEATURE EXTRACTION                             |
|   - Deep Appearance Embeddings (ResNet-50 / L2 Normalized Feature Vectors)                        |
|   - Spatial Color Histograms (HSV 3D Distribution)                                                |
|   - Windshield ROI & Driver Visibility (Adaptive Occlusion Weighting)                             |
|   - Optical Velocity & Section Transit Speed                                                      |
+-----------------------------------------------+---------------------------------------------------+
                                                |
                                                v
+---------------------------------------------------------------------------------------------------+
|                               STAGE 3: CROSS-CAMERA RE-IDENTIFICATION                             |
|   - Spatio-Temporal Arrival Gating (Minimum / Maximum Travel Time Constraints)                    |
|   - Dynamic Feature Fusion: S_total = w_visual*S_visual + w_color*S_color + w_speed*S_speed ...   |
|   - Outage Tolerance: Direct bridge association (CAM_01 -> CAM_03) during CAM_02 blackout         |
+-----------------------------------------------+---------------------------------------------------+
                                                |
                                                v
+---------------------------------------------------------------------------------------------------+
|                               STAGE 4: MISSION CONTROL TELEMETRY HUD                              |
|   - Full HD 1080p Canvas: Side-by-Side (1280x1080 + 640x1080) or Quad 2x2 CCTV Grid              |
|   - Privacy Defense: k-Anonymity (k >= 10) & Differential Privacy via Laplace Mechanism           |
|   - Real-time Dossier Registry: Global ID, Section Speeds, Arrival ETAs, Class Breakdown          |
+---------------------------------------------------------------------------------------------------+
```

---

## Methodological Approach

### 1. Spatial-Temporal Corridor Formulation

In high-speed corridors, naive visual matching across cameras frequently leads to false positives due to vehicle similarities (e.g., common white sedans). Lowke constrains visual search through physical kinematics:

* **Transit Window Bounds**: A candidate in camera $k$ is considered for association with a track from camera $k-1$ only if the elapsed transit time $\Delta t$ satisfies:
  $$\Delta t_{\min} \le \Delta t \le \Delta t_{\max}$$
  where $\Delta t_{\min} = \frac{d_{\text{gap}}}{v_{\max}}$ and $\Delta t_{\max} = \frac{d_{\text{gap}}}{v_{\min}}$.
* **Arrival Gating**: Out-of-camera candidates are strictly masked while the target vehicle is physically in transit through blind spots.

### 2. Multi-Stage Detection & Tracking Pipeline

* **Detection**: Utilizes YOLOv8 at 640px image resolution with a calibrated confidence threshold of 0.15 to reliably capture blurred, distant, or partially occluded vehicles entering camera viewports.
* **Intra-Camera Tracking**: Within a single camera feed, local tracklets are maintained using intersection-over-union (IoU) association and linear constant-velocity Kalman filtering.

### 3. 5-Parameter AI Dossier & Feature Fusion

Each tracked target is assigned a persistent dossier aggregating five distinct parameter channels:

| Parameter Channel | Extractor Method | Metric / Representation | Adaptive Behavior |
|:---|:---|:---|:---|
| **1. Visual Reference Photo** | Normalized Cross-Correlation (NCC) + HSV Corr | $64 \times 64$ RGB crop similarity ($S_{\text{visual}} \in [0, 1]$) | Compares against initial lock photograph |
| **2. Chassis Color Distribution** | 3D HSV Color Histogram | Bhattacharyya distance / Histogram correlation | Robust against illumination and contrast shifts |
| **3. Driver Windshield Avatar** | Segmented Windshield ROI Analysis | Color vector & silhouette classification | Automatically zeros weight when obscured by rain/night |
| **4. Kinematic Speed Vector** | Optical displacement over time ($\Delta x / \Delta t$) | Calibrated road speed in km/h | Cross-checks expected section velocity |
| **5. Spatial Position & Kalman ETA** | 2D Kalman Filter + Linear Transit Countdown | $\mathbf{x} = [x, y, v_x, v_y]^T$ & $d = v \cdot \Delta t$ | Predicts arrival seconds into downstream cameras |

The cumulative Re-ID similarity score $S_{\text{total}}$ is computed as:
$$S_{\text{total}} = \frac{w_v S_{\text{visual}} + w_c S_{\text{color}} + w_d S_{\text{driver}} + w_s S_{\text{speed}} + w_p S_{\text{position}}}{w_v + w_c + w_d + w_s + w_p}$$

When environmental conditions deteriorate (e.g. nocturnal lighting or severe weather), the driver avatar parameter is automatically flagged as obscured ($w_d \to 0$), and its weight is dynamically redistributed across chassis color, speed radar, and Kalman kinematics.

### 4. Fault-Tolerant Camera Outage & Impostor Rejection

* **Zero-Jump Lock Protection**: If an active camera is disabled or drops offline, the system locks the target's physical state in memory and initiates a blind-spot transit countdown. Downstream cameras strictly ignore background vehicles until the target arrives within the physical corridor boundary ($|x_{\text{target}} - x_{\text{mid}}| < 115\text{m}$).
* **Camera Bridging**: In the event of an intermediate sensor failure (e.g. `CAM_02` offline), the pipeline automatically bridges `CAM_01` to `CAM_03`, dynamically doubling the transit distance and travel time thresholds without terminating the active track.

### 5. Privacy-Preserving Telemetry

To ensure compliance with data protection standards (GDPR / CCPA) in public surveillance:
* **$k$-Anonymity**: Vehicle records are released to the public telemetry log only if at least $k = 10$ vehicles share equivalent quasi-identifiers (chassis color family, vehicle classification, speed bracket).
* **Differential Privacy**: Numerical metrics (such as average section velocity and transit duration) are perturbed using Laplace noise:
  $$\tilde{v} = v + \text{Lap}\left(0, \frac{\Delta S}{\epsilon}\right)$$
  with privacy budget $\epsilon = 0.5$ and query sensitivity $\Delta S$.

---

## Models & Feature Extractors

| Model | Architecture | Size / Params | Primary Role | Implementation Framework |
|:---|:---|:---|:---|:---|
| **YOLOv8n** | Modified CSPDarknet53 + C2f + PAN-FPN | ~3.2M parameters | Primary object detection (Bicycle, Car, Motorcycle, Bus, Truck) | PyTorch / Ultralytics |
| **YOLOv8s** | Scaled CSPDarknet backbone | ~11.2M parameters | High-accuracy detection baseline for complex highway scenes | PyTorch / Ultralytics |
| **Deep Re-ID Embedder** | ResNet-50 Feature Extractor | 2048-dim feature vector | Deep appearance embedding extraction with L2 normalization | PyTorch / Torchvision |
| **TinyCarNet** | Dual-Head Feedforward Neural Network | 96 -> 48 -> (10 + 8) | Instantaneous chassis paint & driver silhouette classification (< 0.2ms) | Pure Vectorized NumPy |
| **Kalman Filter Engine** | Discrete 2D Linear State Space | 4-state vector ($\mathbf{x}$) | Forward trajectory projection & in-camera occlusion coasting | NumPy / SciPy |

---

## Mathematical Formulations

### Global Timestamp Synchronization
Given individual camera stream frame indices and user-calibrated virtual startup offsets:
$$\text{Timestamp}_{\text{global}} = \text{VirtualStartTime} + \left(\frac{\text{frame\_index}}{\text{FPS}}\right) \times 1.0\text{s}$$

### Section Velocity Estimation
Between consecutive camera checkpoints $A$ and $B$ positioned at known corridor coordinates:
$$v_{A \to B} = \frac{d(A, B)}{t_{\text{arrival}}(B) - t_{\text{departure}}(A)} \times 3.6 \quad [\text{km/h}]$$

### Normalized Cross-Correlation (NCC)
Given reference template $T$ and candidate ROI patch $I$:
$$\gamma(x, y) = \frac{\sum_{x', y'} \left(T(x', y') - \bar{T}\right) \left(I(x + x', y + y') - \bar{I}\right)}{\sqrt{\sum_{x', y'} \left(T(x', y') - \bar{T}\right)^2 \sum_{x', y'} \left(I(x + x', y + y') - \bar{I}\right)^2}}$$

### Laplace Perturbation
$$\text{Lap}(x; \mu, b) = \frac{1}{2b} \exp\left(-\frac{|x - \mu|}{b}\right), \quad b = \frac{\Delta S}{\epsilon}$$

---

## Repository Structure

```
Lowke/
├── README.md                      # Unified system documentation & mathematical formulation
├── requirements.txt               # Top-level dependencies reference
├── run_sim.sh                     # Linux/macOS root launcher for Highway CCTV Simulator
├── run_sim.bat                    # Windows root launcher for Highway CCTV Simulator
├── run_unreal.sh                  # Linux/macOS root launcher for Unreal Re-ID Pipeline
├── run_unreal.bat                 # Windows root launcher for Unreal Re-ID Pipeline
│
├── Unreal/                        # Real-world video Re-ID pipeline & telemetry HUD
│   ├── config.json                # Camera stream definitions, distances, and virtual offsets
│   ├── configs/                   # Parameter configuration presets
│   ├── models/                    # YOLOv8 neural network weights (yolov8n.pt, yolov8s.pt)
│   ├── multi_cam_reid.py          # Unified multi-camera Re-ID execution engine
│   ├── output/                    # Generated 1080p stitched showcase videos
│   ├── requirements.txt           # Pipeline dependencies (ultralytics, torch, opencv)
│   ├── run.sh                     # Linux/macOS launcher with automated .venv setup
│   ├── run.bat                    # Windows launcher with automated .venv setup
│   ├── SAMPLE OUTPUTS/            # Reference output videos and showcase demonstrations
│   └── README.md                  # Module documentation
│
└── sim/                           # Interactive 60 FPS Highway Corridor CCTV Workstation
    ├── cctv_tracker.desktop       # Linux desktop integration launcher
    ├── cv_tracker.py              # 5-parameter AI dossier, Kalman filter, & handover logic
    ├── hud_overlay.py             # Pelco/Traffic-Pro industrial surveillance HUD interface
    ├── main.py                    # Main event dispatcher and simulation loop
    ├── sim_world.py               # Highway corridor physics, camera views, & weather rendering
    ├── tiny_car_net.py            # Vectorized multi-head neural attribute classifier (NumPy)
    ├── requirements.txt           # Simulator dependencies (pygame-ce, opencv, numpy)
    ├── run.sh                     # Linux/macOS launcher with automated .venv setup
    ├── run.bat                    # Windows launcher with automated .venv setup
    └── README.md                  # Simulator mechanics and control reference
```

---

## Installation & Quick Start

The repository is fully cross-platform. Automated runners detect system Python (Python 3.8+), create an isolated virtual environment (`.venv`), install missing packages from `requirements.txt`, and launch the application.

### Highway CCTV Simulator (`sim/`)

* **Linux / macOS**:
  ```bash
  # From root directory:
  ./run_sim.sh

  # Or from the sim folder:
  cd sim && ./run.sh
  ```

* **Windows**:
  ```cmd
  REM From root directory:
  run_sim.bat

  REM Or from the sim folder:
  cd sim && run.bat
  ```

### Unreal Re-ID Telemetry Pipeline (`Unreal/`)

* **Linux / macOS**:
  ```bash
  # Standard execution:
  ./run_unreal.sh

  # Process specific video with custom frame limit:
  ./run_unreal.sh --frames 300

  # Headless mode (no GUI player):
  ./run_unreal.sh --no-player
  ```

* **Windows**:
  ```cmd
  REM Standard execution:
  run_unreal.bat

  REM Process specific video with custom frame limit:
  run_unreal.bat --frames 300

  REM Headless mode (no GUI player):
  run_unreal.bat --no-player
  ```

---

## Interactive Controls & Keybindings

### Highway Simulator Controls (`sim/`)

| Input | Action |
|:---|:---|
| **Click (Left Mouse)** | Click any vehicle in `CAM 01` - `CAM 04` to lock and extract 5-parameter profile |
| **`1`, `2`, `3`, `4`** | Toggle individual cameras ON / OFF on the fly (prediction model maintains track) |
| **`SPACE`** | Pause / Resume simulation physics |
| **`L`** | Auto-lock the next vehicle entering `CAM 01` |
| **`T`** | Retrain `TinyCarNet` neural network live on CPU (< 0.2s) |
| **`U`** | Disengage target lock |
| **`ESC`** | Exit simulator |

### Showcase Video Player Controls (`Unreal/`)

| Key | Action |
|:---|:---|
| **`1`** | 1x Speed (Standard Real-time Playback) |
| **`2`** | 2x Speed (Showcase Fast-Forward Mode - Default) |
| **`SPACE`** | Pause / Resume Playback |
| **`R`** | Replay video from beginning |
| **`Q` / `ESC`** | Quit video player |

---

## Performance & Benchmarks

Benchmarked on modern x86_64 CPU hardware (Intel Core i5/i7, Quad-Core):

* **Highway Corridor Simulator (`sim/`)**:
  * Physics & Graphics Loop: **60.0 FPS** (Stable)
  * Feature Extraction & Kalman Projection: **0.18 ms / frame**
  * TinyCarNet Live Inference: **0.12 ms / patch**
* **Unreal Re-ID Engine (`Unreal/`)**:
  * YOLOv8n Inference (CPU, 640px): **18–25 FPS**
  * YOLOv8n Inference (CUDA, 640px): **85–120 FPS**
  * 1080p Telemetry HUD Stitched Output: **25.0 FPS export**
