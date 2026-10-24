# Multi-Camera & Single-Camera Vehicle Tracking & Telemetry HUD

High-performance vehicle tracking, cross-camera Re-Identification (Re-ID), and Mission Control Telemetry HUD for traffic surveillance.

---

## 1. Project Directory Structure

```text
Proj/
├── configs/
│   └── config.json            # Camera registry, timestamps, and distance matrices
├── models/
│   ├── yolov8n.pt             # CPU-optimized YOLOv8 Nano weights
│   └── yolov8s.pt             # YOLOv8 Small weights
├── output/
│   └── final_showcase.mp4     # Generated 1080p stitched showcase video
├── videos/
│   └── cam1.mp4               # Input raw traffic footage
├── multi_cam_reid.py          # Unified single-camera & multi-camera tracking engine
├── requirements.txt           # Python package dependencies
├── run.bat                    # One-click Windows launch script
└── README.md                  # System documentation
```

---

## 2. Dynamic Layout Modes

1. **Single-Video Mode (1 Camera):**
   - **Main Viewport (1280x1080):** High-definition feed with neon corner-bracket bounding boxes, vehicle classifications, and track IDs.
   - **Mission Control HUD (640x1080):** Full-height telemetry dashboard featuring live vehicle count, active tracking, classification breakdown (Cars, Buses, Trucks, Motorcycles), privacy metrics (k-anonymity & differential privacy), and live vehicle dossier roster.

2. **Multi-Camera Mode (3 Cameras):**
   - **Quad Grid (1920x1080):**
     * Quadrant 1 (Top-Left): `CAM_01` (Inlet checkpoint)
     * Quadrant 2 (Top-Right): `CAM_02` (Mid-corridor checkpoint)
     * Quadrant 3 (Bottom-Left): `CAM_03` (Outlet checkpoint)
     * Quadrant 4 (Bottom-Right): Cross-camera Re-ID association gallery & corridor transit meters.

---

## 3. How to Run

### Option A: Using the Windows Batch Launcher
Double-click `run.bat` or run in terminal:
```cmd
run.bat
```

### Option B: Direct Python Execution
```powershell
# Auto-detects input video in videos/ folder
python multi_cam_reid.py

# Process a specific video
python multi_cam_reid.py videos/cam1.mp4

# Quick preview (e.g. first 150 frames)
python multi_cam_reid.py --frames 150

# Run headless without opening the interactive player
python multi_cam_reid.py --no-player
```

### Showcase Player Controls:
- **`[1]`**: 1x Normal Playback Speed
- **`[2]`**: 2x Fast-Forward Showcase Playback
- **`[SPACE]`**: Pause / Resume Playback
- **`[R]`**: Replay from the beginning
- **`[Q]` or `[ESC]`**: Exit player
