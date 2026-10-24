# Project Zomboid CCTV Tracker — Linear Highway Corridor & 5-Parameter Re-ID

A high-performance surveillance simulation and computer vision vehicle re-identification (Re-ID) system built with **Pygame**, **OpenCV**, and **NumPy**, tuned to run at **30–60+ FPS on a ThinkPad T470s**.

---

## 1. System Architecture: Linear 4-Camera Highway Corridor

The road is a continuous 2000m multi-lane highway running West ➔ East with 4 sequential cameras and unmonitored blind-spot gaps:

```
===================================================================================================
HIGHWAY CORRIDOR (West ➔ East: 0m to 2000m)
  [ CAM 01 ] ──(250m Gap)──► [ CAM 02 ] ──(250m Gap)──► [ CAM 03 ] ──(250m Gap)──► [ CAM 04 ]
  (0m - 250m)                (500m - 750m)              (1000m-1250m)              (1500m-1750m)
===================================================================================================
```

* **Project Zomboid Isometric Angle ($30^\circ$ / $45^\circ$ overhead):**
  1. **Chassis Panels:** High-visibility vehicle paint with dynamic lighting and contrast degradation.
  2. **Windshield:** Transparent glass revealing the driver's colored shirt/avatar.
  3. **Rigid Sub-Part Reticles:** Hierarchical tracking of vehicle chassis, windshield/driver ROI, and front bumper radar.

---

## 2. Multi-Parameter AI Vehicle Profiler & Re-ID

When a car is locked in `CAM 01` (or clicked at any point), the AI crops a $64 \times 64$ pixel patch from the raw camera feed and assigns a Registry ID (e.g., `#VH-042`).

```
                    ┌───────────────────────────────────────────────┐
                    │      64x64 Isometric Car Pixel Crop           │
                    └───────────────────────┬───────────────────────┘
                                            │
         ┌───────────────┬──────────────────┼──────────────────┬───────────────┐
         ▼               ▼                  ▼                  ▼               ▼
   ┌───────────┐   ┌───────────┐      ┌───────────┐      ┌───────────┐   ┌─────────────┐
   │ 1. VISUAL │   │ 2. CHASSIS│      │ 3. DRIVER │      │ 4. SPEED  │   │ 5. POSITION │
   │   MATCH   │   │   COLOR   │      │   AVATAR  │      │  RADAR    │   │ PREDICTION  │
   ├───────────┤   ├───────────┤      ├───────────┤      ├───────────┤   ├─────────────┤
   │ NCC &     │   │ Outer     │      │ Windshield│      │ Optical   │   │ Kalman      │
   │ HSV Corr  │   │ Chassis   │      │ ROI patch │      │ velocity  │   │ Filter +    │
   │ vs Ref.   │   │ HSV Hist  │      │ color avg │      │ Δpx / Δt  │   │ Blind-spot  │
   │ Photo     │   │           │      │ (if vis.) │      │           │   │ ETA math    │
   └───────────┘   └───────────┘      └───────────┘      └───────────┘   └─────────────┘
```

### How the Parameters Work:

1. **Visual Reference Photo Match:**
   - Compares the live camera crop against the initial reference photograph using Normalized Cross-Correlation (NCC) and HSV correlation.
2. **Car Body Color:**
   - *Simulator:* Chassis color polygon affected by the Color Similarity and Day/Night sliders.
   - *AI Model:* Computes HSV color histogram on the outer chassis boundary.
3. **Driver Avatar (If Available):**
   - *Simulator:* Visible driver avatar (head + colored shirt/jacket: Yellow, Red, Cyan, White, Green, Orange, Purple). Obscured when Night or Dust/Rain is increased!
   - *AI Model:* Inspects the windshield ROI. If contrast is clear, detects the shirt color (`Driver: DETECTED [Yellow Shirt]`). If obscured by glare/night, flags `Driver: OBSCURED (Weight = 0.0)` and dynamically reallocates weight to the other parameters!
4. **Speed Vector:**
   - *Simulator:* Cruise speed (45–75 km/h) with natural traffic variance.
   - *AI Model:* Computes pixel displacement over time ($\Delta px / \Delta t$) inside the camera and converts it to road speed (`Est. Speed: 67.4 km/h`).
5. **Position Prediction (In-Cam & Between-Cams):**
   - *In-Camera:* Kalman Filter draws a live forward trajectory line (`o---o---o--->`) for $+0.5\text{s}$ and $+1.0\text{s}$, and coasts through bridge occlusions in CAM 02.
   - *Between Cameras (The Blind Spot):* Predicts the transit progress in the gap ($d = v \cdot \Delta t$) with a countdown progress bar: `In Transit: 140m / 250m to CAM 02 — ETA: 1.8s`.

---

## 3. UI Layout & Controls (1280×720)

```
+-----------------------------------+-----------------------------------+----------------------------------+
| CAM 01 (0m-250m)      [PASSED]    | CAM 02 (500m-750m)  [★ ACTIVE]    |   AI TARGET DOSSIER: #VH-042     |
|   ── ── ── ── ── ── ── ── ──      |   ── ── [🚗 #VH-042] ──o──o──►    |  +----------+                    |
|   ── ── ── ── ── ── ── ── ──      |   ── [🚗] ── ── ── [🚗] ── ──     |  | 64x64    |  Status: LOCKED    |
|                                   |                                   |  | Mugshot  |  Active Cam: CAM 02|
+-----------------------------------+-----------------------------------+  +----------+  Match Conf: 96.4% |
| CAM 03 (1000m-1250m)  [PREDICTED] | CAM 04 (1500m-1750m) [STANDBY]    |----------------------------------|
|   ── ── ── ── [ETA: 4.2s] ──      |   ── ── ── ── ── ── ── ── ──      | 1. Car Color:   Crimson (94%)    |
|   ── ── ── ── ── ── ── ── ──      |   ── ── ── ── ── ── ── ── ──      | 2. Driver:      Yellow Shirt(88%)|
|                                   |                                   | 3. Sticker:     Dual Stripe (97%)|
+-----------------------------------+-----------------------------------+ 4. Est. Speed:  74.2 km/h        |
|                     CORRIDOR POSITION PREDICTOR                       | 5. Next Pos:    CAM 03 in 4.2s   |
|  [CAM 01] ═══════ [CAM 02: ★ HERE] ═══(Est)--> [CAM 03] ════ [CAM 04] |----------------------------------|
+-----------------------------------------------------------------------+ ENVIRONMENT SLIDERS              |
|  [▶ PLAY / PAUSE]   [🎯 AUTO-LOCK NEXT CAR]   [🧠 TRAIN MODEL]         | [====o===] Traffic Density (40)  |
|                                                                       | [======o=] Car Color Similarity  |
|                                                                       | [==o=====] Day / Night           |
|                                                                       | [====o===] Dust / Rain Noise     |
+-----------------------------------------------------------------------+----------------------------------+
```

---

## 4. How the Sliders Demonstrate Robust Multi-Attribute Re-ID

1. **Traffic Density Slider (10 to 80 cars):**
   - Pack the highway corridor with dozens of vehicles. The AI tracker maintains lock on `#VH-042` across sequential camera handovers despite heavy clutter.
2. **Car Color Similarity Slider (0% to 100%):**
   - Move to 100% to turn all vehicles asphalt grey. In the AI Dossier Panel, Car Color weight drops to 5%, and the model successfully re-identifies the target using **Driver Color**, **Roof Sticker**, **Vehicle Speed**, and **Kalman ETA**!
3. **Day / Night & Rain / Dust Sliders:**
   - Crank up Night or Dust to obscure the windshield. The AI updates: `Driver: OBSCURED (Weight = 0.0)` and redistributes weight to Roof Decal, Speed, and Kinematics.

---

## 5. How to Run

```bash
# Launch with shell script
./run.sh

# Or directly in virtual environment
./.venv/bin/python main.py
```

### Hotkeys & Buttons:
- **Click any car in CAM 01–04:** Lock target and extract 5-parameter profile.
- **`[▶ PLAY / PAUSE]` (or SPACE):** Pause simulation physics to inspect crops and bounding boxes.
- **`[🎯 AUTO-LOCK NEXT CAR]` (or L):** Automatically lock the next car entering CAM 01.
- **`[🧠 TRAIN MODEL]` (or T):** Re-train `TinyCarNet` live on CPU (< 0.2s).
- **`[⟲ UNLOCK]` (or U):** Disengage tracking lock.
- **ESC:** Exit application.
