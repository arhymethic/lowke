# Multi-Camera Video & Timestamp Configuration Registry

This repository contains the video clips and central configuration registry for multi-camera vehicle tracking, cross-camera Re-ID, and speed estimation.

## Files Structure

```
Proj/
├── config.json              # Camera registry, virtual start timestamps, and distance matrix
├── README.md                # Documentation and ingestion formula
└── videos/                  # Raw video clips
    ├── vdo (1).avi          # CAM_01 (31s)
    ├── vdo (2).avi          # CAM_02 (28s)
    ├── vdo (3).avi          # CAM_03 (41s)
    ├── vdo (4).avi          # CAM_04 (46s)
    ├── vdo (5).avi          # CAM_05 (47s)
    ├── vdo (6).avi          # CAM_06 (31s)
    ├── vdo (7).avi          # CAM_07 (31s)
    ├── vdo (8).avi          # CAM_08 (60s)
    └── vdo.avi              # CAM_MASTER (211s master continuous feed)
```

## Camera Registry Summary

| Camera ID | Video File | Duration | Virtual Start Time | Distance to Next |
| :--- | :--- | :--- | :--- | :--- |
| **CAM_01** | `videos/vdo (1).avi` | 31s | `2026-10-24T10:00:00.000` | 350m (to CAM_02) |
| **CAM_02** | `videos/vdo (2).avi` | 28s | `2026-10-24T10:00:25.000` | 500m (to CAM_03) |
| **CAM_03** | `videos/vdo (3).avi` | 41s | `2026-10-24T10:01:00.000` | 450m (to CAM_04) |
| **CAM_04** | `videos/vdo (4).avi` | 46s | `2026-10-24T10:01:40.000` | 400m (to CAM_05) |
| **CAM_05** | `videos/vdo (5).avi` | 47s | `2026-10-24T10:02:25.000` | 350m (to CAM_06) |
| **CAM_06** | `videos/vdo (6).avi` | 31s | `2026-10-24T10:03:10.000` | 300m (to CAM_07) |
| **CAM_07** | `videos/vdo (7).avi` | 31s | `2026-10-24T10:03:45.000` | 500m (to CAM_08) |
| **CAM_08** | `videos/vdo (8).avi` | 60s | `2026-10-24T10:04:20.000` | Terminal |
| **CAM_MASTER** | `videos/vdo.avi` | 211s | `2026-10-24T10:00:00.000` | Full stream |

## Downstream Processing Reference

### Frame-to-Global Timestamp Calculation
For any detection at `frame_idx` within a camera video:

$$\text{Timestamp}_{\text{frame}} = \text{VirtualStartTime} + \left(\frac{\text{frame\_idx}}{\text{fps}}\right) \times 1\text{ second}$$

### Average Section Speed Calculation
When a vehicle with global ID is detected crossing CAM_A at $T_A$ and CAM_B at $T_B$:

$$\Delta t = T_B - T_A \quad (\text{in seconds})$$

$$\text{Speed } (v) = \frac{\text{Distance}_{A \to B}}{\Delta t} \times 3.6 \quad (\text{in km/h})$$
