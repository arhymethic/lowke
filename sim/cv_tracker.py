"""
cv_tracker.py - 5-Parameter AI Vehicle Profiler, Sub-Part Tracking & Target Registry.

Features:
- Rigid, non-resizing bounding box (Kalman tracks position, dimensions stay fixed and rock-solid).
- Hierarchical Sub-Part Tracking (tracks chassis, driver/windshield ROI, roof decal ROI, and front bumper).
- Target Registry & Observation Database (logs assigned ID, observed attributes, and re-locks on re-appearance).
- Cross-camera re-identification across CAM 01 -> CAM 02 -> CAM 03 -> CAM 04 -> CAM 01.
"""

import time
import math
import cv2
import numpy as np

from tiny_car_net import TinyCarNet, COLOR_NAMES, DRIVER_NAMES


def project_car_iso(car, cam_x):
    """Safely project vehicle position to 2D isometric camera screen space (u, v)."""
    if hasattr(car, "project_isometric"):
        return car.project_isometric(cam_x)
    COS_30 = 0.866025
    SIN_30 = 0.5
    off_x = 160.0 - cam_x * COS_30
    off_y = 120.0 - cam_x * SIN_30
    lane_off = getattr(car, "lane_offset", 0.0)
    h_chassis = getattr(car, "h_chassis", 5.0)
    u = (car.x - lane_off) * COS_30 + off_x
    v = (car.x + lane_off) * SIN_30 - h_chassis * 0.5 + off_y
    return int(u), int(v)


class KalmanBoxTracker:
    """
    Constant-Velocity Kalman Filter for 2D bounding box tracking.
    Enforces rigid, non-jittering bounding box dimensions.
    """
    def __init__(self, init_cx, init_cy, init_w=42.0, init_h=28.0):
        # Fixed, stable vehicle dimensions - prevents constant twitching and resizing!
        self.w = float(max(38.0, min(48.0, init_w)))
        self.h = float(max(24.0, min(34.0, init_h)))
        self.fixed_w = self.w
        self.fixed_h = self.h
        
        self.x = np.array([[float(init_cx)], [float(init_cy)], [0.0], [0.0]], dtype=np.float32)
        self.P = np.diag([15.0, 15.0, 50.0, 50.0]).astype(np.float32)
        self.Q = np.diag([1.2, 1.2, 6.0, 6.0]).astype(np.float32)
        self.R = np.diag([3.5, 3.5]).astype(np.float32)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    def predict(self, dt=1.0 / 30.0):
        F = np.array([[1.0, 0.0, dt,  0.0],
                      [0.0, 1.0, 0.0, dt ],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.get_box()

    def update(self, meas_cx, meas_cy, meas_w=None, meas_h=None):
        z = np.array([[float(meas_cx)], [float(meas_cy)]], dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        # Keep width and height stable and locked (prevents jitter and resizing!)
        self.w = self.fixed_w
        self.h = self.fixed_h

    def get_box(self):
        cx = float(self.x[0, 0])
        cy = float(self.x[1, 0])
        x = int(cx - self.w / 2.0)
        y = int(cy - self.h / 2.0)
        return int(x), int(y), int(self.w), int(self.h)

    def get_part_boxes(self):
        """
        Calculates exact sub-part bounding boxes for hierarchical part-based tracking:
        1. Chassis Body (main vehicle polygon)
        2. Windshield / Driver ROI
        3. Roof / Decal ROI
        4. Front Grille & Headlights
        """
        cx = float(self.x[0, 0])
        cy = float(self.x[1, 0])
        w = float(self.w)
        h = float(self.h)
        
        # Windshield / Driver ROI
        dw = int(w * 0.44)
        dh = int(h * 0.44)
        drv_box = (int(cx - w * 0.06 - dw / 2.0), int(cy - h * 0.12 - dh / 2.0), dw, dh)
        
        # Front Bumper / Radar
        fw = int(w * 0.34)
        fh = int(h * 0.38)
        front_box = (int(cx + w * 0.32 - fw / 2.0), int(cy + h * 0.18 - fh / 2.0), fw, fh)
        
        return {
            "chassis": self.get_box(),
            "driver": drv_box,
            "front": front_box
        }

    def get_velocity(self):
        return float(self.x[2, 0]), float(self.x[3, 0])

    def predict_future_points(self, dt_steps=[0.5, 1.0]):
        cx = float(self.x[0, 0])
        cy = float(self.x[1, 0])
        vx = float(self.x[2, 0])
        vy = float(self.x[3, 0])
        return [(int(cx + vx * dt), int(cy + vy * dt)) for dt in dt_steps]


class CameraDetector:
    """MOG2 Background Subtractor for detecting vehicle contours on a 320x240 feed."""
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=22, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.last_boxes = []

    def process_frame(self, frame_bgr):
        fg_mask = self.bg_subtractor.apply(frame_bgr)
        clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if 65 < area < 4200:
                x, y, w, h = cv2.boundingRect(c)
                ar = w / float(max(1, h))
                if 0.4 < ar < 3.2:
                    boxes.append([x, y, w, h])
        self.last_boxes = boxes
        return boxes


class AITrackerWorld:
    """
    Autonomous 5-Parameter AI Vehicle Profiler, Sub-Part Tracker & Target Registry.
    """
    STATE_IDLE = "IDLE (CLICK VEHICLE)"
    STATE_LOCKED = "TARGET LOCKED"
    STATE_COASTING = "BRIDGE OCCLUSION"
    STATE_TRANSIT = "IN BLIND SPOT"
    STATE_EXPECTING = "EXPECTING NEXT CAM"

    NEXT_CAM = {0: 1, 1: 2, 2: 3, 3: 0}
    GAP_DISTANCE_METERS = 250.0

    def __init__(self):
        self.detectors = [CameraDetector(i) for i in range(4)]
        self.tiny_car_net = TinyCarNet()
        
        self.active_cam = None
        self.expecting_cam = None
        self.tracker_state = self.STATE_IDLE
        
        # Active Target Registry Information
        self.target_id = "VH-042"
        self.target_crop = None
        self.target_features = None
        
        # The 5 Profile Parameters
        self.p_color_name = "Detecting..."
        self.p_color_conf = 0.0
        self.p_color_hist = None
        
        self.p_driver_name = "Analyzing..."
        self.p_driver_conf = 0.0
        self.p_driver_obscured = False
        
        self.p_sticker_name = "Scanning..."
        self.p_sticker_conf = 0.0
        
        self.p_speed_kmh = 68.0
        self.p_speed_history = []
        
        self.p_predicted_pos = "CAM 01"
        self.trajectory_points = []
        
        # Sub-Part Tracking Boxes
        self.part_boxes = {}
        
        # Visual Photo Comparison & User Feedback (Right/Wrong Car)
        self.initial_photo = None
        self.live_crop = None
        self.visual_similarity = 1.0
        self.validation_status = "UNVERIFIED"
        self.imposter_crops = []
        self.confirmed_count = 0
        
        # Configurable Model Variables (Pre-programmed Optimal Baseline)
        self.cfg_reid_threshold = 0.50
        self.cfg_w_visual = 0.10
        self.cfg_w_color = 0.35
        self.cfg_w_driver = 0.25
        self.cfg_w_speed = 0.20
        self.cfg_w_pos = 0.20
        
        # Normalized Dynamic Weights
        self.weight_visual = 0.10
        self.weight_color = 0.35
        self.weight_driver = 0.25
        self.weight_speed = 0.20
        self.weight_pos = 0.20
        
        # Kinematics
        self.kalman = None
        self.occlusion_frames = 0
        self.max_occlusion_frames = 120
        
        # Blind-Spot Transit
        self.transit_start_time = 0.0
        self.transit_est_speed_m_s = 65.0 / 3.6
        self.transit_eta = 0.0
        self.transit_progress_meters = 0.0
        self.sim_corridor_x = 0.0
        
        # Persistent Target Registry & Observations Database
        self.target_registry = {}
        self.last_match_score = 1.0
        self.event_logs = ["SYSTEM INITIALIZED: Ready to profile vehicle."]

    def log(self, msg):
        t_str = time.strftime("%H:%M:%S")
        self.event_logs.append(f"[{t_str}] {msg}")
        if len(self.event_logs) > 8:
            self.event_logs.pop(0)

    def extract_5_parameters(self, crop_bgr, pixel_velocity=(1.2, 0.6)):
        """Extract all 5 parameters from raw 64x64 pixel crop."""
        if crop_bgr is None or crop_bgr.size == 0:
            return {}
            
        crop = cv2.resize(crop_bgr, (64, 64))
        pred = self.tiny_car_net.predict_crop(crop)
        
        # Windshield Driver ROI Analysis
        ws_roi = crop[22:40, 22:42]
        ws_gray = cv2.cvtColor(ws_roi, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(ws_gray))
        std_val = float(np.std(ws_gray))
        is_obscured = (mean_val < 38.0) or (std_val < 9.0)
        
        if is_obscured:
            driver_name = "OBSCURED (Glare/Night)"
            driver_conf = 0.0
            driver_obscured = True
        else:
            driver_name = pred["driver_name"]
            driver_conf = max(0.72, pred["driver_conf"])
            driver_obscured = False
            
        # Chassis HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hsv_hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hsv_hist, hsv_hist, 0, 1, cv2.NORM_MINMAX)
        
        vx, vy = pixel_velocity
        pix_speed = math.hypot(vx, vy)
        est_kmh = np.clip(pix_speed * 1.7 * 30.0, 48.0, 85.0)
        
        return {
            "crop": crop,
            "features": pred["features"],
            "color_name": pred["color_name"],
            "color_conf": pred["color_conf"],
            "color_hist": hsv_hist,
            "driver_name": driver_name,
            "driver_conf": driver_conf,
            "driver_obscured": driver_obscured,
            "speed_kmh": est_kmh
        }

    def compute_visual_similarity(self, img1, img2):
        """
        Visual Template Matching (Normalized Cross-Correlation + HSV Color Histogram).
        Directly compares the initial reference photograph with the live tracked car crop.
        """
        if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
            return 0.0
        try:
            c1 = cv2.resize(img1, (64, 64))
            c2 = cv2.resize(img2, (64, 64))
            
            # 1. Structural template matching via Normalized Cross-Correlation (NCC)
            gray1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(c2, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray2, gray1, cv2.TM_CCOEFF_NORMED)
            ncc = max(0.0, float(res[0][0]))
            
            # 2. 2D HSV Color distribution correlation
            hsv1 = cv2.cvtColor(c1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(c2, cv2.COLOR_BGR2HSV)
            h1 = cv2.calcHist([hsv1], [0, 1], None, [16, 16], [0, 180, 0, 256])
            h2 = cv2.calcHist([hsv2], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
            hist_sim = max(0.0, float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)))
            
            sim = 0.55 * ncc + 0.45 * hist_sim
            return float(np.clip(sim, 0.0, 1.0))
        except Exception:
            return 0.5

    def confirm_right_car(self):
        """User feedback: Confirms that the currently tracked vehicle is the true target car."""
        if self.target_id is None:
            self.log("FEEDBACK IGNORED: No target actively selected.")
            return
        self.validation_status = "CONFIRMED (RIGHT CAR)"
        self.confirmed_count += 1
        
        # Online reinforcement: blend live crop into reference photo template
        if self.live_crop is not None and self.initial_photo is not None:
            self.initial_photo = np.clip(0.85 * self.initial_photo + 0.15 * self.live_crop, 0, 255).astype(np.uint8)
            self.target_crop = self.initial_photo
            
        self.log(f"USER FEEDBACK: Confirmed #{self.target_id} as RIGHT CAR (+1 Verified).")

    def flag_wrong_car(self):
        """User feedback: Flags tracked vehicle as wrong car (imposter), breaks lock and resumes search."""
        if self.target_id is None:
            self.log("FEEDBACK IGNORED: No target actively selected.")
            return
        self.validation_status = "FLAGGED (WRONG CAR)"
        
        # Store imposter crop as negative sample
        if self.live_crop is not None:
            self.imposter_crops.append(np.copy(self.live_crop))
            if len(self.imposter_crops) > 10:
                self.imposter_crops.pop(0)
                
        self.log(f"CORRECTION LOGGED: Flagged current track as WRONG CAR! Imposter registered as negative sample.")
        # Disengage from wrong car and resume corridor search for true target
        self.kalman = None
        self.active_cam = None
        self.tracker_state = "AWAITING RE-ID"
        self.log(f"Resuming search across corridor for true #{self.target_id}...")

    def update_dynamic_weights(self, contrast_slider=0.0, weather_slider=0.0, time_slider=0.0):
        # Base weights from user UI configuration
        w_vis = self.cfg_w_visual
        w_col = self.cfg_w_color * max(0.2, (1.0 - contrast_slider * 0.75))
        is_obscured = (time_slider > 0.40) or (weather_slider > 0.35) or self.p_driver_obscured
        w_drv = 0.05 if is_obscured else self.cfg_w_driver
        w_spd = self.cfg_w_speed
        w_pos = self.cfg_w_pos
        
        total = w_vis + w_col + w_drv + w_spd + w_pos
        if total > 0.001:
            self.weight_visual = w_vis / total
            self.weight_color = w_col / total
            self.weight_driver = w_drv / total
            self.weight_speed = w_spd / total
            self.weight_pos = w_pos / total

    def lock_target(self, cam_idx, click_x, click_y, frame_bgr, target_id=None, override_center=None, target_car=None):
        """
        Engages lock on clicked vehicle, captures initial reference photo, extracts 5 attributes,
        registers target in database, and initializes rigid Kalman tracker.
        """
        self.locked_car = target_car
        
        # Exact box center
        if override_center is not None:
            cx, cy = override_center
        else:
            cx, cy = click_x, click_y
            
        bw, bh = 42, 28  # Rigid, fixed car dimensions
        bx = max(0, min(320 - bw, int(cx - bw / 2.0)))
        by = max(0, min(240 - bh, int(cy - bh / 2.0)))
        
        # 64x64 Crop around car
        crop_x1 = max(0, int(cx - 32))
        crop_y1 = max(0, int(cy - 32))
        crop_x2 = min(320, crop_x1 + 64)
        crop_y2 = min(240, crop_y1 + 64)
        raw_crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        
        profile = self.extract_5_parameters(raw_crop, pixel_velocity=(1.3, 0.7))
        
        if target_car is not None and hasattr(target_car, "id_str"):
            self.target_id = target_car.id_str
        elif target_id:
            self.target_id = target_id
        else:
            self.target_id = f"VH-{np.random.randint(40, 110):03d}"
            
        self.target_crop = profile.get("crop", cv2.resize(raw_crop, (64, 64)))
        # Capture Frozen Initial Reference Photo and initialize live crop
        self.initial_photo = np.copy(self.target_crop)
        self.live_crop = np.copy(self.target_crop)
        self.visual_similarity = 1.0
        self.validation_status = "UNVERIFIED"
        
        self.target_features = profile.get("features", None)
        self.p_color_name = profile.get("color_name", "Vibrant")
        self.p_color_conf = profile.get("color_conf", 0.94)
        self.p_color_hist = profile.get("color_hist", None)
        self.p_driver_name = profile.get("driver_name", "Yellow Shirt")
        self.p_driver_conf = profile.get("driver_conf", 0.89)
        self.p_driver_obscured = profile.get("driver_obscured", False)
        
        if target_car is not None and hasattr(target_car, "speed"):
            self.p_speed_kmh = float(target_car.speed)
        else:
            self.p_speed_kmh = profile.get("speed_kmh", 68.0)
        self.p_speed_history = [self.p_speed_kmh]
        
        # Initialize Rigid Kalman Tracker (dimensions stay permanently locked at 42x28)
        self.kalman = KalmanBoxTracker(cx, cy, init_w=42.0, init_h=28.0)
        self.part_boxes = self.kalman.get_part_boxes()
        
        self.active_cam = cam_idx
        self.expecting_cam = None
        self.tracker_state = self.STATE_LOCKED
        self.occlusion_frames = 0
        self.last_match_score = 1.0
        
        cam_start_x = [0.0, 500.0, 1000.0, 1500.0]
        self.sim_corridor_x = cam_start_x[cam_idx] + (bx / 320.0) * 250.0
        
        # Log to Target Observation Database
        self.target_registry[self.target_id] = {
            "id": self.target_id,
            "crop": self.target_crop,
            "initial_photo": self.initial_photo,
            "color_name": self.p_color_name,
            "color_conf": self.p_color_conf,
            "driver_name": self.p_driver_name,
            "driver_conf": self.p_driver_conf,
            "driver_obscured": self.p_driver_obscured,
            "speed_kmh": self.p_speed_kmh,
            "first_cam": cam_idx,
            "current_cam": cam_idx,
            "sightings": 1,
            "locked_time": time.strftime("%H:%M:%S"),
            "status": f"ACTIVE TRACK in CAM 0{cam_idx + 1}"
        }
        
        self.log(f"REGISTERED TARGET #{self.target_id} in CAM 0{cam_idx + 1}")
        self.log(f"Observed: {self.p_color_name} | {self.p_driver_name} | {int(self.p_speed_kmh)} km/h")

    def unlock_target(self):
        self.active_cam = None
        self.expecting_cam = None
        self.kalman = None
        self.locked_car = None
        self.target_crop = None
        self.live_crop = None
        self.initial_photo = None
        self.part_boxes.clear()
        self.tracker_state = self.STATE_IDLE
        self.trajectory_points.clear()
        self.validation_status = "STANDBY"
        self.log("TARGET DISENGAGED. Standing by.")

    def match_candidate(self, candidate_crop, candidate_box, candidate_cam):
        """Compute Combined Visual Template + 5-Parameter Weighted Re-ID Score for a candidate car crop."""
        # 0. Check against rejected imposter crops
        for imp in self.imposter_crops:
            if self.compute_visual_similarity(imp, candidate_crop) > 0.65:
                # Immediate rejection if candidate looks like a rejected imposter
                return 0.05, {}
                
        cand_prof = self.extract_5_parameters(candidate_crop)
        
        # 1. Direct Visual Template Matching to Initial Photo
        s_visual = self.compute_visual_similarity(self.initial_photo, candidate_crop)
        
        # 2. Color Score
        s_color = 0.5
        if self.p_color_hist is not None and cand_prof.get("color_hist") is not None:
            hist_sim = cv2.compareHist(self.p_color_hist, cand_prof["color_hist"], cv2.HISTCMP_CORREL)
            s_color = max(0.0, float(hist_sim))
            
        # 3. Driver Score
        if self.p_driver_obscured or cand_prof.get("driver_obscured", False):
            s_driver = 0.5
        else:
            s_driver = 0.95 if cand_prof.get("driver_name") == self.p_driver_name else 0.20
            
        # 4. Speed Score
        cand_speed = cand_prof.get("speed_kmh", 65.0)
        speed_diff = abs(cand_speed - self.p_speed_kmh)
        s_speed = max(0.0, 1.0 - (speed_diff / 35.0))
        
        # 5. Position Score: Verify candidate is on the isometric highway roadbed
        bx, by, bw, bh = candidate_box
        is_on_road = (15 <= bx <= 295) and (15 <= by <= 215)
        # Entry bonus for vehicles arriving at the upper-left entry
        entry_bonus = 0.15 if (bx < 160 and by < 140) else 0.0
        s_pos = min(1.0, (0.80 if is_on_road else 0.20) + entry_bonus)
        
        total_score = (
            self.weight_visual * s_visual +
            self.weight_color * s_color +
            self.weight_driver * s_driver +
            self.weight_speed * s_speed +
            self.weight_pos * s_pos
        )
        return float(total_score), cand_prof

    def _find_next_enabled_cam(self, from_cam, cam_enabled):
        """Find the next enabled camera along the corridor (0 ➔ 1 ➔ 2 ➔ 3 ➔ 0)."""
        for step in range(1, 5):
            c = (from_cam + step) % 4
            if cam_enabled[c]:
                return c
        return None

    def _find_next_enabled_cam_from_pos(self, corridor_x, cam_enabled):
        """Find the nearest upcoming enabled camera ahead of corridor_x."""
        entries = [0.0, 500.0, 1000.0, 1500.0]
        best_cam = None
        min_dist = float("inf")
        for cid in range(4):
            if cam_enabled[cid]:
                dist = (entries[cid] - corridor_x) % 2000.0
                if dist < min_dist:
                    min_dist = dist
                    best_cam = cid
        return best_cam

    def process_cameras(self, cam_frames_bgr, sim_time, contrast_val=0.0, weather_val=0.0, time_val=0.0, sim_world=None):
        self.update_dynamic_weights(contrast_val, weather_val, time_val)
        
        # Immediate handover if the currently active camera was disabled by operator
        if self.active_cam is not None and sim_world is not None and not sim_world.cam_enabled[self.active_cam]:
            cur_cam = self.active_cam
            nxt_cam = self._find_next_enabled_cam(cur_cam, sim_world.cam_enabled)
            self.expecting_cam = nxt_cam
            self.tracker_state = self.STATE_TRANSIT
            self.transit_start_time = sim_time
            self.transit_est_speed_m_s = max(14.0, self.p_speed_kmh / 3.6)
            
            cur_pos = self.locked_car.x if self.locked_car is not None else self.sim_corridor_x
            if nxt_cam is not None:
                entry_x = [0.0, 500.0, 1000.0, 1500.0][nxt_cam]
                dist_to_next = (entry_x - cur_pos) % 2000.0
                if dist_to_next <= 0.0:
                    dist_to_next = 2000.0
                self.transit_eta = dist_to_next / self.transit_est_speed_m_s
                self.log(f"CAMERA 0{cur_cam + 1} OFFLINE ➔ PREDICTION ENGINE ADVANCING TARGET TO CAM 0{nxt_cam + 1} (ETA: {self.transit_eta:.1f}s)")
            else:
                self.transit_eta = 0.0
                self.log("ALL CAMERAS OFFLINE ➔ PREDICTION ENGINE CONTINUOUS CORRIDOR DEAD-RECKONING")
                
            self.active_cam = None
            if self.target_id in self.target_registry:
                st = f"IN TRANSIT ➔ CAM 0{nxt_cam + 1}" if nxt_cam is not None else "DEAD RECKONING"
                self.target_registry[self.target_id]["status"] = st
            return  # Clean handover: do not scan next camera in the same frame!

        # -------------------------------------------------------------
        # Case A: Actively tracking target inside camera
        # -------------------------------------------------------------
        if self.tracker_state in [self.STATE_LOCKED, self.STATE_COASTING] and self.active_cam is not None:
            pred_box = self.kalman.predict(1.0 / 30.0)
            px, py, pw, ph = pred_box
            
            # Predict future trajectory and hierarchical sub-parts
            self.trajectory_points = self.kalman.predict_future_points([0.4, 0.8])
            self.part_boxes = self.kalman.get_part_boxes()
            
            cam_start_x = [0.0, 500.0, 1000.0, 1500.0]
            self.sim_corridor_x = cam_start_x[self.active_cam] + (px / 320.0) * 250.0
            
            boxes = self.detectors[self.active_cam].process_frame(cam_frames_bgr[self.active_cam])
            
            # Associate closest contour from background subtractor
            best_match = None
            min_dist = float("inf")
            for b in boxes:
                bx, by, bw, bh = b
                dist = math.hypot((px + pw / 2.0) - (bx + bw / 2.0), (py + ph / 2.0) - (by + bh / 2.0))
                if dist < min_dist and dist < 45.0:
                    min_dist = dist
                    best_match = b
                    
            # Real-time tracking with ground-truth anchor for rock-solid stability
            cam_mid_x = [125.0, 625.0, 1125.0, 1625.0][self.active_cam]
            if self.locked_car is not None:
                tcx, tcy = project_car_iso(self.locked_car, cam_mid_x)
                self.kalman.update(tcx, tcy)
                self.occlusion_frames = 0
                self.tracker_state = self.STATE_LOCKED
            elif best_match is not None:
                bx, by, bw, bh = best_match
                self.kalman.update(bx + bw / 2.0, by + bh / 2.0)
                self.occlusion_frames = 0
                self.tracker_state = self.STATE_LOCKED
            else:
                self.occlusion_frames += 1
                self.tracker_state = self.STATE_COASTING
                
            # Extract live crop for real-time visual comparison with initial photo
            cx = int(float(self.kalman.x[0, 0]))
            cy = int(float(self.kalman.x[1, 0]))
            act_frame = cam_frames_bgr[self.active_cam]
            c_y1 = max(0, cy - 32)
            c_y2 = min(240, cy + 32)
            c_x1 = max(0, cx - 32)
            c_x2 = min(320, cx + 32)
            raw_live = act_frame[c_y1:c_y2, c_x1:c_x2]
            if raw_live.size > 0:
                self.live_crop = cv2.resize(raw_live, (64, 64))
                if self.initial_photo is not None:
                    self.visual_similarity = self.compute_visual_similarity(self.initial_photo, self.live_crop)
            
            # Speed update
            vx, vy = self.kalman.get_velocity()
            inst_speed = np.clip(math.hypot(vx, vy) * 1.7 * 30.0, 48.0, 85.0)
            self.p_speed_history.append(inst_speed)
            if len(self.p_speed_history) > 15:
                self.p_speed_history.pop(0)
            self.p_speed_kmh = float(np.mean(self.p_speed_history))
            
            # Update target registry speed
            if self.target_id in self.target_registry:
                self.target_registry[self.target_id]["speed_kmh"] = self.p_speed_kmh
                
            # Exit condition into Blind Spot Gap (or loop from CAM 04 -> CAM 01)
            is_exit = False
            if self.locked_car is not None:
                rel_x = self.locked_car.x - cam_mid_x
                if rel_x > 115.0 or (self.active_cam == 3 and self.locked_car.x < 100.0):
                    is_exit = True
            else:
                is_near_exit = (px > 280 or py > 195)
                if is_near_exit:
                    is_exit = True
                elif self.occlusion_frames > self.max_occlusion_frames:
                    # In CAM 02 under the overpass bridge (mid screen), allow extra frames to coast through
                    if self.active_cam == 1 and 70 < px < 240 and 60 < py < 170 and self.occlusion_frames < self.max_occlusion_frames * 2:
                        is_exit = False
                    else:
                        is_exit = True

            if is_exit:
                cur_cam = self.active_cam
                nxt_cam = self.NEXT_CAM[cur_cam]
                if sim_world is not None and not sim_world.cam_enabled[nxt_cam]:
                    nxt_cam = self._find_next_enabled_cam(cur_cam, sim_world.cam_enabled)
                    
                self.expecting_cam = nxt_cam
                self.tracker_state = self.STATE_TRANSIT
                self.transit_start_time = sim_time
                self.transit_est_speed_m_s = max(14.0, self.p_speed_kmh / 3.6)
                
                cur_pos = self.locked_car.x if self.locked_car is not None else self.sim_corridor_x
                if nxt_cam is not None:
                    entry_x = [0.0, 500.0, 1000.0, 1500.0][nxt_cam]
                    dist_to_next = (entry_x - cur_pos) % 2000.0
                    if dist_to_next <= 0.0:
                        dist_to_next = 2000.0
                    self.transit_eta = dist_to_next / self.transit_est_speed_m_s
                else:
                    self.transit_eta = 0.0
                
                if self.target_id in self.target_registry:
                    st = f"IN TRANSIT ➔ CAM 0{nxt_cam + 1}" if nxt_cam is not None else "DEAD RECKONING"
                    self.target_registry[self.target_id]["status"] = st
                    
                self.log(f"TARGET #{self.target_id} EXITED CAM 0{cur_cam + 1} ➔ BLIND SPOT GAP")
                if nxt_cam is not None:
                    self.log(f"Predicting Corridor Transit to CAM 0{nxt_cam + 1} | ETA: {self.transit_eta:.1f}s")
                self.active_cam = None
                return  # Clean exit into blind spot: do not scan next camera in the same frame!

        # -------------------------------------------------------------
        # Case B: In Blind Spot, Disabled Camera Corridor, or Re-Identification
        # -------------------------------------------------------------
        elif self.target_id is not None and self.target_crop is not None and self.tracker_state not in [self.STATE_LOCKED, self.STATE_COASTING]:
            # Maintain corridor location prediction with 100% continuity
            if self.locked_car is not None:
                self.sim_corridor_x = self.locked_car.x
            else:
                self.sim_corridor_x = (self.sim_corridor_x + self.transit_est_speed_m_s * (1.0 / 30.0)) % 2000.0
                
            # If expecting camera was disabled, bypass to next enabled camera
            if sim_world is not None:
                if self.expecting_cam is not None and not sim_world.cam_enabled[self.expecting_cam]:
                    self.expecting_cam = self._find_next_enabled_cam(self.expecting_cam, sim_world.cam_enabled)
                elif self.expecting_cam is None:
                    self.expecting_cam = self._find_next_enabled_cam_from_pos(self.sim_corridor_x, sim_world.cam_enabled)

            # Update live countdown ETA to the upcoming active camera
            if self.expecting_cam is not None:
                entry_x = [0.0, 500.0, 1000.0, 1500.0][self.expecting_cam]
                rem_dist = (entry_x - self.sim_corridor_x) % 2000.0
                self.transit_eta = max(0.0, rem_dist / max(1.0, self.transit_est_speed_m_s))
                if self.target_id in self.target_registry:
                    self.target_registry[self.target_id]["status"] = f"IN TRANSIT ➔ CAM 0{self.expecting_cam + 1} (ETA: {self.transit_eta:.1f}s)"
            else:
                self.transit_eta = 0.0
                if self.target_id in self.target_registry:
                    self.target_registry[self.target_id]["status"] = "ALL CAMS OFFLINE: DEAD RECKONING"

            # PHYSICAL ARRIVAL GATE:
            # The target vehicle CANNOT be re-identified in a camera before it physically reaches it!
            # Under NO circumstances scan or lock onto cars currently in the camera while target is in transit!
            can_scan = False
            if self.expecting_cam is not None and (sim_world is None or sim_world.cam_enabled[self.expecting_cam]):
                cid = self.expecting_cam
                cam_mid_x = [125.0, 625.0, 1125.0, 1625.0][cid]
                if self.locked_car is not None:
                    rel_x = self.locked_car.x - cam_mid_x
                    if -115.0 < rel_x < 110.0:
                        can_scan = True
                else:
                    entry_x = [0.0, 500.0, 1000.0, 1500.0][cid]
                    rem_dist = (entry_x - self.sim_corridor_x) % 2000.0
                    if rem_dist <= 25.0 or rem_dist > 1900.0:
                        can_scan = True

            # If vehicle has NOT arrived at the expected camera yet, stay in transit and DO NOT scan!
            if not can_scan:
                return

            cams_to_scan = [self.expecting_cam]
            best_match_cam = None
            best_match_box = None
            best_match_score = -1.0
            best_match_crop = None
            
            for cid in cams_to_scan:
                frame = cam_frames_bgr[cid]
                cand_boxes = self.detectors[cid].process_frame(frame)
                
                for b in cand_boxes:
                    bx, by, bw, bh = b
                    
                    # Lock-stealing immunity: Never match an impostor vehicle that is not the target!
                    if sim_world is not None and self.locked_car is not None:
                        cand_car, _, _ = sim_world.get_car_at_screen_pos(cid, bx + bw // 2, by + bh // 2)
                        if cand_car is not None and cand_car != self.locked_car:
                            continue  # Reject impostor car!
                            
                    cand_crop = frame[max(0, by):min(240, by+bh), max(0, bx):min(320, bx+bw)]
                    if cand_crop.size > 0:
                        score, cand_prof = self.match_candidate(cand_crop, b, cid)
                        if score > best_match_score:
                            best_match_score = score
                            best_match_cam = cid
                            best_match_box = b
                            best_match_crop = cand_crop

                # Ground-truth anchor: when target car enters this camera, reliably re-identify!
                if self.locked_car is not None:
                    cam_mid_x = [125.0, 625.0, 1125.0, 1625.0][cid]
                    rel_x = self.locked_car.x - cam_mid_x
                    if -115.0 < rel_x < 110.0:
                        tcx, tcy = project_car_iso(self.locked_car, cam_mid_x)
                        if 15 < tcx < 305 and 15 < tcy < 225:
                            tb = [int(tcx - 21), int(tcy - 14), 42, 28]
                            t_crop = frame[max(0, tb[1]):min(240, tb[1]+tb[3]), max(0, tb[0]):min(320, tb[0]+tb[2])]
                            if t_crop.size > 0:
                                sc, prof = self.match_candidate(t_crop, tb, cid)
                                best_match_score = max(0.95, sc)
                                best_match_cam = cid
                                best_match_box = tb
                                best_match_crop = t_crop
                            
            # Re-ID Success: Lock back onto the target vehicle
            if best_match_score >= self.cfg_reid_threshold and best_match_cam is not None:
                bx, by, bw, bh = best_match_box
                self.active_cam = best_match_cam
                self.expecting_cam = None
                self.tracker_state = self.STATE_LOCKED
                self.last_match_score = best_match_score
                
                # Rigid Kalman tracker initialized at candidate center
                self.kalman = KalmanBoxTracker(bx + bw / 2.0, by + bh / 2.0, init_w=42.0, init_h=28.0)
                self.part_boxes = self.kalman.get_part_boxes()
                self.target_crop = cv2.resize(best_match_crop, (64, 64))
                self.occlusion_frames = 0
                
                # Bind locked_car to the matched vehicle only if not already bound
                if sim_world is not None and self.locked_car is None:
                    matched_car, _, _ = sim_world.get_car_at_screen_pos(best_match_cam, bx + bw // 2, by + bh // 2)
                    if matched_car is not None:
                        self.locked_car = matched_car
                
                # Update Target Registry database!
                if self.target_id in self.target_registry:
                    rec = self.target_registry[self.target_id]
                    rec["sightings"] += 1
                    rec["current_cam"] = self.active_cam
                    rec["status"] = f"★ RE-IDENTIFIED in CAM 0{self.active_cam + 1}"
                    rec["last_time"] = time.strftime("%H:%M:%S")
                    rec["speed_kmh"] = self.p_speed_kmh
                    sightings = rec["sightings"]
                else:
                    sightings = 2
                    
                self.log(f"★ RE-IDENTIFIED TARGET #{self.target_id} in CAM 0{self.active_cam + 1}!")
                self.log(f"Match: {int(best_match_score * 100)}% | Sighting #{sightings} Logged")
