"""
main.py - Linear Highway Corridor CCTV Re-ID & Multi-Attribute Tracking Workstation.
Aesthetic: Professional Late-90s / 2000s Industrial Law Enforcement Surveillance System.
Fully Adaptive: Dynamically scales to any resolution (1280x720, 1600x900, 1920x1080) with pygame.RESIZABLE.
"""

import sys
import time
import argparse
import pygame
import cv2
import numpy as np

from sim_world import SimWorld
from cv_tracker import AITrackerWorld
from hud_overlay import HUDOverlay
from tiny_car_net import COLOR_NAMES, DRIVER_NAMES, STICKER_NAMES


def train_tiny_car_net_live(ai_tracker, sim_world):
    """
    Generate synthetic dataset from simulator cars and train TinyCarNet live.
    Trains Chassis Paint Color (10 classes) and Driver Avatar (8 classes) in < 0.15s.
    """
    print("[TRAINING] Generating 360 high-precision vehicle samples from simulator...")
    ai_tracker.log("MODEL TRAINING: Sampling 360 vehicle crops...")
    
    samples_X = []
    labels_color = []
    labels_driver = []
    
    for _ in range(360):
        c_idx = np.random.randint(0, 10)
        d_idx = np.random.randint(0, 8)
        
        patch = np.zeros((64, 64, 3), dtype=np.uint8)
        patch[:] = (55, 68, 38)  # PZ grass tone
        
        # Body chassis
        base_col = sim_world.cars[0].VIBRANT_PALETTE[c_idx]
        cv2.rectangle(patch, (12, 16), (52, 50), base_col, -1)
        
        # Windshield
        cv2.rectangle(patch, (22, 22), (42, 40), (48, 62, 74), -1)
        if d_idx > 0:
            drv_col = sim_world.cars[0].DRIVER_SHIRTS[d_idx - 1]
            cv2.circle(patch, (32, 29), 3, (235, 195, 160), -1)
            cv2.rectangle(patch, (28, 32), (36, 38), drv_col, -1)
            
        noise = np.random.randint(-10, 10, (64, 64, 3))
        patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        feat = ai_tracker.tiny_car_net.extract_patch_features(patch)
        samples_X.append(feat)
        labels_color.append(c_idx)
        labels_driver.append(d_idx)
        
    X = np.array(samples_X, dtype=np.float32)
    yc = np.array(labels_color, dtype=np.int64)
    yd = np.array(labels_driver, dtype=np.int64)
    
    t0 = time.time()
    loss, acc = ai_tracker.tiny_car_net.train_on_samples(X, yc, yd, epochs=30, lr=0.035)
    dur = time.time() - t0
    
    print(f"[TRAINING] Complete in {dur:.2f}s | Final Loss: {loss:.3f} | Accuracy: {acc*100:.1f}%")
    ai_tracker.log(f"TRAIN COMPLETE ({dur:.1f}s): Acc {acc*100:.1f}% | Loss {loss:.2f}")


def auto_lock_next_cam01_car(ai_tracker, sim_world):
    """Pick car currently inside the first available enabled camera coverage and lock onto it."""
    cam_targets = [i for i in range(4) if sim_world.cam_enabled[i]]
    if not cam_targets:
        ai_tracker.log("All cameras offline. Enable a camera to auto-lock.")
        return
        
    cam_idx = cam_targets[0]
    cam_x = sim_world.cam_x_centers[cam_idx]
    cam_cars = [c for c in sim_world.cars if abs(c.x - cam_x) < 95.0]
    if cam_cars:
        chosen = max(cam_cars, key=lambda c: c.x)
        sim_world.render_cctv_views()
        raw_rgb = pygame.surfarray.array3d(sim_world.cam_surfs[cam_idx]).transpose(1, 0, 2)
        frame_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
        
        cos30 = 0.866025
        sin30 = 0.5
        off_x = 160.0 - cam_x * cos30
        off_y = 120.0 - cam_x * sin30
        click_x = int((chosen.x - chosen.lane_offset) * cos30 + off_x)
        click_y = int((chosen.x + chosen.lane_offset) * sin30 + off_y)
        ai_tracker.lock_target(cam_idx, click_x, click_y, frame_bgr, target_id=chosen.id_str, override_center=(click_x, click_y), target_car=chosen)
    else:
        ai_tracker.log(f"No cars currently in CAM 0{cam_idx + 1}. Waiting...")


def main():
    parser = argparse.ArgumentParser(description="Linear Highway Corridor CCTV Re-ID")
    parser.add_argument("--train", action="store_true", help="Pre-train model on startup")
    parser.add_argument("--width", type=int, default=1280, help="Initial window width")
    parser.add_argument("--height", type=int, default=720, help="Initial window height")
    args = parser.parse_args()
    
    pygame.init()
    pygame.display.set_caption("PELCO/TRAFFIC-PRO v4.8 - HIGHWAY CORRIDOR SURVEILLANCE & RE-ID")
    
    # Fully adaptive window with RESIZABLE flag
    screen = pygame.display.set_mode((args.width, args.height), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    
    sim_world = SimWorld()
    ai_tracker = AITrackerWorld()
    hud = HUDOverlay(args.width, args.height)
    
    # Fast 1-second warm-up training so model is accurate out of the box
    train_tiny_car_net_live(ai_tracker, sim_world)
    
    cam_frames_bgr = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(4)]
    
    sim_time = 0.0
    running = True
    is_paused = False
    
    print("[SYSTEM] Pelco/Traffic-Pro CCTV Workstation successfully initialized.")
    print("[SYSTEM] Mode: Adaptive Resolution | Aesthetic: Professional Workstation | Corridor: 2000m")
    
    has_initial_lock = False
    
    while running:
        dt = clock.tick(60) / 1000.0
        dt = min(0.05, dt)
        
        # -------------------------------------------------------------
        # 1. Event Handling
        # -------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
                
            elif event.type in (pygame.VIDEORESIZE, getattr(pygame, "WINDOWRESIZED", 32777), getattr(pygame, "WINDOWSIZECHANGED", 32778)):
                # In SDL2/pygame-ce on Hyprland (Wayland), the surface is automatically resized.
                # DO NOT call pygame.display.set_mode() here, as destroying/re-creating the surface
                # triggers an infinite resize loop with Hyprland's tiling manager!
                w = getattr(event, "w", None) or getattr(event, "x", None)
                h = getattr(event, "h", None) or getattr(event, "y", None)
                if w and h:
                    hud.update_layout(w, h)
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                elif event.key == pygame.K_SPACE:
                    is_paused = not is_paused
                elif event.key == pygame.K_r:
                    ai_tracker.confirm_right_car()
                elif event.key == pygame.K_w:
                    ai_tracker.flag_wrong_car()
                elif event.key == pygame.K_l:
                    auto_lock_next_cam01_car(ai_tracker, sim_world)
                elif event.key == pygame.K_u:
                    ai_tracker.unlock_target()
                elif event.key in (pygame.K_1, pygame.K_KP1):
                    sim_world.toggle_camera(0)
                elif event.key in (pygame.K_2, pygame.K_KP2):
                    sim_world.toggle_camera(1)
                elif event.key in (pygame.K_3, pygame.K_KP3):
                    sim_world.toggle_camera(2)
                elif event.key in (pygame.K_4, pygame.K_KP4):
                    sim_world.toggle_camera(3)
                    
            # Pass events to HUD (sliders, buttons, and resize)
            action = hud.handle_event(event)
            if action == "pause":
                is_paused = not is_paused
            elif action == "lock_next":
                auto_lock_next_cam01_car(ai_tracker, sim_world)
            elif action == "right_car":
                ai_tracker.confirm_right_car()
            elif action == "wrong_car":
                ai_tracker.flag_wrong_car()
            elif action == "unlock":
                ai_tracker.unlock_target()
            elif action and action.startswith("toggle_cam_"):
                cam_idx = int(action.split("_")[-1])
                sim_world.toggle_camera(cam_idx)
                
            # Click on CCTV feeds to lock target (only if camera is active)
            if not action and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                click_info = hud.get_cctv_click(event.pos)
                if click_info is not None:
                    cid, lx, ly = click_info
                    if sim_world.cam_enabled[cid]:
                        clicked_car, best_u, best_v = sim_world.get_car_at_screen_pos(cid, lx, ly)
                        raw_rgb = pygame.surfarray.array3d(sim_world.cam_surfs[cid]).transpose(1, 0, 2)
                        frame_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
                        
                        t_id = clicked_car.id_str if clicked_car else None
                        center_pt = (best_u, best_v) if clicked_car else (lx, ly)
                        ai_tracker.lock_target(cid, lx, ly, frame_bgr, target_id=t_id, override_center=center_pt, target_car=clicked_car)
                    
        # Synchronize All AI Model Hyperparameters with AI Tracker
        ai_tracker.cfg_reid_threshold = hud.slider_reid_thresh.val
        ai_tracker.cfg_w_visual = hud.slider_w_visual.val
        ai_tracker.cfg_w_color = hud.slider_w_color.val
        ai_tracker.cfg_w_driver = hud.slider_w_driver.val
        ai_tracker.cfg_w_speed = hud.slider_w_speed.val
        ai_tracker.cfg_w_pos = hud.slider_w_pos.val
        
        # Synchronize Sliders with Simulator World
        sim_world.set_car_count(hud.slider_traffic.val)
        sim_world.contrast_slider = hud.slider_contrast.val
        sim_world.time_of_day = hud.slider_time.val
        sim_world.weather_slider = hud.slider_weather.val
        
        # -------------------------------------------------------------
        # 2. Simulator Physics & Rendering
        # -------------------------------------------------------------
        if not is_paused:
            sim_time += dt
            sim_world.update(dt)
            
        sim_world.render_cctv_views()
        
        # Initial lock on startup
        if not has_initial_lock and sim_time > 0.3:
            auto_lock_next_cam01_car(ai_tracker, sim_world)
            has_initial_lock = True
            
        # -------------------------------------------------------------
        # 3. Pixel Extraction into BGR NumPy arrays
        # -------------------------------------------------------------
        for i in range(4):
            if sim_world.cam_enabled[i] and (ai_tracker.active_cam is None or i == ai_tracker.active_cam or i == ai_tracker.expecting_cam or ai_tracker.tracker_state == "AWAITING RE-ID"):
                raw_rgb = pygame.surfarray.array3d(sim_world.cam_surfs[i]).transpose(1, 0, 2)
                cam_frames_bgr[i] = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
                
        # -------------------------------------------------------------
        # 4. AI Tracker World Execution
        # -------------------------------------------------------------
        if not is_paused:
            ai_tracker.process_cameras(
                cam_frames_bgr, sim_time,
                contrast_val=hud.slider_contrast.val,
                weather_val=hud.slider_weather.val,
                time_val=hud.slider_time.val,
                sim_world=sim_world
            )
            
        # -------------------------------------------------------------
        # 5. UI Render & Flip
        # -------------------------------------------------------------
        active_screen = pygame.display.get_surface()
        if active_screen is not None:
            hud.draw(active_screen, sim_world, ai_tracker, clock.get_fps(), is_paused=is_paused)
            pygame.display.flip()
        
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
