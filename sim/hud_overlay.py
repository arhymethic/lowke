"""
hud_overlay.py - Professional Industrial / Law Enforcement CCTV Surveillance Workstation.
Aesthetic: Authentic late 90s / early 2000s professional workstation software (Windows 2000 / Pelco / Milestone CCTV).
Fully adaptive: Dynamically calculates layout on any window resolution (1280x720, 1600x900, 1920x1080).
"""

import time
import math
import pygame
import cv2
import numpy as np


def draw_3d_bevel(surface, rect, raised=True, fill_color=None):
    """Draw classic Win32 / industrial 3D beveled rectangle."""
    x, y, w, h = rect
    c_hi = (75, 82, 92) if raised else (20, 24, 28)
    c_sh = (18, 22, 26) if raised else (68, 76, 86)
    
    if fill_color:
        pygame.draw.rect(surface, fill_color, (x, y, w, h))
        
    # Top and Left edges (light if raised, dark if sunken)
    pygame.draw.line(surface, c_hi, (x, y), (x + w - 1, y), 1)
    pygame.draw.line(surface, c_hi, (x, y), (x, y + h - 1), 1)
    
    # Bottom and Right edges (dark if raised, light if sunken)
    pygame.draw.line(surface, c_sh, (x, y + h - 1), (x + w - 1, y + h - 1), 1)
    pygame.draw.line(surface, c_sh, (x + w - 1, y), (x + w - 1, y + h - 1), 1)


class RetroSlider:
    """Tactical industrial slider with groove, tick marks, and thumb handle."""
    def __init__(self, rect, min_val, max_val, init_val, label, unit="", is_int=False):
        self.rect = pygame.Rect(rect)
        self.min_val = min_val
        self.max_val = max_val
        self.val = init_val
        self.label = label
        self.unit = unit
        self.is_int = is_int
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self._update_val(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self._update_val(event.pos[0])
                return True
        return False

    def _update_val(self, mouse_x):
        rel = (mouse_x - (self.rect.x + 8)) / float(max(1, self.rect.width - 16))
        rel = max(0.0, min(1.0, rel))
        raw = self.min_val + rel * (self.max_val - self.min_val)
        self.val = int(round(raw)) if self.is_int else float(raw)

    def draw(self, surface, font):
        x, y, w, h = self.rect
        
        # Label & numeric readout
        val_str = f"{int(self.val)}{self.unit}" if self.is_int else f"{int(self.val * 100)}{self.unit}"
        txt_l = font.render(self.label, True, (195, 205, 215))
        txt_v = font.render(val_str, True, (240, 220, 120))
        surface.blit(txt_l, (x, y))
        surface.blit(txt_v, (x + w - txt_v.get_width(), y))
        
        # Sunken track groove
        groove_y = y + 17
        pygame.draw.rect(surface, (18, 22, 26), (x, groove_y, w, 7))
        draw_3d_bevel(surface, (x, groove_y, w, 7), raised=False)
        
        # Fill proportion
        frac = (self.val - self.min_val) / float(self.max_val - self.min_val)
        fill_w = int((w - 16) * frac)
        if fill_w > 0:
            pygame.draw.rect(surface, (45, 130, 110), (x + 2, groove_y + 1, fill_w, 5))
            
        # Tick marks
        for i in range(5):
            tx = x + int(i * (w / 4.0))
            pygame.draw.line(surface, (65, 75, 85), (tx, groove_y + 9), (tx, groove_y + 13), 1)
            
        # 3D Raised Slider Thumb
        thumb_x = x + 8 + fill_w
        thumb_rect = (thumb_x - 7, groove_y - 4, 14, 16)
        draw_3d_bevel(surface, thumb_rect, raised=True, fill_color=(68, 76, 86))
        pygame.draw.line(surface, (195, 205, 215), (thumb_x, groove_y - 1), (thumb_x, groove_y + 9), 1)


class RetroButton:
    """Tactical 3D beveled button with authentic pressed state and optional accent color."""
    def __init__(self, rect, label, tag="", accent_color=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.tag = tag
        self.pressed = False
        self.accent_color = accent_color

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_pressed = self.pressed
            self.pressed = False
            if was_pressed and self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, surface, font, is_active=False):
        x, y, w, h = self.rect
        if self.pressed or is_active:
            bg = (34, 42, 50) if not self.accent_color else (int(self.accent_color[0] * 0.35), int(self.accent_color[1] * 0.35), int(self.accent_color[2] * 0.35))
        else:
            bg = (48, 56, 64) if not self.accent_color else (int(self.accent_color[0] * 0.20), int(self.accent_color[1] * 0.20), int(self.accent_color[2] * 0.20))
            
        draw_3d_bevel(surface, (x, y, w, h), raised=not (self.pressed or is_active), fill_color=bg)
        
        if self.accent_color:
            txt_col = (255, 255, 255) if (self.pressed or is_active) else self.accent_color
        else:
            txt_col = (255, 255, 255) if (self.pressed or is_active) else (215, 225, 235)
            
        txt = font.render(self.label, True, txt_col)
        tx = x + (w - txt.get_width()) // 2 + (1 if self.pressed else 0)
        ty = y + (h - txt.get_height()) // 2 + (1 if self.pressed else 0)
        surface.blit(txt, (tx, ty))


class HUDOverlay:
    """
    Adaptive Professional Surveillance Workstation UI.
    Dynamically recalculates coordinates for any window width & height.
    """
    def __init__(self, init_width=1280, init_height=720):
        pygame.font.init()
        # High-legibility enlarged system fonts
        self.font_title = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_body = pygame.font.SysFont("monospace", 12)
        self.font_small = pygame.font.SysFont("monospace", 11, bold=True)
        self.font_tiny = pygame.font.SysFont("monospace", 10, bold=True)
        self.font_osd = pygame.font.SysFont("monospace", 11, bold=True)
        
        self.win_w = init_width
        self.win_h = init_height
        
        # Surface caching to eliminate redundant frame conversions
        self._cached_ref_img = None
        self._cached_ref_surf = None
        
        # 1. AI Model Configuration Sliders (6 Tunable Hyperparameters)
        self.slider_reid_thresh = RetroSlider((0, 0, 100, 26), 0.30, 0.95, 0.65, "RE-ID MATCH THRESHOLD", "%")
        self.slider_w_visual = RetroSlider((0, 0, 100, 26), 0.0, 1.0, 0.10, "PHOTO TEMPLATE WEIGHT", "%")
        self.slider_w_color = RetroSlider((0, 0, 100, 26), 0.0, 1.0, 0.35, "CHASSIS PAINT WEIGHT", "%")
        self.slider_w_driver = RetroSlider((0, 0, 100, 26), 0.0, 1.0, 0.25, "DRIVER AVATAR WEIGHT", "%")
        self.slider_w_speed = RetroSlider((0, 0, 100, 26), 0.0, 1.0, 0.20, "RADAR SPEED WEIGHT", "%")
        self.slider_w_pos = RetroSlider((0, 0, 100, 26), 0.0, 1.0, 0.20, "CORRIDOR TIMING WEIGHT", "%")
        self.ai_sliders = [
            self.slider_reid_thresh, self.slider_w_visual, self.slider_w_color,
            self.slider_w_driver, self.slider_w_speed, self.slider_w_pos
        ]
        
        # 2. Simulation Environment Sliders (4 Variables)
        self.slider_traffic = RetroSlider((0, 0, 100, 26), 10, 80, 35, "TRAFFIC DENSITY", " CARS", is_int=True)
        self.slider_contrast = RetroSlider((0, 0, 100, 26), 0.0, 1.0, 0.0, "COLOR SIMILARITY / GREY", "%")
        self.slider_time = RetroSlider((0, 0, 100, 26), 0.0, 1.0, 0.0, "DAY / NIGHT LIGHTING", "%")
        self.slider_weather = RetroSlider((0, 0, 100, 26), 0.0, 1.0, 0.0, "DUST / RAIN NOISE", "%")
        self.env_sliders = [self.slider_traffic, self.slider_contrast, self.slider_time, self.slider_weather]
        
        # Active Tab for Right Panel Sliders: "ai" or "env"
        self.active_tab = "ai"
        self.btn_tab_ai = RetroButton((0, 0, 100, 26), "[ 🎛 AI MODEL CONFIG (6) ]", "tab_ai")
        self.btn_tab_env = RetroButton((0, 0, 100, 26), "[ 🌦 SIM ENVIRONMENT (4) ]", "tab_env")
        
        # Human-in-the-Loop Feedback Buttons (Right Panel Inspector)
        self.btn_panel_right = RetroButton((0, 0, 95, 28), "[ ✓ RIGHT CAR (R) ]", "right_car", accent_color=(50, 255, 140))
        self.btn_panel_wrong = RetroButton((0, 0, 95, 28), "[ ✗ WRONG CAR (W) ]", "wrong_car", accent_color=(255, 80, 80))
        self.btn_panel_unlock = RetroButton((0, 0, 90, 28), "[ ⟲ DISENGAGE (U) ]", "unlock")
        self.panel_buttons = [self.btn_panel_right, self.btn_panel_wrong, self.btn_panel_unlock]
        
        # Bottom Corridor Interactive Buttons (5 buttons)
        self.btn_play = RetroButton((0, 0, 100, 34), "[ ▶ PLAY/PAUSE ]", "pause")
        self.btn_lock = RetroButton((0, 0, 100, 34), "[ 🎯 AUTO-LOCK NEXT ]", "lock_next")
        self.btn_right = RetroButton((0, 0, 100, 34), "[ ✓ RIGHT CAR (R) ]", "right_car", accent_color=(50, 255, 140))
        self.btn_wrong = RetroButton((0, 0, 100, 34), "[ ✗ WRONG CAR (W) ]", "wrong_car", accent_color=(255, 80, 80))
        self.btn_reset = RetroButton((0, 0, 100, 34), "[ ⟲ DISENGAGE (U) ]", "unlock")
        self.buttons = [self.btn_play, self.btn_lock, self.btn_right, self.btn_wrong, self.btn_reset]
        
        self.cctv_slots = []
        self.update_layout(init_width, init_height)

    def update_layout(self, width, height):
        """
        Dynamically adapts UI rects to current window resolution.
        """
        self.win_w = int(width)
        self.win_h = int(height)
        
        top_bar_h = 24
        status_bar_h = 22
        avail_h = max(200, self.win_h - top_bar_h - status_bar_h)
        
        # Left section (64% width): Cameras & Corridor
        left_w = int(self.win_w * 0.64)
        right_w = self.win_w - left_w
        right_x = left_w
        
        # Corridor bar height at bottom of left area
        corr_h = 114
        btn_bar_h = 34
        cam_area_h = avail_h - corr_h - btn_bar_h - 10
        
        # 2x2 Camera slots
        slot_w = (left_w - 18) // 2
        slot_h = (cam_area_h - 16) // 2
        
        # Calculate scaled 320x240 video rect inside each slot
        cam_scale = min((slot_w - 12) / 320.0, (slot_h - 26) / 240.0)
        c_w = int(320 * cam_scale)
        c_h = int(240 * cam_scale)
        
        names = [
            "CAM 01: WEST ENTRY (0m-250m) [CH-01]",
            "CAM 02: HIGHWAY SECTOR (500m-750m) [CH-02]",
            "CAM 03: OVERPASS JCT (1000m-1250m) [CH-03]",
            "CAM 04: EAST EXIT (1500m-1750m) [CH-04]"
        ]
        
        self.cctv_slots = []
        for i in range(4):
            col = i % 2
            row = i // 2
            sx = 6 + col * (slot_w + 6)
            sy = top_bar_h + 4 + row * (slot_h + 6)
            s_rect = pygame.Rect(sx, sy, slot_w, slot_h)
            
            # Center 320x240 (scaled) viewport inside slot
            vx = sx + (slot_w - c_w) // 2
            vy = sy + 20 + (slot_h - 24 - c_h) // 2
            c_rect = pygame.Rect(vx, vy, c_w, c_h)
            
            # Power toggle button in the header
            pwr_w, pwr_h = 56, 16
            pwr_rect = pygame.Rect(sx + slot_w - pwr_w - 4, sy + 3, pwr_w, pwr_h)
            
            self.cctv_slots.append({
                "id": i, "name": names[i],
                "rect": s_rect, "cam_rect": c_rect,
                "pwr_rect": pwr_rect,
                "scale": cam_scale
            })
            
        # Corridor Bar rect
        corr_y = top_bar_h + cam_area_h + 4
        self.corr_rect = pygame.Rect(6, corr_y, left_w - 12, corr_h)
        
        # Bottom Button Bar rects (5 buttons)
        btn_y = corr_y + corr_h + 4
        btn_w = (left_w - 32) // 5
        for i, b in enumerate(self.buttons):
            b.rect = pygame.Rect(6 + i * (btn_w + 5), btn_y, btn_w, btn_bar_h)
            
        # Right Panel layout
        self.right_rect = pygame.Rect(right_x, top_bar_h, right_w, avail_h)
        rx = right_x + 8
        rw = right_w - 16
        
        # Dual Photo frames: enlarged to 108x108 each
        box_dim = 108
        mug_y = top_bar_h + 46
        
        # Panel Feedback Buttons: below dual photo frames
        p_btn_w = (rw - 12) // 3
        p_btn_y = mug_y + box_dim + 10
        p_btn_h = 28
        self.btn_panel_right.rect = pygame.Rect(rx + 2, p_btn_y, p_btn_w, p_btn_h)
        self.btn_panel_wrong.rect = pygame.Rect(rx + 6 + p_btn_w, p_btn_y, p_btn_w, p_btn_h)
        self.btn_panel_unlock.rect = pygame.Rect(rx + 10 + 2 * p_btn_w, p_btn_y, p_btn_w, p_btn_h)
        
        # Observation box: below feedback buttons
        obs_box_y = p_btn_y + p_btn_h + 8
        obs_box_h = 108
        
        # Tab Buttons: below observation box
        tab_y = obs_box_y + obs_box_h + 8
        tab_w = (rw - 8) // 2
        tab_h = 26
        self.btn_tab_ai.rect = pygame.Rect(rx + 2, tab_y, tab_w, tab_h)
        self.btn_tab_env.rect = pygame.Rect(rx + 6 + tab_w, tab_y, tab_w, tab_h)
        
        # Sliders layout: below tab buttons
        slider_w = rw - 12
        sl_y_start = tab_y + tab_h + 6
        for i, s in enumerate(self.ai_sliders):
            s.rect = pygame.Rect(rx + 6, sl_y_start + i * 34, slider_w, 28)
        for i, s in enumerate(self.env_sliders):
            s.rect = pygame.Rect(rx + 6, sl_y_start + i * 38, slider_w, 32)

    def handle_event(self, event):
        if event.type in (pygame.VIDEORESIZE, getattr(pygame, "WINDOWRESIZED", 32777), getattr(pygame, "WINDOWSIZECHANGED", 32778)):
            w = getattr(event, "w", None) or getattr(event, "x", self.win_w)
            h = getattr(event, "h", None) or getattr(event, "y", self.win_h)
            self.update_layout(w, h)
            return None
            
        # Tab button clicks
        if self.btn_tab_ai.handle_event(event):
            self.active_tab = "ai"
            return "tab_ai"
        if self.btn_tab_env.handle_event(event):
            self.active_tab = "env"
            return "tab_env"
            
        # Sliders clicks
        active_sliders = self.ai_sliders if self.active_tab == "ai" else self.env_sliders
        for s in active_sliders:
            if s.handle_event(event):
                return "slider"
                
        # Panel feedback buttons
        for b in self.panel_buttons:
            if b.handle_event(event):
                return b.tag
                
        # Camera power toggle button clicks
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for slot in self.cctv_slots:
                pr = slot.get("pwr_rect")
                if pr and pr.collidepoint(event.pos):
                    return f"toggle_cam_{slot['id']}"

        # Bottom corridor buttons
        for b in self.buttons:
            if b.handle_event(event):
                return b.tag
                
        return None

    def get_cctv_click(self, mouse_pos):
        mx, my = mouse_pos
        for slot in self.cctv_slots:
            cr = slot["cam_rect"]
            if cr.collidepoint(mx, my):
                scale = slot.get("scale", 1.0)
                lx = int((mx - cr.x) / scale)
                ly = int((my - cr.y) / scale)
                lx = max(0, min(319, lx))
                ly = max(0, min(239, ly))
                return slot["id"], lx, ly
        return None

    def draw(self, screen, sim_world, ai_tracker, fps, is_paused=False):
        w, h = screen.get_size()
        if w != self.win_w or h != self.win_h:
            self.update_layout(w, h)
            
        # Workstation base background
        screen.fill((36, 42, 48))
        blink = (int(time.time() * 2) % 2) == 0
        t_now = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # -------------------------------------------------------------
        # 1. TOP MENU / TITLE BAR (Classic 90s/2000s Workstation Style)
        # -------------------------------------------------------------
        top_bar = pygame.Rect(0, 0, w, 24)
        draw_3d_bevel(screen, top_bar, raised=True, fill_color=(46, 54, 62))
        
        # Application title
        txt_app = self.font_title.render("PELCO/TRAFFIC-PRO v4.8 - HIGHWAY CORRIDOR SURVEILLANCE & RE-ID", True, (240, 245, 250))
        screen.blit(txt_app, (10, 4))
        
        # Classic menu bar items
        menu_items = ["File", "Cameras", "AI Profiler", "Diagnostics", "Help"]
        mx = 580
        for item in menu_items:
            t = self.font_small.render(item, True, (190, 200, 210))
            screen.blit(t, (mx, 6))
            mx += t.get_width() + 18
            
        # -------------------------------------------------------------
        # 2. 4 CCTV MONITORS (Recessed Bezels + OSD Green Monospace Text)
        # -------------------------------------------------------------
        for slot in self.cctv_slots:
            cid = slot["id"]
            sr = slot["rect"]
            cr = slot["cam_rect"]
            scale = slot["scale"]
            
            # Recessed monitor slot bezel
            draw_3d_bevel(screen, sr, raised=False, fill_color=(24, 28, 32))
            
            # Camera Header Bar
            pygame.draw.rect(screen, (34, 40, 48), (sr.x + 2, sr.y + 2, sr.width - 4, 18))
            draw_3d_bevel(screen, (sr.x + 2, sr.y + 2, sr.width - 4, 18), raised=True)
            
            is_enabled = sim_world.cam_enabled[cid]
            pwr_rect = slot.get("pwr_rect", pygame.Rect(sr.x + sr.width - 60, sr.y + 3, 56, 16))
            
            # LED indicator dot
            if is_enabled:
                led_col = (255, 30, 30) if blink else (130, 20, 20)
            else:
                led_col = (65, 70, 75)
            pygame.draw.circle(screen, led_col, (sr.x + 10, sr.y + 11), 3)
            
            txt_cam = self.font_medium.render(slot["name"], True, (220, 230, 240) if is_enabled else (135, 145, 155))
            screen.blit(txt_cam, (sr.x + 18, sr.y + 4))
            
            # Interactive Power Toggle Button [PWR: ON] / [PWR: OFF]
            pwr_bg = (24, 62, 42) if is_enabled else (68, 26, 26)
            pwr_border = (52, 195, 105) if is_enabled else (195, 62, 62)
            pwr_txt_col = (185, 255, 215) if is_enabled else (255, 175, 175)
            pwr_label = f"[{cid+1}] ON" if is_enabled else f"[{cid+1}] OFF"
            
            draw_3d_bevel(screen, pwr_rect, raised=is_enabled, fill_color=pwr_bg)
            pygame.draw.rect(screen, pwr_border, pwr_rect, 1)
            t_pwr = self.font_tiny.render(pwr_label, True, pwr_txt_col)
            screen.blit(t_pwr, (pwr_rect.x + (pwr_rect.width - t_pwr.get_width()) // 2, pwr_rect.y + 2))
            
            # Status Badge (positioned left of power button)
            if not is_enabled:
                badge_str = "[FEED OFFLINE]"
                badge_col = (255, 95, 95)
            elif ai_tracker.active_cam == cid:
                if ai_tracker.tracker_state == ai_tracker.STATE_COASTING:
                    badge_str = "[★ BRIDGE OCCLUSION]"
                    badge_col = (245, 215, 30)
                else:
                    badge_str = "[★ LIVE TARGET LOCK]"
                    badge_col = (50, 255, 120)
            elif ai_tracker.expecting_cam == cid:
                badge_str = f"[EXPECTING ETA: {ai_tracker.transit_eta:.1f}s]"
                badge_col = (80, 210, 255)
            else:
                badge_str = "[STANDBY]"
                badge_col = (130, 145, 160)
                
            txt_badge = self.font_tiny.render(badge_str, True, badge_col)
            screen.blit(txt_badge, (pwr_rect.x - txt_badge.get_width() - 8, sr.y + 4))
            
            # Video Viewport (Sunken CRT/LCD screen)
            draw_3d_bevel(screen, (cr.x - 2, cr.y - 2, cr.width + 4, cr.height + 4), raised=False, fill_color=(0, 0, 0))
            
            # Blit camera feed (fast nearest-neighbor scale for authentic pixel-art and 60+ FPS)
            raw_cam = sim_world.cam_surfs[cid]
            if scale != 1.0 or cr.width != 320 or cr.height != 240:
                scaled_cam = pygame.transform.scale(raw_cam, (cr.width, cr.height))
                screen.blit(scaled_cam, (cr.x, cr.y))
            else:
                screen.blit(raw_cam, (cr.x, cr.y))
                
            # Classic OSD Timestamp inside video feed (CCTV green or offline red)
            if is_enabled:
                osd_txt = self.font_osd.render(f"LIVE {t_now} | 1080i/30fps", True, (80, 255, 120))
            else:
                osd_txt = self.font_osd.render("FEED DISABLED | PREDICTION ENGINE RUNNING", True, (255, 110, 110))
            screen.blit(osd_txt, (cr.x + 4, cr.y + cr.height - 15))
            
            # Reticle, Trajectory & Hierarchical Sub-Part Tracking in Active Cam
            if ai_tracker.active_cam == cid and ai_tracker.kalman is not None:
                parts = ai_tracker.kalman.get_part_boxes()
                kx, ky, kw, kh = parts["chassis"]
                kx_s = cr.x + int(max(0, min(320 - kw, kx)) * scale)
                ky_s = cr.y + int(max(0, min(240 - kh, ky)) * scale)
                kw_s = int(kw * scale)
                kh_s = int(kh * scale)
                
                # Forward Trajectory prediction line
                if len(ai_tracker.trajectory_points) >= 2:
                    p1 = (cr.x + int(ai_tracker.trajectory_points[0][0] * scale), cr.y + int(ai_tracker.trajectory_points[0][1] * scale))
                    p2 = (cr.x + int(ai_tracker.trajectory_points[1][0] * scale), cr.y + int(ai_tracker.trajectory_points[1][1] * scale))
                    curr = (kx_s + kw_s // 2, ky_s + kh_s // 2)
                    
                    p1_c = (max(cr.left, min(cr.right - 2, p1[0])), max(cr.top, min(cr.bottom - 2, p1[1])))
                    p2_c = (max(cr.left, min(cr.right - 2, p2[0])), max(cr.top, min(cr.bottom - 2, p2[1])))
                    if cr.collidepoint(curr):
                        pygame.draw.line(screen, (0, 255, 180), curr, p1_c, 2)
                        pygame.draw.line(screen, (0, 210, 255), p1_c, p2_c, 1)
                        pygame.draw.circle(screen, (0, 255, 180), p1_c, 3)
                        pygame.draw.circle(screen, (0, 210, 255), p2_c, 2)
                    
                if ai_tracker.tracker_state == ai_tracker.STATE_COASTING:
                    self._draw_dashed_rect(screen, (255, 215, 30), (kx_s, ky_s, kw_s, kh_s))
                    txt_coast = self.font_tiny.render("COASTING: KALMAN ESTIMATE", True, (255, 215, 30))
                    screen.blit(txt_coast, (kx_s - 10, ky_s - 13))
                else:
                    # 1. Main Rigid Chassis Reticle (Rock-solid, fixed dimension green brackets)
                    self._draw_brackets(screen, (30, 255, 130), (kx_s, ky_s, kw_s, kh_s))
                    tag = f"#{ai_tracker.target_id} [{int(ai_tracker.p_speed_kmh)} km/h]"
                    txt_tag = self.font_tiny.render(tag, True, (30, 255, 130))
                    screen.blit(txt_tag, (kx_s, ky_s - 12))
                    
                    # 2. Sub-Part: Windshield / Driver ROI Reticle (Cyan)
                    if "driver" in parts:
                        dx, dy, dw, dh = parts["driver"]
                        dx_s = cr.x + int(dx * scale)
                        dy_s = cr.y + int(dy * scale)
                        self._draw_sub_part_bracket(screen, (0, 225, 255), (dx_s, dy_s, int(dw * scale), int(dh * scale)), "DRIVER")
                    
                    # 3. Sub-Part: Front Bumper / Radar Reticle (White)
                    if "front" in parts:
                        fx, fy, fw, fh = parts["front"]
                        fx_s = cr.x + int(fx * scale)
                        fy_s = cr.y + int(fy * scale)
                        self._draw_sub_part_bracket(screen, (230, 240, 255), (fx_s, fy_s, int(fw * scale), int(fh * scale)), "RADAR")

        # -------------------------------------------------------------
        # 3. CORRIDOR POSITION PREDICTOR (0m to 2000m)
        # -------------------------------------------------------------
        draw_3d_bevel(screen, self.corr_rect, raised=False, fill_color=(24, 28, 34))
        
        # Corridor Title Bar
        pygame.draw.rect(screen, (36, 42, 50), (self.corr_rect.x + 2, self.corr_rect.y + 2, self.corr_rect.width - 4, 20))
        draw_3d_bevel(screen, (self.corr_rect.x + 2, self.corr_rect.y + 2, self.corr_rect.width - 4, 20), raised=True)
        txt_corr = self.font_medium.render("LINEAR HIGHWAY CORRIDOR KINEMATICS TRACKER (0m ➔ 2000m)", True, (220, 235, 245))
        screen.blit(txt_corr, (self.corr_rect.x + 10, self.corr_rect.y + 3))
        
        # Track line geometry
        tx_start = self.corr_rect.x + 36
        tx_end = self.corr_rect.x + self.corr_rect.width - 36
        track_w = max(200, tx_end - tx_start)
        
        # 1. UPPER LANE: Dynamic Vehicle Pin & ETA Badge (NEVER overlaps camera tags below)
        pin_y = self.corr_rect.y + 44
        if ai_tracker.tracker_state != ai_tracker.STATE_IDLE:
            target_x_m = ai_tracker.sim_corridor_x % 2000.0
            px_target = tx_start + int((target_x_m / 2000.0) * track_w)
            px_target = max(tx_start, min(tx_end, px_target))
            
            if ai_tracker.tracker_state == ai_tracker.STATE_TRANSIT:
                col = (255, 195, 45)
                nxt = f"CAM 0{ai_tracker.expecting_cam + 1}" if ai_tracker.expecting_cam is not None else "DEAD RECKONING"
                t_str = f"▼ #{ai_tracker.target_id} [TRANSIT ➔ {nxt} | ETA: {ai_tracker.transit_eta:.1f}s]"
            else:
                col = (40, 255, 130) if ai_tracker.active_cam is not None else (255, 215, 60)
                loc_txt = f"CAM 0{ai_tracker.active_cam + 1}" if ai_tracker.active_cam is not None else "DEAD RECKONING (DARK SECTOR)"
                t_str = f"▼ #{ai_tracker.target_id} [{loc_txt}]"
                
            t_lbl = self.font_small.render(t_str, True, col)
            lbl_w = t_lbl.get_width()
            lbl_x = max(tx_start, min(tx_end - lbl_w, px_target - lbl_w // 2))
            screen.blit(t_lbl, (lbl_x, pin_y - 18))
            
            # Pointer Arrow head
            pygame.draw.polygon(screen, col, [
                (px_target, pin_y + 8), (px_target - 5, pin_y - 2), (px_target + 5, pin_y - 2)
            ])
            
        # 2. MIDDLE LANE: High-Contrast Highway Track with Camera Sectors & Blind Spots
        track_y = self.corr_rect.y + 62
        # Background roadbed
        pygame.draw.rect(screen, (16, 20, 24), (tx_start, track_y - 4, track_w, 10))
        draw_3d_bevel(screen, (tx_start, track_y - 4, track_w, 10), raised=False)
        
        # 4 Camera Monitored Segments (Glowing teal-emerald when online, dark red when offline)
        cam_spans = [(0, 250, "CAM 01", 0), (500, 750, "CAM 02", 1), (1000, 1250, "CAM 03", 2), (1500, 1750, "CAM 04", 3)]
        for m_start, m_end, name, c_idx in cam_spans:
            px_s = tx_start + int((m_start / 2000.0) * track_w)
            px_e = tx_start + int((m_end / 2000.0) * track_w)
            is_en = sim_world.cam_enabled[c_idx]
            if is_en:
                pygame.draw.rect(screen, (40, 150, 130), (px_s, track_y - 3, px_e - px_s, 8))
                pygame.draw.rect(screen, (80, 240, 200), (px_s, track_y - 1, px_e - px_s, 2))
            else:
                pygame.draw.rect(screen, (60, 28, 28), (px_s, track_y - 3, px_e - px_s, 8))
                pygame.draw.rect(screen, (160, 60, 60), (px_s, track_y - 1, px_e - px_s, 2))
            
        # 4 Blind-Spot Gap Segments (Subtle amber-slate roadbed)
        gap_spans = [(250, 500, "GAP 1"), (750, 1000, "GAP 2"), (1250, 1500, "GAP 3"), (1750, 2000, "GAP 4 (LOOP)")]
        for gs, ge, gname in gap_spans:
            px_s = tx_start + int((gs / 2000.0) * track_w)
            px_e = tx_start + int((ge / 2000.0) * track_w)
            pygame.draw.rect(screen, (28, 34, 42), (px_s, track_y - 3, px_e - px_s, 8))
            # Dashed centerline through blind spot gap
            for dash_x in range(px_s + 4, px_e - 4, 10):
                pygame.draw.line(screen, (60, 72, 85), (dash_x, track_y + 1), (dash_x + 5, track_y + 1), 1)
                
        # Distance Tick Marks along track: 0m, 500m, 1000m, 1500m, 2000m
        for m_val in [0, 500, 1000, 1500, 2000]:
            mx_pos = tx_start + int((m_val / 2000.0) * track_w)
            pygame.draw.line(screen, (100, 115, 130), (mx_pos, track_y + 6), (mx_pos, track_y + 10), 1)
            t_m = self.font_tiny.render(f"{m_val}m", True, (130, 145, 160))
            screen.blit(t_m, (mx_pos - t_m.get_width() // 2, track_y + 11))
            
        # 3. LOWER LANE: Camera Sector Badges
        tag_y = self.corr_rect.y + 88
        for m_start, m_end, name, c_idx in cam_spans:
            px_s = tx_start + int((m_start / 2000.0) * track_w)
            is_en = sim_world.cam_enabled[c_idx]
            badge_txt = f"[{name}]" if is_en else f"[{name}: OFF]"
            badge_col = (80, 240, 200) if is_en else (220, 90, 90)
            lbl = self.font_tiny.render(badge_txt, True, badge_col)
            screen.blit(lbl, (px_s + 2, tag_y))
            
        for gs, ge, gname in gap_spans:
            px_s = tx_start + int((gs / 2000.0) * track_w)
            lbl = self.font_tiny.render(f"-{gname}-", True, (140, 150, 160))
            screen.blit(lbl, (px_s + 4, tag_y))
                
        # Draw Bottom Buttons
        for b in self.buttons:
            b.draw(screen, self.font_small)

        # -------------------------------------------------------------
        # 4. RIGHT PANEL: TARGET VISUAL INSPECTOR & FORENSIC DOSSIER
        # -------------------------------------------------------------
        draw_3d_bevel(screen, self.right_rect, raised=True, fill_color=(42, 48, 56))
        
        rx = self.right_rect.x + 8
        rw = self.right_rect.width - 16
        ry = self.right_rect.y + 4
        
        # 4A. Header banner
        pygame.draw.rect(screen, (30, 36, 44), (rx, ry, rw, 22))
        draw_3d_bevel(screen, (rx, ry, rw, 22), raised=False)
        txt_dos = self.font_title.render("TARGET VISUAL INSPECTOR & RE-ID DOSSIER", True, (240, 245, 250))
        screen.blit(txt_dos, (rx + 8, ry + 3))
        ry += 26
        
        # 4B. Side-by-Side Dual Photo Inspector: Enlarged 108x108 Frames (100x100 Images)
        box_dim = 108
        img_dim = 100
        pad = 4
        
        screen.blit(self.font_tiny.render("[ 📷 INITIAL REF PHOTO ]", True, (80, 230, 200)), (rx + 4, ry))
        screen.blit(self.font_tiny.render("[ 🔴 LIVE CAMERA CROP ]", True, (255, 120, 120)), (rx + box_dim + 12, ry))
        
        mug_y = ry + 16
        # Photo 1: Initial Reference Photo (Enlarged)
        mug_rect1 = (rx + 4, mug_y, box_dim, box_dim)
        draw_3d_bevel(screen, mug_rect1, raised=False, fill_color=(18, 22, 26))
        
        ref_photo = ai_tracker.initial_photo if ai_tracker.initial_photo is not None else ai_tracker.target_crop
        if ref_photo is not None and ref_photo.size > 0:
            if self._cached_ref_img is None or self._cached_ref_img is not ref_photo:
                self._cached_ref_img = ref_photo
                rgb_ref = cv2.cvtColor(ref_photo, cv2.COLOR_BGR2RGB)
                ref_s = pygame.surfarray.make_surface(rgb_ref.transpose(1, 0, 2))
                self._cached_ref_surf = pygame.transform.scale(ref_s, (img_dim, img_dim))
            screen.blit(self._cached_ref_surf, (rx + 4 + pad, mug_y + pad))
            # Reticle corner accents
            pygame.draw.line(screen, (40, 220, 140), (rx + 4, mug_y + box_dim // 2), (rx + 16, mug_y + box_dim // 2), 2)
            pygame.draw.line(screen, (40, 220, 140), (rx + box_dim - 8, mug_y + box_dim // 2), (rx + box_dim + 4, mug_y + box_dim // 2), 2)
        else:
            screen.blit(self.font_small.render("NO REF", True, (110, 120, 130)), (rx + 32, mug_y + 36))
            screen.blit(self.font_tiny.render("CLICK CAR", True, (80, 180, 220)), (rx + 22, mug_y + 56))
            
        # Photo 2: Realtime Live Tracked Crop (Enlarged)
        mug_rect2 = (rx + box_dim + 12, mug_y, box_dim, box_dim)
        draw_3d_bevel(screen, mug_rect2, raised=False, fill_color=(18, 22, 26))
        
        live_img = ai_tracker.live_crop if ai_tracker.live_crop is not None else ai_tracker.target_crop
        if live_img is not None and live_img.size > 0 and ai_tracker.tracker_state == ai_tracker.STATE_LOCKED:
            rgb_live = cv2.cvtColor(live_img, cv2.COLOR_BGR2RGB)
            live_s = pygame.surfarray.make_surface(rgb_live.transpose(1, 0, 2))
            live_s = pygame.transform.scale(live_s, (img_dim, img_dim))
            screen.blit(live_s, (rx + box_dim + 12 + pad, mug_y + pad))
            # Live tracking crosshair
            cy_mid = mug_y + box_dim // 2
            pygame.draw.line(screen, (0, 225, 255), (rx + box_dim + 12, cy_mid), (rx + box_dim + 24, cy_mid), 2)
            pygame.draw.line(screen, (0, 225, 255), (rx + 2 * box_dim, cy_mid), (rx + 2 * box_dim + 12, cy_mid), 2)
        else:
            st_text = "STANDBY" if ai_tracker.target_id is None else "IN TRANSIT"
            screen.blit(self.font_small.render(st_text, True, (140, 150, 160)), (rx + box_dim + 30, mug_y + 42))
            
        # Telemetry Block (Placed to the right of the dual photos)
        dt_x = rx + 2 * box_dim + 22
        t_id_str = f"#{ai_tracker.target_id}" if ai_tracker.target_id else "NONE"
        screen.blit(self.font_medium.render(f"ASSIGNED ID: {t_id_str}", True, (245, 225, 100)), (dt_x, mug_y))
        
        st_col = (40, 255, 130) if ai_tracker.tracker_state == ai_tracker.STATE_LOCKED else (245, 190, 35) if ai_tracker.tracker_state in [ai_tracker.STATE_TRANSIT, ai_tracker.STATE_COASTING, "AWAITING RE-ID"] else (140, 150, 160)
        screen.blit(self.font_small.render(f"TRACK STATE: {ai_tracker.tracker_state}", True, st_col), (dt_x, mug_y + 20))
        
        val_col = (50, 255, 140) if "CONFIRMED" in ai_tracker.validation_status else (255, 80, 80) if "FLAGGED" in ai_tracker.validation_status else (245, 215, 60)
        screen.blit(self.font_small.render(f"VALIDATION : {ai_tracker.validation_status}", True, val_col), (dt_x, mug_y + 38))
        
        cam_str = f"CAM 0{ai_tracker.active_cam + 1}" if ai_tracker.active_cam is not None else f"GAP ➔ CAM 0{ai_tracker.expecting_cam + 1}" if ai_tracker.expecting_cam is not None else "STANDBY"
        screen.blit(self.font_small.render(f"LOCATION   : {cam_str}", True, (200, 215, 225)), (dt_x, mug_y + 56))
        
        # Real-time Visual Template Match %
        vis_pct = int(ai_tracker.visual_similarity * 100) if ai_tracker.target_id else 0
        vis_col = (40, 255, 130) if vis_pct >= 70 else (245, 190, 35) if vis_pct >= 50 else (255, 80, 80)
        screen.blit(self.font_small.render(f"PHOTO MATCH: {vis_pct}% SIMILARITY", True, vis_col), (dt_x, mug_y + 74))
        
        # Match Progress Bar (Thicker 8px)
        bar_w_total = max(80, rw - (2 * box_dim + 28))
        pygame.draw.rect(screen, (18, 22, 26), (dt_x, mug_y + 92, bar_w_total, 8))
        draw_3d_bevel(screen, (dt_x, mug_y + 92, bar_w_total, 8), raised=False)
        if vis_pct > 0:
            fill_w = int((bar_w_total - 2) * (vis_pct / 100.0))
            pygame.draw.rect(screen, vis_col, (dt_x + 1, mug_y + 93, fill_w, 6))
            
        ry = mug_y + box_dim + 10
        
        # 4C. Human-in-the-Loop Feedback Buttons (Right / Wrong Car Validation)
        self.btn_panel_right.draw(screen, self.font_small)
        self.btn_panel_wrong.draw(screen, self.font_small)
        self.btn_panel_unlock.draw(screen, self.font_small)
        ry += 36
        
        # -------------------------------------------------------------
        # 5. DEDICATED VEHICLE OBSERVATION LOG BOX (Engraved)
        # -------------------------------------------------------------
        box_h = 108
        draw_3d_bevel(screen, (rx, ry, rw, box_h), raised=False, fill_color=(28, 33, 40))
        pygame.draw.rect(screen, (38, 45, 55), (rx + 2, ry + 2, rw - 4, 18))
        draw_3d_bevel(screen, (rx + 2, ry + 2, rw - 4, 18), raised=True)
        screen.blit(self.font_small.render("OBSERVED VEHICLE ATTRIBUTES", True, (220, 235, 245)), (rx + 8, ry + 3))
        
        py = ry + 24
        screen.blit(self.font_body.render(f"• CHASSIS PAINT: {ai_tracker.p_color_name} ({int(ai_tracker.p_color_conf * 100)}%)", True, (215, 225, 235)), (rx + 8, py))
        screen.blit(self.font_body.render(f"• DRIVER AVATAR: {ai_tracker.p_driver_name}", True, (215, 225, 235)), (rx + rw // 2, py))
        py += 20
        
        screen.blit(self.font_body.render(f"• RADAR SPEED  : {ai_tracker.p_speed_kmh:.1f} km/h", True, (215, 225, 235)), (rx + 8, py))
        corr_info = f"CAM 0{ai_tracker.active_cam + 1}" if ai_tracker.active_cam is not None else "Transit Search"
        screen.blit(self.font_body.render(f"• HIGHWAY POS  : {ai_tracker.sim_corridor_x:.0f}m [{corr_info}]", True, (80, 240, 180)), (rx + rw // 2, py))
        py += 22
        
        # Part-based Tracking Status row
        has_tgt = ai_tracker.target_id is not None
        c_chassis = (30, 255, 130) if has_tgt else (100, 110, 120)
        c_driver = (0, 225, 255) if has_tgt else (100, 110, 120)
        c_radar = (230, 240, 255) if has_tgt else (100, 110, 120)
        screen.blit(self.font_small.render("PARTS: [■ CHASSIS: 42x28]", True, c_chassis), (rx + 8, py))
        screen.blit(self.font_small.render("[■ DRIVER]", True, c_driver), (rx + 175, py))
        screen.blit(self.font_small.render("[■ RADAR]", True, c_radar), (rx + 265, py))
        
        ry += box_h + 8
        
        # -------------------------------------------------------------
        # 6. CONFIGURATION TABS & SLIDERS (Tunable AI & Sim Variables)
        # -------------------------------------------------------------
        self.btn_tab_ai.draw(screen, self.font_small, is_active=(self.active_tab == "ai"))
        self.btn_tab_env.draw(screen, self.font_small, is_active=(self.active_tab == "env"))
        
        active_sliders = self.ai_sliders if self.active_tab == "ai" else self.env_sliders
        for s in active_sliders:
            s.draw(screen, self.font_small)
            
        # Incident & Sightings Audit Log Box
        log_y = active_sliders[-1].rect.y + 36
        log_h = (self.right_rect.y + self.right_rect.height) - log_y - 6
        if log_h > 40:
            draw_3d_bevel(screen, (rx, log_y, rw, log_h), raised=False, fill_color=(20, 24, 28))
            pygame.draw.rect(screen, (34, 40, 48), (rx + 2, log_y + 2, rw - 4, 18))
            screen.blit(self.font_small.render("CORRIDOR SIGHTINGS & ACTIVE RE-ID AUDIT LOG", True, (200, 215, 230)), (rx + 8, log_y + 3))
            
            ly = log_y + 22
            max_lines = max(2, (log_h - 26) // 16)
            for line in ai_tracker.event_logs[-max_lines:]:
                col = (80, 255, 140) if "CONFIRMED" in line or "RE-IDENTIFIED" in line else (255, 80, 80) if "WRONG CAR" in line or "FLAGGED" in line else (245, 195, 45) if "BLIND SPOT" in line or "EXITED" in line else (210, 220, 230)
                screen.blit(self.font_small.render(line, True, col), (rx + 8, ly))
                ly += 16
                
        # -------------------------------------------------------------
        # 7. STATUS BAR AT BOTTOM
        # -------------------------------------------------------------
        status_bar = pygame.Rect(0, h - 22, w, 22)
        draw_3d_bevel(screen, status_bar, raised=True, fill_color=(46, 54, 62))
        
        stat_txt = f"[SYSTEM READY]  PELCO CCTV WORKSTATION  |  VIEWPORT: {w}x{h} (ADAPTIVE)  |  CORRIDOR: 2000m  |  FPS: {fps:.1f}  |  STATUS: {'PAUSED' if is_paused else 'LIVE'}"
        screen.blit(self.font_small.render(stat_txt, True, (220, 230, 240)), (10, h - 17))

    def _draw_brackets(self, surface, color, rect):
        x, y, w, h = rect
        arm = min(8, min(w, h) // 3)
        pygame.draw.line(surface, color, (x, y), (x + arm, y), 2)
        pygame.draw.line(surface, color, (x, y), (x, y + arm), 2)
        pygame.draw.line(surface, color, (x + w, y), (x + w - arm, y), 2)
        pygame.draw.line(surface, color, (x + w, y), (x + w, y + arm), 2)
        pygame.draw.line(surface, color, (x, y + h), (x + arm, y + h), 2)
        pygame.draw.line(surface, color, (x, y + h), (x, y + h - arm), 2)
        pygame.draw.line(surface, color, (x + w, y + h), (x + w - arm, y + h), 2)
        pygame.draw.line(surface, color, (x + w, y + h), (x + w, y + h - arm), 2)

    def _draw_sub_part_bracket(self, surface, color, rect, label=""):
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            return
        arm = max(2, min(5, min(w, h) // 2))
        pygame.draw.line(surface, color, (x, y), (x + arm, y), 1)
        pygame.draw.line(surface, color, (x, y), (x, y + arm), 1)
        pygame.draw.line(surface, color, (x + w, y), (x + w - arm, y), 1)
        pygame.draw.line(surface, color, (x + w, y), (x + w, y + arm), 1)
        pygame.draw.line(surface, color, (x, y + h), (x + arm, y + h), 1)
        pygame.draw.line(surface, color, (x, y + h), (x, y + h - arm), 1)
        pygame.draw.line(surface, color, (x + w, y + h), (x + w - arm, y + h), 1)
        pygame.draw.line(surface, color, (x + w, y + h), (x + w, y + h - arm), 1)
        if label:
            txt = self.font_tiny.render(label, True, color)
            surface.blit(txt, (x + 2, y - 9))

    def _draw_dashed_rect(self, surface, color, rect, dash_len=4):
        x, y, w, h = rect
        for dx in range(0, w, dash_len * 2):
            pygame.draw.line(surface, color, (x + dx, y), (min(x + w, x + dx + dash_len), y), 1)
            pygame.draw.line(surface, color, (x + dx, y + h), (min(x + w, x + dx + dash_len), y + h), 1)
        for dy in range(0, h, dash_len * 2):
            pygame.draw.line(surface, color, (x, y + dy), (x, min(y + h, y + dy + dash_len)), 1)
            pygame.draw.line(surface, color, (x + w, y + dy), (x + w, min(y + h, y + dy + dash_len)), 1)
