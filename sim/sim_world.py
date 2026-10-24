"""
sim_world.py - Authentic Project Zomboid Isometric Graphics & Highway Simulator.

Visual Style:
- Exact Project Zomboid palette & aesthetics:
  - Textured moss/olive grass with procedural weed tufts, foliage clumps, and dirt speckles.
  - Light concrete sidewalk and 3D beveled stone curb running parallel to highway.
  - Dark asphalt highway with crisp yellow centerline and asphalt grain.
  - Authentic 90s boxy vehicles (Franklin / Chevalier pickups, vans, sedans, SUVs):
    - Detailed grilles, bumpers, square headlights, amber turn signals.
    - Black rubber tires with silver hubcaps under wheel wells.
    - Sloped windshield with visible driver avatar.
    - Emergency lightbars (red/amber beacons), racing stripes, roof racks, star logos.
    - Optional utility trailers with colored tarps attached via hitch!
"""

import math
import random
import pygame
import numpy as np

COS_30 = math.cos(math.radians(30))  # ~0.866025
SIN_30 = math.sin(math.radians(30))  # 0.5


def iso_project(x, y, z, offset_x, offset_y, scale=1.0):
    """Project 3D world coordinates into 2D isometric screen coordinates."""
    u = (x - y) * COS_30 * scale + offset_x
    v = (x + y) * SIN_30 * scale - z * scale + offset_y
    return int(u), int(v)


class LinearHighway:
    """Continuous straight highway corridor (0m to 2000m)."""
    def __init__(self, length=2000.0):
        self.total_length = length
        self.lane_offsets = [-15.0, 0.0, 15.0]

    def get_pose(self, x, lane_offset=0.0):
        x_clamped = x % self.total_length
        return (x_clamped, lane_offset), 0.0, (0.0, 1.0)


class SimCar:
    """
    Project Zomboid style 90s vehicle with rich isometric details:
    - Archetypes: Pickup (Emergency / Utility with trailer), Boxy Sedan, SUV, Van
    - Visible driver avatar inside windshield
    - Front grille, square headlights, turn signals, chrome bumper
    - Rubber tires with hubcaps
    - Decals: Emergency Lightbar, Dual Racing Stripe, Star/Logo, Roof Rack, Checkerboard
    """
    CAR_TYPES = [
        {"type": "Pickup",  "length": 32.0, "width": 15.0, "h_chassis": 7.0, "h_cabin": 6.0, "has_bed": True,  "prob": 0.35},
        {"type": "Sedan",   "length": 28.0, "width": 14.0, "h_chassis": 6.0, "h_cabin": 5.5, "has_bed": False, "prob": 0.35},
        {"type": "SUV",     "length": 31.0, "width": 16.0, "h_chassis": 8.0, "h_cabin": 6.8, "has_bed": False, "prob": 0.20},
        {"type": "Compact", "length": 23.0, "width": 13.0, "h_chassis": 5.5, "h_cabin": 5.0, "has_bed": False, "prob": 0.10},
    ]

    # Authentic 90s car colors & emergency livery
    VIBRANT_PALETTE = [
        (130, 28, 36),    # 0: Maroon / Burgundy (PZ Fire Dept / Ambulance)
        (38, 75, 145),    # 1: Deep Police / State Blue
        (45, 95, 55),     # 2: Forest / Hunter Green
        (225, 175, 25),   # 3: Highway Maintenance Amber / Yellow
        (185, 75, 25),    # 4: Rust / Copper Orange
        (95, 45, 125),    # 5: Plum / Dark Purple
        (40, 135, 140),   # 6: Teal / Classic 90s Cyan
        (230, 228, 225),  # 7: Off-White / Beige Fleet
        (32, 34, 38),     # 8: Asphalt Black
        (165, 40, 85),    # 9: Magenta / Rosewood
    ]

    DRIVER_SHIRTS = [
        (245, 215, 30),   # 1: Yellow Shirt
        (215, 38, 38),    # 2: Red Jacket
        (35, 190, 205),   # 3: Cyan Hoodie
        (240, 240, 245),  # 4: White T-Shirt
        (45, 180, 65),    # 5: Green Vest
        (235, 120, 25),   # 6: Orange Polo
        (155, 60, 215),   # 7: Purple Jacket
    ]

    _id_counter = 42

    def __init__(self, x, lane_index, speed):
        self.car_id = SimCar._id_counter
        SimCar._id_counter += 1
        self.id_str = f"VH-{self.car_id:03d}"
        
        self.x = x
        self.lane_index = lane_index
        self.lane_offset = (lane_index - 1) * 15.0  # -15, 0, +15
        self.speed = speed
        
        archetype = random.choices(self.CAR_TYPES, weights=[t["prob"] for t in self.CAR_TYPES])[0]
        self.type_name = archetype["type"]
        self.length = archetype["length"]
        self.width = archetype["width"]
        self.h_chassis = archetype["h_chassis"]
        self.h_cabin = archetype["h_cabin"]
        self.has_bed = archetype["has_bed"]
        
        # Chance of utility trailer behind pickup (like in the PZ screenshot!)
        self.has_trailer = (self.type_name == "Pickup") and (random.random() < 0.40)
        self.trailer_color = random.choice([(55, 95, 175), (140, 145, 150), (120, 75, 40)])
        
        # Color & Livery
        self.color_idx = random.randint(0, len(self.VIBRANT_PALETTE) - 1)
        self.base_color = self.VIBRANT_PALETTE[self.color_idx]
        self.two_tone_hood = (self.type_name == "Pickup") and (random.random() < 0.65)
        self.hood_color = (235, 235, 240) if self.two_tone_hood else self.base_color
        
        # Driver avatar
        self.driver_idx = random.randint(1, len(self.DRIVER_SHIRTS))
        self.driver_color = self.DRIVER_SHIRTS[self.driver_idx - 1]
        
        # Decals removed: Clean roof for robust computer vision tracking
        self.sticker_type = 0
            
        self.pos = (self.x, self.lane_offset)

    def update(self, dt, corridor_length=2000.0):
        self.x = (self.x + self.speed * dt) % corridor_length
        self.pos = (self.x, self.lane_offset)

    def project_isometric(self, cam_x):
        """Project vehicle 3D coordinate to 2D isometric screen space (u, v) for camera at cam_x."""
        COS_30 = 0.866025
        SIN_30 = 0.5
        off_x = 160.0 - cam_x * COS_30
        off_y = 120.0 - cam_x * SIN_30
        u = (self.x - self.lane_offset) * COS_30 + off_x
        v = (self.x + self.lane_offset) * SIN_30 - self.h_chassis * 0.5 + off_y
        return int(u), int(v)


class SimWorld:
    """
    Project Zomboid Isometric Highway Simulator.
    Renders 4 CCTV viewports (320x240) with authentic graphics:
    - Earthy textured grass, foliage tufts, and weeds.
    - Concrete curb and sidewalk with seams.
    - Asphalt road with crisp yellow centerlines.
    - Authentic 90s boxy vehicles with driver avatar and trailers.
    """
    CAM_NAMES = [
        "CAM 01: WEST ENTRY (0m-250m)",
        "CAM 02: HIGHWAY SECTOR (500m-750m)",
        "CAM 03: OVERPASS JCT (1000m-1250m)",
        "CAM 04: EAST EXIT (1500m-1750m)"
    ]

    def __init__(self):
        self.highway = LinearHighway(length=2000.0)
        self.cam_x_centers = [125.0, 625.0, 1125.0, 1625.0]
        self.cam_ranges = [(0.0, 250.0), (500.0, 750.0), (1000.0, 1250.0), (1500.0, 1750.0)]
        self.cam_gaps = [(250.0, 500.0), (750.0, 1000.0), (1250.0, 1500.0)]
        
        # 4 Pygame camera surfaces (320x240)
        self.cam_surfs = [pygame.Surface((320, 240)) for _ in range(4)]
        self.cam_enabled = [True, True, True, True]
        
        # Active traffic
        self.cars = []
        self.target_car_count = 35
        self._spawn_cars(self.target_car_count)
        
        # Sliders
        self.traffic_slider = 35
        self.contrast_slider = 0.0
        self.time_of_day = 0.0
        self.weather_slider = 0.0
        
        # Weather particles
        self.rain_drops = [
            [random.randint(0, 320), random.randint(0, 240), random.uniform(9.0, 18.0)]
            for _ in range(120)
        ]
        
        # Pre-allocate overlay surfaces to eliminate frame-rate drops from allocations
        self._dark_surf = pygame.Surface((320, 240))
        self._dark_surf.fill((0, 0, 0))
        self._rain_surf = pygame.Surface((320, 240), pygame.SRCALPHA)
        
        # Pre-generate Project Zomboid procedural terrain textures & foliage
        self._grass_bg_surf = pygame.Surface((320, 240))
        self._generate_pz_terrain_texture()
        
        # Pre-render static camera backgrounds (grass + road + curb + markings) to run at solid 60 FPS
        self._static_bgs = [pygame.Surface((320, 240)) for _ in range(4)]
        self._pre_render_static_backgrounds()

    def _pre_render_static_backgrounds(self):
        """Pre-renders static terrain, sidewalk, road, and curbs for each camera once."""
        for cam_idx in range(4):
            surf = self._static_bgs[cam_idx]
            cam_x = self.cam_x_centers[cam_idx]
            off_x = 160.0 - cam_x * COS_30
            off_y = 120.0 - cam_x * SIN_30
            surf.blit(self._grass_bg_surf, (0, 0))
            self._render_pz_road_and_curb(surf, off_x, off_y, cam_x)

    def _generate_pz_terrain_texture(self):
        """
        Creates authentic Project Zomboid textured grass:
        Mottled olive/moss tones with procedural grass blade tufts and weeds.
        """
        w, h = 320, 240
        np.random.seed(101)
        # Base olive ground tones: RGB (58, 74, 34) to (68, 86, 38)
        base = np.zeros((h, w, 3), dtype=np.uint8)
        r_chan = np.random.randint(52, 68, (h, w), dtype=np.uint8)
        g_chan = np.random.randint(68, 88, (h, w), dtype=np.uint8)
        b_chan = np.random.randint(30, 44, (h, w), dtype=np.uint8)
        base[:, :, 0] = r_chan
        base[:, :, 1] = g_chan
        base[:, :, 2] = b_chan
        
        surf = pygame.surfarray.make_surface(base.transpose(1, 0, 2))
        
        # Scatter tall grass clumps, weeds, and small foliage patches
        tuft_colors = [
            (95, 115, 52), (110, 132, 60), (78, 96, 42), (135, 120, 70), (45, 60, 28)
        ]
        for _ in range(280):
            gx = random.randint(0, w - 1)
            gy = random.randint(0, h - 1)
            col = random.choice(tuft_colors)
            # Draw tiny 2-3 blade grass clump
            for k in range(3):
                bx = gx + (k - 1) * 2
                pygame.draw.line(surf, col, (bx, gy), (bx + random.choice([-1, 0, 1]), gy - random.randint(3, 6)), 1)
                
        # Small foliage/shrub clusters
        for _ in range(35):
            sx = random.randint(0, w - 1)
            sy = random.randint(0, h - 1)
            radius = random.randint(3, 7)
            shrub_col = (42, 56, 25)
            pygame.draw.circle(surf, shrub_col, (sx, sy), radius)
            pygame.draw.circle(surf, (62, 80, 36), (sx - 1, sy - 1), max(1, radius - 2))
            
        self._grass_bg_surf = surf

    def _spawn_cars(self, count):
        self.cars.clear()
        step = self.highway.total_length / max(1, count)
        for i in range(count):
            x = (i * step + random.uniform(-10.0, 10.0)) % self.highway.total_length
            lane = random.choice([0, 1, 2])
            speed = random.uniform(50.0, 75.0)
            self.cars.append(SimCar(x, lane, speed))

    def set_car_count(self, count):
        count = int(np.clip(count, 5, 80))
        if count == len(self.cars):
            return
        if count > len(self.cars):
            for _ in range(count - len(self.cars)):
                x = random.uniform(0, self.highway.total_length)
                lane = random.choice([0, 1, 2])
                speed = random.uniform(48.0, 74.0)
                self.cars.append(SimCar(x, lane, speed))
        else:
            self.cars = self.cars[:count]

    def update(self, dt):
        for car in self.cars:
            car.update(dt, self.highway.total_length)
            
        if self.weather_slider > 0.05:
            rain_speed = 320.0 * (0.6 + 0.8 * self.weather_slider)
            for drop in self.rain_drops:
                drop[0] += 0.4 * rain_speed * dt
                drop[1] += rain_speed * dt
                if drop[0] > 320 or drop[1] > 240:
                    drop[0] = random.randint(-40, 280)
                    drop[1] = random.randint(-30, 0)

    def _get_car_effective_color(self, base_color):
        c = self.contrast_slider
        if c <= 0.01:
            return base_color
        grey = (118.0, 120.0, 124.0)
        return (
            int(base_color[0] * (1.0 - c) + grey[0] * c),
            int(base_color[1] * (1.0 - c) + grey[1] * c),
            int(base_color[2] * (1.0 - c) + grey[2] * c),
        )

    def _apply_lighting_to_color(self, color):
        if self.time_of_day <= 0.02:
            return color
        dark_factor = 1.0 - 0.75 * self.time_of_day
        return (
            int(color[0] * dark_factor),
            int(color[1] * dark_factor),
            int(color[2] * dark_factor),
        )

    def toggle_camera(self, cam_idx):
        if 0 <= cam_idx < 4:
            self.cam_enabled[cam_idx] = not self.cam_enabled[cam_idx]
            return self.cam_enabled[cam_idx]
        return False

    def set_camera_enabled(self, cam_idx, enabled):
        if 0 <= cam_idx < 4:
            self.cam_enabled[cam_idx] = bool(enabled)

    def _render_camera_offline(self, surf, cam_idx):
        """Renders authentic CCTV signal loss screen with analog raster scanlines."""
        surf.fill((16, 20, 24))
        
        # Subtle analog scanlines
        for y in range(0, 240, 4):
            pygame.draw.line(surf, (24, 30, 36), (0, y), (320, y), 1)
            
        # Vintage static speckles
        for _ in range(45):
            rx = random.randint(0, 319)
            ry = random.randint(0, 239)
            b = random.randint(40, 80)
            surf.set_at((rx, ry), (b, b, b))
            
        # Centered Warning Box
        box_w, box_h = 240, 94
        bx = (320 - box_w) // 2
        by = (240 - box_h) // 2
        pygame.draw.rect(surf, (24, 30, 38), (bx, by, box_w, box_h))
        pygame.draw.rect(surf, (180, 50, 50), (bx, by, box_w, box_h), 1)
        
        # Header banner
        pygame.draw.rect(surf, (180, 40, 40), (bx + 2, by + 2, box_w - 4, 20))
        font = pygame.font.Font(None, 18)
        font_sm = pygame.font.Font(None, 16)
        
        t_loss = font.render(f"NO SIGNAL ── CH-0{cam_idx + 1} OFFLINE", True, (255, 255, 255))
        surf.blit(t_loss, (bx + (box_w - t_loss.get_width()) // 2, by + 5))
        
        t_sub = font_sm.render("CAMERA SENSOR FEED DISCONNECTED", True, (220, 225, 230))
        surf.blit(t_sub, (bx + (box_w - t_sub.get_width()) // 2, by + 28))
        
        t_ai = font_sm.render("PREDICTION ENGINE: ONLINE (TRACKING)", True, (60, 240, 140))
        surf.blit(t_ai, (bx + (box_w - t_ai.get_width()) // 2, by + 48))
        
        t_hint = font_sm.render(f"Press Key [{cam_idx + 1}] or Click [ON/OFF] to restore", True, (160, 180, 200))
        surf.blit(t_hint, (bx + (box_w - t_hint.get_width()) // 2, by + 70))

    def render_cctv_views(self):
        """Render all 4 Project Zomboid style CCTV cameras at high FPS."""
        for cam_idx in range(4):
            surf = self.cam_surfs[cam_idx]
            if not self.cam_enabled[cam_idx]:
                self._render_camera_offline(surf, cam_idx)
                continue
                
            cam_x = self.cam_x_centers[cam_idx]
            
            off_x = 160.0 - cam_x * COS_30
            off_y = 120.0 - cam_x * SIN_30
            
            # 1. Blit pre-rendered static background (instant, 0% CPU overhead)
            surf.blit(self._static_bgs[cam_idx], (0, 0))
            if self.time_of_day > 0.02:
                self._dark_surf.set_alpha(int(190 * self.time_of_day))
                surf.blit(self._dark_surf, (0, 0))
                
            # 2. Visible vehicles sorted by isometric depth (x + y)
            visible = [c for c in self.cars if abs(c.x - cam_x) < 240.0]
            visible.sort(key=lambda c: (c.x + c.lane_offset))
            
            # Shadows and headlights first
            for car in visible:
                self._render_car_shadow(surf, car, off_x, off_y)
                
            # 3D Project Zomboid cars
            for car in visible:
                self._render_car_pz(surf, car, off_x, off_y)
                
            # Overpass Bridge on CAM 02 & Gantry on CAM 03
            if cam_idx == 1:
                self._render_overpass_bridge(surf, off_x, off_y, cam_x + 15.0)
            elif cam_idx == 2:
                self._render_highway_gantry(surf, off_x, off_y, cam_x - 10.0)
                
            # Weather Shaders (Rain / Dust)
            self._apply_weather_shaders(surf)

    def get_car_at_screen_pos(self, cam_idx, click_x, click_y):
        """
        Accurately find which car was clicked inside this camera viewport.
        Returns (car, screen_u, screen_v) or (None, click_x, click_y).
        """
        if not self.cam_enabled[cam_idx]:
            return None, click_x, click_y
            
        cam_x = self.cam_x_centers[cam_idx]
        off_x = 160.0 - cam_x * COS_30
        off_y = 120.0 - cam_x * SIN_30
        
        best_car = None
        min_dist = float("inf")
        best_u, best_v = click_x, click_y
        
        for car in self.cars:
            if abs(car.x - cam_x) < 240.0:
                u, v = iso_project(car.x, car.lane_offset, car.h_chassis * 0.5, off_x, off_y)
                dist = math.hypot(click_x - u, click_y - v)
                if dist < min_dist and dist < 45.0:
                    min_dist = dist
                    best_car = car
                    best_u, best_v = u, v
                    
        return best_car, best_u, best_v

    def _render_pz_road_and_curb(self, surf, off_x, off_y, cam_x):
        """
        Renders authentic Project Zomboid roadway extending fully across the screen:
        - Wide smooth asphalt road (#4c4e53).
        - Crisp solid yellow centerline (#d4ad39).
        - Light concrete sidewalk (#cec8bd) with transverse expansion seams.
        - 3D raised concrete curb (#e2ded4) with dark shadow lip (#383632).
        """
        asphalt = self._apply_lighting_to_color((74, 76, 82))
        yellow_line = self._apply_lighting_to_color((215, 175, 45))
        sidewalk_col = self._apply_lighting_to_color((202, 196, 185))
        curb_top = self._apply_lighting_to_color((225, 220, 210))
        curb_shadow = self._apply_lighting_to_color((52, 50, 46))
        seam_col = self._apply_lighting_to_color((165, 160, 150))
        
        # Extend span far past 320x240 screen diagonal so road goes fully across
        x_start = cam_x - 380.0
        x_end = cam_x + 380.0
        steps = 48
        dx = (x_end - x_start) / steps
        
        # Highway width = 50 world units (-25 to +25)
        # Upper sidewalk runs from y = -25 to y = -36
        pts_road_top = []
        pts_road_bot = []
        pts_center = []
        pts_sidewalk_top = []
        
        for i in range(steps + 1):
            cur_x = x_start + i * dx
            pts_road_top.append(iso_project(cur_x, -24.0, 0, off_x, off_y))
            pts_road_bot.append(iso_project(cur_x, 24.0, 0, off_x, off_y))
            pts_center.append(iso_project(cur_x, 0.0, 0, off_x, off_y))
            pts_sidewalk_top.append(iso_project(cur_x, -36.0, 0, off_x, off_y))
            
        # 1. Sidewalk polygon
        sw_poly = pts_sidewalk_top + list(reversed(pts_road_top))
        if len(sw_poly) >= 3:
            pygame.draw.polygon(surf, sidewalk_col, sw_poly)
            
            # Sidewalk transverse seams every ~25 world units
            seam_step = int(x_start // 25) * 25
            while seam_step < x_end:
                if seam_step >= x_start:
                    p_s1 = iso_project(seam_step, -36.0, 0, off_x, off_y)
                    p_s2 = iso_project(seam_step, -24.0, 0, off_x, off_y)
                    pygame.draw.line(surf, seam_col, p_s1, p_s2, 1)
                seam_step += 25
                
        # 2. Road asphalt polygon
        road_poly = pts_road_top + list(reversed(pts_road_bot))
        if len(road_poly) >= 3:
            pygame.draw.polygon(surf, asphalt, road_poly)
            
            # Solid crisp yellow centerline (exact Project Zomboid look)
            pygame.draw.lines(surf, yellow_line, False, pts_center, 2)
            
            # 3. 3D Beveled Concrete Curb
            # Curb lip: upper road edge with highlight and shadow line
            pygame.draw.lines(surf, curb_top, False, pts_road_top, 3)
            pygame.draw.lines(surf, curb_shadow, False, [(u, v + 1) for u, v in pts_road_top], 1)
            
            # Outer road edge line
            pygame.draw.lines(surf, (60, 62, 66), False, pts_road_bot, 2)

    def _render_car_shadow(self, surf, car, off_x, off_y):
        """Cast subtle isometric ground shadow under vehicle and trailer."""
        cx, cy = car.pos
        hl = car.length * 0.52
        hw = car.width * 0.58
        
        # Shadow offset towards bottom-right (Project Zomboid sun angle)
        corners = [
            (cx - hl + 1.0, cy - hw + 2.0),
            (cx + hl + 1.0, cy - hw + 2.0),
            (cx + hl + 1.0, cy + hw + 2.0),
            (cx - hl + 1.0, cy + hw + 2.0)
        ]
        poly = [iso_project(x, y, 0, off_x, off_y) for x, y in corners]
        pygame.draw.polygon(surf, (22, 25, 28), poly)
        
        # Trailer shadow if attached
        if car.has_trailer:
            tx = cx - car.length * 0.85
            t_hl = 9.0
            t_hw = car.width * 0.5
            t_corners = [
                (tx - t_hl + 1.0, cy - t_hw + 2.0),
                (tx + t_hl + 1.0, cy - t_hw + 2.0),
                (tx + t_hl + 1.0, cy + t_hw + 2.0),
                (tx - t_hl + 1.0, cy + t_hw + 2.0)
            ]
            t_poly = [iso_project(x, y, 0, off_x, off_y) for x, y in t_corners]
            pygame.draw.polygon(surf, (22, 25, 28), t_poly)
            
        # Headlights at night
        if self.time_of_day > 0.25:
            beam_reach = 48.0
            beam_spread = 20.0
            fx = cx + hl
            fy = cy
            p_center = iso_project(fx, fy, 0, off_x, off_y)
            p_left = iso_project(fx + beam_reach, fy - beam_spread, 0, off_x, off_y)
            p_right = iso_project(fx + beam_reach, fy + beam_spread, 0, off_x, off_y)
            
            beam_surf = pygame.Surface((320, 240), pygame.SRCALPHA)
            alpha = int(90 * self.time_of_day)
            pygame.draw.polygon(beam_surf, (255, 245, 180, alpha), [p_center, p_left, p_right])
            surf.blit(beam_surf, (0, 0), special_flags=pygame.BLEND_ADD)

    def _render_car_pz(self, surf, car, off_x, off_y):
        """
        Renders Project Zomboid 90s vehicle:
        - Boxy chassis with 2-tone hood / cab
        - Detailed front grille, dual headlights, amber indicators, chrome bumper
        - 4 rubber tires under wheel wells
        - Sloped windshield with glass tint & driver avatar
        - Decals: Emergency lightbar (flashing beacons!), racing stripes, star/logo, roof rack
        - Utility trailer with blue tarp cover (if attached)
        """
        cx, cy = car.pos
        eff_color = self._get_car_effective_color(car.base_color)
        top_color = self._apply_lighting_to_color(eff_color)
        hood_eff = self._get_car_effective_color(car.hood_color)
        hood_col = self._apply_lighting_to_color(hood_eff)
        
        side_color1 = self._apply_lighting_to_color(
            (int(eff_color[0] * 0.72), int(eff_color[1] * 0.72), int(eff_color[2] * 0.72))
        )
        side_color2 = self._apply_lighting_to_color(
            (int(eff_color[0] * 0.58), int(eff_color[1] * 0.58), int(eff_color[2] * 0.58))
        )
        
        hl = car.length * 0.5
        hw = car.width * 0.5
        h1 = car.h_chassis
        h2 = h1 + car.h_cabin
        
        # 1. Wheels & Tires (Black rubber tires with silver hubcaps)
        wheel_offsets = [
            (hl * 0.65, -hw * 0.95),  # Front-Left
            (hl * 0.65,  hw * 0.95),  # Front-Right
            (-hl * 0.65,  hw * 0.95), # Rear-Right
            (-hl * 0.65, -hw * 0.95), # Rear-Left
        ]
        tire_col = self._apply_lighting_to_color((24, 26, 28))
        rim_col = self._apply_lighting_to_color((180, 185, 190))
        for wx, wy in wheel_offsets:
            wpt = iso_project(cx + wx, cy + wy, 1.5, off_x, off_y)
            pygame.draw.circle(surf, tire_col, wpt, 3)
            pygame.draw.circle(surf, rim_col, wpt, 1)
            
        # 2. Lower Chassis Box
        local_c = [(hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)]
        c_bot = [iso_project(cx + lx, cy + ly, 1.5, off_x, off_y) for lx, ly in local_c]
        c_top = [iso_project(cx + lx, cy + ly, h1, off_x, off_y) for lx, ly in local_c]
        
        # Draw side panels
        for i in range(4):
            nxt = (i + 1) % 4
            poly = [c_bot[i], c_bot[nxt], c_top[nxt], c_top[i]]
            col = side_color1 if i % 2 == 0 else side_color2
            pygame.draw.polygon(surf, col, poly)
            
        # Chassis top (Hood & rear bed)
        pygame.draw.polygon(surf, top_color, c_top)
        
        # Two-tone front hood (like the white/maroon ambulance/fire truck in the screenshot)
        if car.two_tone_hood:
            hood_local = [(hl, -hw), (hl, hw), (hl * 0.2, hw), (hl * 0.2, -hw)]
            hood_pts = [iso_project(cx + lx, cy + ly, h1, off_x, off_y) for lx, ly in hood_local]
            pygame.draw.polygon(surf, hood_col, hood_pts)
            
        # 3. Front Grille, Headlights, Bumper & Turn Signals
        chrome_col = self._apply_lighting_to_color((190, 195, 200))
        amber_col = self._apply_lighting_to_color((235, 165, 30))
        headlight_col = self._apply_lighting_to_color((250, 248, 230))
        
        # Front Bumper
        b1 = iso_project(cx + hl + 0.5, cy - hw, 1.0, off_x, off_y)
        b2 = iso_project(cx + hl + 0.5, cy + hw, 1.0, off_x, off_y)
        b3 = iso_project(cx + hl + 0.5, cy + hw, 3.2, off_x, off_y)
        b4 = iso_project(cx + hl + 0.5, cy - hw, 3.2, off_x, off_y)
        pygame.draw.polygon(surf, chrome_col, [b1, b2, b3, b4])
        
        # Radiator Grille (Dark rectangle between headlights)
        g1 = iso_project(cx + hl + 0.2, cy - hw * 0.45, 3.5, off_x, off_y)
        g2 = iso_project(cx + hl + 0.2, cy + hw * 0.45, 3.5, off_x, off_y)
        g3 = iso_project(cx + hl + 0.2, cy + hw * 0.45, h1 - 0.5, off_x, off_y)
        g4 = iso_project(cx + hl + 0.2, cy - hw * 0.45, h1 - 0.5, off_x, off_y)
        pygame.draw.polygon(surf, (25, 26, 30), [g1, g2, g3, g4])
        
        # Dual Square Headlights
        hl_l = iso_project(cx + hl + 0.2, cy - hw * 0.70, 4.5, off_x, off_y)
        hl_r = iso_project(cx + hl + 0.2, cy + hw * 0.70, 4.5, off_x, off_y)
        pygame.draw.circle(surf, headlight_col, hl_l, 2)
        pygame.draw.circle(surf, headlight_col, hl_r, 2)
        
        # Amber Turn Signals on outer edges
        sig_l = iso_project(cx + hl + 0.2, cy - hw * 0.88, 4.2, off_x, off_y)
        sig_r = iso_project(cx + hl + 0.2, cy + hw * 0.88, 4.2, off_x, off_y)
        pygame.draw.circle(surf, amber_col, sig_l, 1)
        pygame.draw.circle(surf, amber_col, sig_r, 1)
        
        # 4. Cabin & Windshield
        # If pickup, cabin is front half, rear is bed
        cabin_hl = hl * 0.38 if car.has_bed else hl * 0.52
        cabin_hw = hw * 0.84
        cabin_cx = cx + (hl * 0.15 if car.has_bed else -hl * 0.05)
        
        cab_local = [
            (cabin_hl, -cabin_hw), (cabin_hl, cabin_hw),
            (-cabin_hl, cabin_hw), (-cabin_hl, -cabin_hw)
        ]
        cab_bot = [iso_project(cabin_cx + lx, cy + ly, h1, off_x, off_y) for lx, ly in cab_local]
        cab_top = [iso_project(cabin_cx + lx, cy + ly, h2, off_x, off_y) for lx, ly in cab_local]
        
        # Windshield glass (tinted dark blue/grey)
        glass_col = self._apply_lighting_to_color((48, 62, 74))
        for i in range(4):
            nxt = (i + 1) % 4
            poly = [cab_bot[i], cab_bot[nxt], cab_top[nxt], cab_top[i]]
            pygame.draw.polygon(surf, glass_col, poly)
            
        # 5. Visible Driver Avatar (inside windshield)
        is_obscured = (self.time_of_day > 0.40) or (self.weather_slider > 0.35)
        if not is_obscured:
            drv_x = cabin_cx + cabin_hl * 0.20
            drv_y = cy - cabin_hw * 0.25
            
            # Head
            head_pt = iso_project(drv_x, drv_y, h1 + car.h_cabin * 0.68, off_x, off_y)
            flesh_col = self._apply_lighting_to_color((235, 195, 160))
            pygame.draw.circle(surf, flesh_col, head_pt, 2)
            
            # Colored shirt
            shirt_col = self._apply_lighting_to_color(car.driver_color)
            p_s1 = iso_project(drv_x - 1.5, drv_y - 2.5, h1 + car.h_cabin * 0.30, off_x, off_y)
            p_s2 = iso_project(drv_x + 1.5, drv_y - 2.5, h1 + car.h_cabin * 0.30, off_x, off_y)
            p_s3 = iso_project(drv_x + 1.5, drv_y + 2.5, h1 + car.h_cabin * 0.30, off_x, off_y)
            p_s4 = iso_project(drv_x - 1.5, drv_y + 2.5, h1 + car.h_cabin * 0.30, off_x, off_y)
            pygame.draw.polygon(surf, shirt_col, [p_s1, p_s2, p_s3, p_s4])
        else:
            # Windshield glare reflection
            glare_col = self._apply_lighting_to_color((24, 30, 38))
            pygame.draw.polygon(surf, glare_col, [cab_bot[0], cab_bot[1], cab_top[1], cab_top[0]])
            
        # Cabin Roof Top
        roof_col = hood_col if car.two_tone_hood else top_color
        pygame.draw.polygon(surf, roof_col, cab_top)

        # 6. Attached Utility Trailer (if equipped)
        if car.has_trailer:
            self._render_trailer_pz(surf, car, off_x, off_y)

    def _render_trailer_pz(self, surf, car, off_x, off_y):
        """Renders attached utility trailer with colored tarp cover (like screenshot)."""
        cx, cy = car.pos
        tx = cx - car.length * 0.85
        t_hl = 8.5
        t_hw = car.width * 0.48
        t_h = 4.5
        
        # Tow Hitch bar
        hitch_start = iso_project(cx - car.length * 0.5, cy, 2.0, off_x, off_y)
        hitch_end = iso_project(tx + t_hl, cy, 2.0, off_x, off_y)
        pygame.draw.line(surf, (80, 85, 90), hitch_start, hitch_end, 2)
        
        # Trailer wheels
        w_l = iso_project(tx, cy - t_hw * 1.05, 1.5, off_x, off_y)
        w_r = iso_project(tx, cy + t_hw * 1.05, 1.5, off_x, off_y)
        tire_col = self._apply_lighting_to_color((24, 26, 28))
        rim_col = self._apply_lighting_to_color((180, 185, 190))
        pygame.draw.circle(surf, tire_col, w_l, 3)
        pygame.draw.circle(surf, rim_col, w_l, 1)
        pygame.draw.circle(surf, tire_col, w_r, 3)
        pygame.draw.circle(surf, rim_col, w_r, 1)
        
        # Trailer body frame
        t_local = [(t_hl, -t_hw), (t_hl, t_hw), (-t_hl, t_hw), (-t_hl, -t_hw)]
        t_bot = [iso_project(tx + lx, cy + ly, 1.5, off_x, off_y) for lx, ly in t_local]
        t_top = [iso_project(tx + lx, cy + ly, t_h, off_x, off_y) for lx, ly in t_local]
        
        # Frame sides
        frame_col = self._apply_lighting_to_color((135, 140, 145))
        for i in range(4):
            nxt = (i + 1) % 4
            poly = [t_bot[i], t_bot[nxt], t_top[nxt], t_top[i]]
            pygame.draw.polygon(surf, frame_col, poly)
            
        # Blue tarp cover on top (exact match to screenshot)
        tarp_col = self._apply_lighting_to_color(car.trailer_color)
        pygame.draw.polygon(surf, tarp_col, t_top)

    def _render_overpass_bridge(self, surf, off_x, off_y, bridge_x):
        """Overpass bridge that occludes traffic on CAM 02."""
        bridge_col = self._apply_lighting_to_color((70, 75, 82))
        rail_col = self._apply_lighting_to_color((110, 115, 122))
        
        b_len = 24.0
        b_wid = 40.0
        z_clearance = 16.0
        z_height = 20.0
        
        corners = [
            (bridge_x - b_len * 0.5, -b_wid),
            (bridge_x + b_len * 0.5, -b_wid),
            (bridge_x + b_len * 0.5,  b_wid),
            (bridge_x - b_len * 0.5,  b_wid)
        ]
        
        pts_bot = [iso_project(x, y, z_clearance, off_x, off_y) for x, y in corners]
        pts_top = [iso_project(x, y, z_height, off_x, off_y) for x, y in corners]
        
        # Pillars
        p_col = self._apply_lighting_to_color((52, 56, 62))
        p1 = iso_project(bridge_x, -b_wid * 0.8, 0, off_x, off_y)
        p2 = iso_project(bridge_x, -b_wid * 0.8, z_clearance, off_x, off_y)
        p3 = iso_project(bridge_x,  b_wid * 0.8, 0, off_x, off_y)
        p4 = iso_project(bridge_x,  b_wid * 0.8, z_clearance, off_x, off_y)
        pygame.draw.line(surf, p_col, p1, p2, 5)
        pygame.draw.line(surf, p_col, p3, p4, 5)
        
        for i in range(4):
            nxt = (i + 1) % 4
            poly = [pts_bot[i], pts_bot[nxt], pts_top[nxt], pts_top[i]]
            pygame.draw.polygon(surf, bridge_col, poly)
        pygame.draw.polygon(surf, rail_col, pts_top)

    def _render_highway_gantry(self, surf, off_x, off_y, gantry_x):
        """Overhead highway sign on CAM 03."""
        frame_col = self._apply_lighting_to_color((85, 90, 100))
        sign_col = self._apply_lighting_to_color((24, 115, 65))
        
        p1 = iso_project(gantry_x, -28.0, 0, off_x, off_y)
        p2 = iso_project(gantry_x, -28.0, 18.0, off_x, off_y)
        p3 = iso_project(gantry_x,  28.0, 0, off_x, off_y)
        p4 = iso_project(gantry_x,  28.0, 18.0, off_x, off_y)
        pygame.draw.line(surf, frame_col, p1, p2, 4)
        pygame.draw.line(surf, frame_col, p3, p4, 4)
        pygame.draw.line(surf, frame_col, p2, p4, 3)
        
        s1 = iso_project(gantry_x, -18.0, 14.0, off_x, off_y)
        s2 = iso_project(gantry_x,  18.0, 14.0, off_x, off_y)
        s3 = iso_project(gantry_x,  18.0, 19.0, off_x, off_y)
        s4 = iso_project(gantry_x, -18.0, 19.0, off_x, off_y)
        pygame.draw.polygon(surf, sign_col, [s1, s2, s3, s4])

    def _apply_weather_shaders(self, surf):
        w = self.weather_slider
        if w > 0.05:
            rain_col = (185, 205, 225, int(150 * w))
            self._rain_surf.fill((0, 0, 0, 0))
            for drop in self.rain_drops:
                x, y, length = drop
                pygame.draw.line(self._rain_surf, rain_col, (x, y), (x + 3, y + int(length)), 1)
            surf.blit(self._rain_surf, (0, 0))
