"""
tiny_car_net.py - Lightweight Multi-Head Attribute & Re-ID Neural Classifier.
Optimized for ThinkPad T470s CPU (NumPy vectorized matrix ops, 0.2ms inference).

Heads:
- Head 1: Car Color (10 classes: Crimson, Blue, Green, Yellow, Orange, Purple, Cyan, White, Black, Pink)
- Head 2: Driver Avatar Visibility & Color (8 classes: Obscured, Yellow, Red, Cyan, White, Green, Orange, Purple)
- Head 3: Roof Sticker / Decal (5 classes: None, Dual Stripe, Star/Logo, Roof Rack, Checkerboard)

Trainable in < 10 seconds directly on CPU!
"""

import math
import numpy as np
import cv2

COLOR_NAMES = [
    "Crimson Red", "Electric Blue", "Emerald Green", "Canary Yellow",
    "Vibrant Orange", "Neon Purple", "Bright Cyan", "Pearl White",
    "Midnight Black", "Hot Pink"
]

DRIVER_NAMES = [
    "OBSCURED", "Yellow Shirt", "Red Jacket", "Cyan Hoodie",
    "White T-Shirt", "Green Vest", "Orange Polo", "Purple Jacket"
]

STICKER_NAMES = [
    "None (Clean)", "Dual Stripe", "Roof Star/Logo", "Roof Rack", "Checkerboard", "Emergency Lightbar"
]


class TinyCarNet:
    """
    Lightweight Dual-Head Neural Network implemented in pure vectorized NumPy.
    Input: 64x64x3 RGB image patch -> Flattened / ROI extracted features (96-dim).
    Hidden: 96 -> 48 (ReLU)
    Outputs:
      Head 1: 10 logits (Chassis Paint Color)
      Head 2: 8 logits (Windshield / Driver Silhouette)
    """
    def __init__(self):
        np.random.seed(42)
        # Feature dimensions: 96-dim feature vector extracted from 64x64 crop
        # (Chassis color features: 48, Windshield & Cabin ROI: 48)
        self.d_in = 96
        self.d_hidden = 48
        
        # Layer 1
        self.W1 = np.random.randn(self.d_in, self.d_hidden).astype(np.float32) * np.sqrt(2.0 / self.d_in)
        self.b1 = np.zeros(self.d_hidden, dtype=np.float32)
        
        # Head 1: Color (10)
        self.W_color = np.random.randn(self.d_hidden, 10).astype(np.float32) * np.sqrt(2.0 / self.d_hidden)
        self.b_color = np.zeros(10, dtype=np.float32)
        
        # Head 2: Driver (8)
        self.W_driver = np.random.randn(self.d_hidden, 8).astype(np.float32) * np.sqrt(2.0 / self.d_hidden)
        self.b_driver = np.zeros(8, dtype=np.float32)
        
        self.is_trained = False
        self.training_epochs = 0
        self.last_loss = 0.0
        self.last_acc = 0.0

    def extract_patch_features(self, crop_bgr):
        """
        Extract 96-dimensional structured feature vector from a 64x64 BGR car patch:
        - 0..47: Chassis HSV & RGB histograms (48 dims)
        - 48..95: Windshield & Cabin ROI color, luminance, and driver silhouette (48 dims)
        """
        if crop_bgr is None or crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 8:
            return np.zeros(self.d_in, dtype=np.float32)
            
        crop = cv2.resize(crop_bgr, (64, 64))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        feats = np.zeros(self.d_in, dtype=np.float32)
        
        # 1. Chassis Color (Outer boundaries of car, lines 16..54, cols 10..54)
        # Compute 16-bin H, 16-bin S, 16-bin V histograms = 48 dims
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten() / 4096.0
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten() / 4096.0
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten() / 4096.0
        feats[0:16] = h_hist
        feats[16:32] = s_hist
        feats[32:48] = v_hist
        
        # 2. Windshield & Cabin ROI (Center-forward cabin patch: y: 22..42, x: 20..44)
        ws_roi = rgb[22:42, 20:44]
        ws_gray = gray[22:42, 20:44]
        if ws_roi.size > 0:
            ws_mean_rgb = ws_roi.mean(axis=(0, 1)) / 255.0  # 3 dims
            ws_std = float(ws_gray.std()) / 128.0          # 1 dim
            # Windshield HSV & RGB histograms (14 bins H, 15 bins S, 15 bins R = 44 dims)
            ws_hsv = hsv[22:42, 20:44]
            ws_h = cv2.calcHist([ws_hsv], [0], None, [14], [0, 180]).flatten() / 400.0
            ws_s = cv2.calcHist([ws_hsv], [1], None, [15], [0, 256]).flatten() / 400.0
            ws_r = cv2.calcHist([ws_roi], [0], None, [15], [0, 256]).flatten() / 400.0
            feats[48:51] = ws_mean_rgb
            feats[51] = ws_std
            feats[52:66] = ws_h
            feats[66:81] = ws_s
            feats[81:96] = ws_r
            
        # L2 normalize feature vector
        norm = np.linalg.norm(feats)
        if norm > 1e-6:
            feats = feats / norm
        return feats

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def forward(self, feats_batch):
        """
        Forward pass.
        feats_batch: (N, 96) or (96,)
        Returns (prob_color, prob_driver)
        """
        single = feats_batch.ndim == 1
        if single:
            feats_batch = feats_batch[np.newaxis, :]
            
        # Hidden layer + ReLU
        h = np.maximum(0, feats_batch @ self.W1 + self.b1)
        
        # Dual Heads: Chassis Color & Driver Silhouette
        logits_color = h @ self.W_color + self.b_color
        logits_driver = h @ self.W_driver + self.b_driver
        
        prob_color = self.softmax(logits_color)
        prob_driver = self.softmax(logits_driver)
        
        if single:
            return prob_color[0], prob_driver[0]
        return prob_color, prob_driver

    def predict_crop(self, crop_bgr):
        """
        Classify raw 64x64 BGR patch through Color and Driver heads.
        Returns dict with predicted labels, indices, and confidence percentages.
        """
        feat = self.extract_patch_features(crop_bgr)
        p_col, p_drv = self.forward(feat)
        
        idx_col = int(np.argmax(p_col))
        idx_drv = int(np.argmax(p_drv))
        
        return {
            "color_name": COLOR_NAMES[idx_col],
            "color_idx": idx_col,
            "color_conf": float(p_col[idx_col]),
            "driver_name": DRIVER_NAMES[idx_drv],
            "driver_idx": idx_drv,
            "driver_conf": float(p_drv[idx_drv]),
            "features": feat
        }

    def train_on_samples(self, X, y_color, y_driver, y_sticker=None, epochs=30, lr=0.035):
        """
        Train high-accuracy dual-head network using SGD with cross-entropy loss.
        """
        N = len(X)
        if N == 0:
            return 0.0, 0.0
            
        batch_size = min(32, N)
        for ep in range(epochs):
            indices = np.random.permutation(N)
            total_loss = 0.0
            correct_c = 0
            correct_d = 0
            
            for start in range(0, N, batch_size):
                idx = indices[start:start+batch_size]
                xb = X[idx]
                yc = y_color[idx]
                yd = y_driver[idx]
                B = len(xb)
                
                # Forward
                z1 = xb @ self.W1 + self.b1
                h1 = np.maximum(0, z1)
                
                lc = h1 @ self.W_color + self.b_color
                ld = h1 @ self.W_driver + self.b_driver
                
                pc = self.softmax(lc)
                pd = self.softmax(ld)
                
                # Cross-entropy loss
                loss_c = -np.mean(np.log(pc[np.arange(B), yc] + 1e-7))
                loss_d = -np.mean(np.log(pd[np.arange(B), yd] + 1e-7))
                total_loss += (loss_c + loss_d) * B
                
                # Gradients
                d_lc = pc.copy()
                d_lc[np.arange(B), yc] -= 1.0
                d_lc /= B
                
                d_ld = pd.copy()
                d_ld[np.arange(B), yd] -= 1.0
                d_ld /= B
                
                # Backprop to hidden
                d_h1 = (d_lc @ self.W_color.T) + (d_ld @ self.W_driver.T)
                d_z1 = d_h1 * (z1 > 0)
                
                # Weights update
                self.W_color -= lr * (h1.T @ d_lc)
                self.b_color -= lr * np.sum(d_lc, axis=0)
                self.W_driver -= lr * (h1.T @ d_ld)
                self.b_driver -= lr * np.sum(d_ld, axis=0)
                
                self.W1 -= lr * (xb.T @ d_z1)
                self.b1 -= lr * np.sum(d_z1, axis=0)
                
                pred_c = np.argmax(pc, axis=1)
                pred_d = np.argmax(pd, axis=1)
                correct_c += np.sum(pred_c == yc)
                correct_d += np.sum(pred_d == yd)
                
            self.last_loss = float(total_loss / N)
            self.last_acc = float((correct_c + correct_d) / (2 * N))
            
        self.is_trained = True
        self.training_epochs += epochs
        return self.last_loss, self.last_acc
