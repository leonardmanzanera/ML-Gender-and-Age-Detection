"""
AG Vision - Aesthetic Engine (V10)
Scores facial attractiveness using mathematical principles:
  - Golden Ratio (Phi = 1.618) proportions
  - Bilateral symmetry analysis
Uses MediaPipe Face Mesh (468 landmarks) for precision geometry.
"""

import math
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )
    HAS_MP = True
except (ImportError, AttributeError):
    HAS_MP = False
    print("[!] mediapipe Tasks API not available. Aesthetic Engine disabled.")


# Golden Ratio constant
PHI = 1.6180339887

# ─── MediaPipe Face Mesh landmark indices ───
LM = {
    # Vertical axis
    "forehead_top":    10,
    "chin_bottom":    152,
    "nose_tip":         1,

    # Face width (cheekbones)
    "left_cheek":     234,
    "right_cheek":    454,

    # Eyes — outer & inner corners
    "left_eye_outer":  33,
    "left_eye_inner": 133,
    "right_eye_inner":362,
    "right_eye_outer":263,

    # Eyebrows — outer edges
    "left_brow_outer": 46,
    "right_brow_outer":276,

    # Lips
    "upper_lip":       13,
    "lower_lip":       14,
    "mouth_left":      61,
    "mouth_right":    291,

    # Nose width
    "nose_left":       48,
    "nose_right":     278,

    # Jaw
    "jaw_left":       132,
    "jaw_right":      361,
}

SYMMETRY_PAIRS = [
    ("left_eye_outer",  "right_eye_outer"),
    ("left_eye_inner",  "right_eye_inner"),
    ("left_brow_outer", "right_brow_outer"),
    ("mouth_left",      "mouth_right"),
    ("nose_left",       "nose_right"),
    ("jaw_left",        "jaw_right"),
    ("left_cheek",      "right_cheek"),
]


class AestheticEngine:
    """
    Computes a 'Golden Score' (0-10) based on:
      - 60% Phi (Golden Ratio) proportions
      - 40% Bilateral symmetry
    """

    def __init__(self):
        self.landmarker = None
        if not HAS_MP:
            return

        model_path = self._ensure_model()
        if model_path is None:
            return

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        print(" [+] AestheticEngine: Ready (Golden Ratio + Symmetry via Tasks API)")

    @staticmethod
    def _ensure_model():
        """Ensure the face landmarker model is available."""
        import os
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_dir = os.path.join(root, "models")
        model_path = os.path.join(model_dir, "face_landmarker.task")

        if os.path.exists(model_path):
            return model_path

        from ag_vision.utils.download_utils import download_file
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        try:
            download_file(url, model_path)
            return model_path
        except Exception as e:
            print(f" [!] Failed to download face model: {e}")
            return None

    def _get_point(self, landmarks, key, w, h):
        idx = LM[key]
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)

    def _dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _phi_score(self, ratio):
        deviation = abs(ratio - PHI) / PHI
        return max(0.0, 1.0 - deviation * 2.0)

    def _ratio_score(self, ratio, ideal):
        if ideal == 0:
            return 0.0
        deviation = abs(ratio - ideal) / ideal
        return max(0.0, 1.0 - deviation * 2.0)

    def analyze(self, bgr_crop):
        if self.landmarker is None:
            return None

        h, w = bgr_crop.shape[:2]
        if h < 20 or w < 20:
            return None

        import cv2
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None

        landmarks = result.face_landmarks[0]

        # ─── 1. Golden Ratio Proportions ───
        forehead = self._get_point(landmarks, "forehead_top", w, h)
        chin     = self._get_point(landmarks, "chin_bottom", w, h)
        l_cheek  = self._get_point(landmarks, "left_cheek", w, h)
        r_cheek  = self._get_point(landmarks, "right_cheek", w, h)
        nose_tip = self._get_point(landmarks, "nose_tip", w, h)
        upper_lip= self._get_point(landmarks, "upper_lip", w, h)
        lower_lip= self._get_point(landmarks, "lower_lip", w, h)
        l_eye_o  = self._get_point(landmarks, "left_eye_outer", w, h)
        r_eye_o  = self._get_point(landmarks, "right_eye_outer", w, h)
        l_eye_i  = self._get_point(landmarks, "left_eye_inner", w, h)
        r_eye_i  = self._get_point(landmarks, "right_eye_inner", w, h)
        nose_l   = self._get_point(landmarks, "nose_left", w, h)
        nose_r   = self._get_point(landmarks, "nose_right", w, h)

        face_height = self._dist(forehead, chin)
        face_width  = self._dist(l_cheek, r_cheek)

        if face_width < 1 or face_height < 1:
            return None

        # Ratio 1: Face height / width → ideal PHI
        r1 = face_height / face_width
        s1 = self._phi_score(r1)

        # Ratio 2: Eye width vs inter-eye distance → ideal 1.0
        left_eye_w  = self._dist(l_eye_o, l_eye_i)
        right_eye_w = self._dist(r_eye_i, r_eye_o)
        inter_eye   = self._dist(l_eye_i, r_eye_i)
        avg_eye_w   = (left_eye_w + right_eye_w) / 2.0
        r2 = avg_eye_w / inter_eye if inter_eye > 1 else 0
        s2 = self._ratio_score(r2, 1.0)

        # Ratio 3: Nose-to-lip vs lip-to-chin → ideal PHI
        nose_to_lip = self._dist(nose_tip, upper_lip)
        lip_to_chin = self._dist(lower_lip, chin)
        r3 = nose_to_lip / lip_to_chin if lip_to_chin > 1 else 0
        s3 = self._phi_score(r3)

        # Ratio 4: Nose width vs mouth width → ideal 1.0 * PHI^-1 ≈ 0.618
        nose_width  = self._dist(nose_l, nose_r)
        mouth_left  = self._get_point(landmarks, "mouth_left", w, h)
        mouth_right = self._get_point(landmarks, "mouth_right", w, h)
        mouth_width = self._dist(mouth_left, mouth_right)
        r4 = nose_width / mouth_width if mouth_width > 1 else 0
        s4 = self._ratio_score(r4, 1.0 / PHI)

        # Ratio 5: Forehead-to-eyebrow vs eyebrow-to-nose → ideal 1.0
        l_brow = self._get_point(landmarks, "left_brow_outer", w, h)
        r_brow = self._get_point(landmarks, "right_brow_outer", w, h)
        brow_center_y = (l_brow[1] + r_brow[1]) / 2.0
        forehead_to_brow = abs(brow_center_y - forehead[1])
        brow_to_nose = abs(nose_tip[1] - brow_center_y)
        r5 = forehead_to_brow / brow_to_nose if brow_to_nose > 1 else 0
        s5 = self._ratio_score(r5, 1.0)

        phi_score = (s1 + s2 + s3 + s4 + s5) / 5.0

        # ─── 2. Bilateral Symmetry ───
        # Midline = average X of forehead_top and chin_bottom
        midline_x = (forehead[0] + chin[0]) / 2.0

        sym_scores = []
        for left_key, right_key in SYMMETRY_PAIRS:
            left_pt  = self._get_point(landmarks, left_key, w, h)
            right_pt = self._get_point(landmarks, right_key, w, h)

            dist_left  = abs(left_pt[0] - midline_x)
            dist_right = abs(right_pt[0] - midline_x)

            if max(dist_left, dist_right) < 1:
                sym_scores.append(1.0)
                continue

            ratio = min(dist_left, dist_right) / max(dist_left, dist_right)
            sym_scores.append(ratio)

        symmetry_score = sum(sym_scores) / len(sym_scores) if sym_scores else 0.5

        # ─── 3. Final Golden Score (0-10) ───
        # 60% phi + 40% symmetry
        final = (phi_score * 0.6 + symmetry_score * 0.4) * 10.0
        final = round(min(10.0, max(0.0, final)), 1)

        return {
            "golden_score": final,
            "symmetry_pct": round(symmetry_score * 100, 1),
            "phi_pct": round(phi_score * 100, 1),
            "ratios": {
                "face_h_w": round(r1, 3),
                "eye_inter": round(r2, 3),
                "nose_lip_chin": round(r3, 3),
                "nose_mouth": round(r4, 3),
                "thirds": round(r5, 3),
            }
        }
