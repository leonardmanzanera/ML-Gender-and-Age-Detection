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
    "left_eye_top":    159,
    "left_eye_bottom": 145,
    "right_eye_inner":362,
    "right_eye_outer":263,
    "right_eye_top":   386,
    "right_eye_bottom":374,

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

    # Forehead center
    "forehead_center": 151,
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
        return max(0.0, 1.0 - deviation * 1.5)

    def _ratio_score(self, ratio, ideal):
        if ideal == 0:
            return 0.0
        deviation = abs(ratio - ideal) / ideal
        return max(0.0, 1.0 - deviation * 1.5)

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

        # ─── 3. Regard (Eye Analysis) ───
        # Canthal Tilt: angle between inner and outer corners (positive if outer is higher -> lower y)
        left_tilt = math.degrees(math.atan2(l_eye_i[1] - l_eye_o[1], l_eye_o[0] - l_eye_i[0]))
        right_tilt = math.degrees(math.atan2(r_eye_i[1] - r_eye_o[1], r_eye_i[0] - r_eye_o[0]))
        avg_tilt = (left_tilt + right_tilt) / 2.0
        # Positive tilt (e.g., 5-8 degrees) is often considered attractive
        tilt_score = max(0.0, 10.0 - abs(avg_tilt - 6.0) * 1.0)
        
        # Eye Openness
        l_eye_t = self._get_point(landmarks, "left_eye_top", w, h)
        l_eye_b = self._get_point(landmarks, "left_eye_bottom", w, h)
        r_eye_t = self._get_point(landmarks, "right_eye_top", w, h)
        r_eye_b = self._get_point(landmarks, "right_eye_bottom", w, h)
        
        l_openness = self._dist(l_eye_t, l_eye_b) / left_eye_w if left_eye_w > 1 else 0
        r_openness = self._dist(r_eye_t, r_eye_b) / right_eye_w if right_eye_w > 1 else 0
        avg_openness = (l_openness + r_openness) / 2.0
        openness_score = 10.0 if 0.3 < avg_openness < 0.5 else max(0.0, 10.0 - abs(avg_openness - 0.4)*20)
        
        regard_score = round(min(10.0, tilt_score * 0.6 + openness_score * 0.4), 1)

        # ─── 4. Harmonie Verticale (Facial Thirds) ───
        # Top 1/3: Hairline (approx forehead_top) to Brow
        # Mid 1/3: Brow to Nose base
        # Low 1/3: Nose base to Chin
        third_1 = abs(brow_center_y - forehead[1])
        third_2 = abs(nose_tip[1] - brow_center_y)
        third_3 = abs(chin[1] - nose_tip[1])
        
        avg_third = (third_1 + third_2 + third_3) / 3.0
        harmonie_score = 10.0
        if avg_third > 1:
            dev1 = abs(third_1 - avg_third) / avg_third
            dev2 = abs(third_2 - avg_third) / avg_third
            dev3 = abs(third_3 - avg_third) / avg_third
            total_dev = (dev1 + dev2 + dev3) / 3.0
            # A 10% average deviation drops the score by 1.5 points
            harmonie_score = max(0.0, 10.0 - (total_dev * 15.0))
        harmonie_score = round(harmonie_score, 1)

        # ─── 5. Teint (Skin Texture) ───
        # Extract patches from cheeks and forehead to measure variance (smoothness)
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        variances = []
        patch_size = max(5, int(w * 0.05))
        
        fh_center = self._get_point(landmarks, "forehead_center", w, h)
        # Approximate cheek centers by moving down from eye and out from nose
        l_cheek_inner = (l_eye_b[0], nose_l[1])
        r_cheek_inner = (r_eye_b[0], nose_r[1])
        
        for pt in [fh_center, l_cheek_inner, r_cheek_inner]:
            x, y = int(pt[0]), int(pt[1])
            x1, y1 = max(0, x - patch_size), max(0, y - patch_size)
            x2, y2 = min(w, x + patch_size), min(h, y + patch_size)
            patch = gray[y1:y2, x1:x2]
            if patch.size > 0:
                var = cv2.Laplacian(patch, cv2.CV_64F).var()
                variances.append(var)
                
        if variances:
            avg_var = sum(variances) / len(variances)
            # Higher variance = sharper edges/more pores. Lower = smoother.
            teint_score = max(0.0, 10.0 - (avg_var / 50.0))
        else:
            teint_score = 5.0
        teint_score = round(teint_score, 1)

        # ─── 6. Final Golden Score (0-10) ───
        # Base aesthetic score (60% phi + 40% symmetry)
        base_score = (phi_score * 0.5 + symmetry_score * 0.5) * 10.0
        
        # Add slight boosts from Regard, Harmonie, Teint
        boost = 0.0
        if harmonie_score > 7.0: boost += 0.3
        if regard_score > 8.0: boost += 0.2
        if teint_score > 8.0: boost += 0.2
        
        final = round(min(10.0, max(0.0, base_score + boost)), 1)
        
        radar_scores = {
            "Symmetry": round(symmetry_score * 10.0, 1),
            "Phi": round(phi_score * 10.0, 1),
            "Regard": regard_score,
            "Harmonie": harmonie_score,
            "Teint": teint_score
        }

        return {
            "golden_score": final,
            "radar": radar_scores,
            "symmetry_pct": round(symmetry_score * 100, 1),
            "phi_pct": round(phi_score * 100, 1),
            "raw_landmarks": landmarks, # Passed for drawing the Golden Mask
            "ratios": {
                "face_h_w": round(r1, 3),
                "eye_inter": round(r2, 3),
                "nose_lip_chin": round(r3, 3),
                "nose_mouth": round(r4, 3),
                "thirds": round(r5, 3),
            }
        }
