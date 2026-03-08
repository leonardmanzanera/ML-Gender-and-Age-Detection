"""
AG Vision - Posture Coach (V7: Coach Posture & Bien-être)
Uses MediaPipe PoseLandmarker (Tasks API) for real-time posture analysis.
"""

import time
import math
import os
import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )
    HAS_MP = True
except (ImportError, AttributeError):
    HAS_MP = False
    print("[!] mediapipe Tasks API not available. Posture Coach disabled.")


class PostureCoach:
    """
    Real-time posture analysis using MediaPipe Pose landmarks.
    Detects slouching (shoulder-ear angle) and recommends breaks.
    """

    SLOUCH_ANGLE_THRESHOLD = 25  # degrees
    BREAK_INTERVAL_SEC = 30 * 60  # 30 minutes

    # Landmark indices (MediaPipe Pose)
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    def __init__(self):
        self.landmarker = None
        self.latest_landmarks = None
        self.session_start = time.time()
        self.is_slouching = False
        self.slouch_start = 0
        self.break_notified = False

        if not HAS_MP:
            return

        # Download the pose model if needed
        model_path = self._ensure_model()
        if model_path is None:
            return

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    @staticmethod
    def _ensure_model():
        """Ensure the pose landmarker model is available."""
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_dir = os.path.join(root, "models")
        model_path = os.path.join(model_dir, "pose_landmarker_lite.task")

        if os.path.exists(model_path):
            return model_path

        # Download the model
        print("[*] Downloading MediaPipe Pose model...")
        os.makedirs(model_dir, exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        try:
            import urllib.request
            urllib.request.urlretrieve(url, model_path)
            print(f" [+] Pose model saved to {model_path}")
            return model_path
        except Exception as e:
            print(f" [!] Failed to download pose model: {e}")
            return None

    def _angle_from_vertical(self, p1, p2):
        """Calculate angle from vertical between two points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(abs(dx), abs(dy)))

    def analyze(self, frame):
        """
        Analyze posture from a BGR frame.
        Returns dict with posture status and alerts.
        """
        if self.landmarker is None:
            return {"status": "disabled", "alert": None, "landmarks": None}

        # Convert to MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run detection
        timestamp_ms = int(time.time() * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        status = "no_pose"
        alert = None
        angle = 0

        # Break reminder
        elapsed = time.time() - self.session_start
        if elapsed > self.BREAK_INTERVAL_SEC and not self.break_notified:
            alert = "⏰ Pause recommandée ! (30 min écoulées)"
            self.break_notified = True

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            h, w = frame.shape[:2]

            # Average ear and shoulder positions
            ear = (
                (landmarks[self.LEFT_EAR].x + landmarks[self.RIGHT_EAR].x) / 2 * w,
                (landmarks[self.LEFT_EAR].y + landmarks[self.RIGHT_EAR].y) / 2 * h,
            )
            shoulder = (
                (landmarks[self.LEFT_SHOULDER].x + landmarks[self.RIGHT_SHOULDER].x) / 2 * w,
                (landmarks[self.LEFT_SHOULDER].y + landmarks[self.RIGHT_SHOULDER].y) / 2 * h,
            )

            angle = self._angle_from_vertical(shoulder, ear)

            if angle > self.SLOUCH_ANGLE_THRESHOLD:
                status = "slouching"
                if not self.is_slouching:
                    self.slouch_start = time.time()
                    self.is_slouching = True
                if (time.time() - self.slouch_start) > 5:
                    alert = f"🧘 Redressez-vous ! ({int(angle)}°)"
            else:
                status = "good"
                self.is_slouching = False

            self.latest_landmarks = landmarks

        return {
            "status": status,
            "alert": alert,
            "angle": round(angle, 1),
            "landmarks": self.latest_landmarks,
        }

    def draw_skeleton(self, frame, result):
        """Draw a simple skeleton on the frame."""
        landmarks = result.get("landmarks")
        if landmarks is None:
            return frame

        h, w = frame.shape[:2]
        color = (0, 255, 0) if result["status"] == "good" else (0, 0, 255)

        # Draw key connections (simplified skeleton)
        connections = [
            (self.LEFT_SHOULDER, self.RIGHT_SHOULDER),
            (self.LEFT_EAR, self.LEFT_SHOULDER),
            (self.RIGHT_EAR, self.RIGHT_SHOULDER),
            (11, 23), (12, 24), (23, 24),  # torso
        ]

        for start, end in connections:
            if start < len(landmarks) and end < len(landmarks):
                pt1 = (int(landmarks[start].x * w), int(landmarks[start].y * h))
                pt2 = (int(landmarks[end].x * w), int(landmarks[end].y * h))
                cv2.line(frame, pt1, pt2, color, 2)

        # Draw key points
        for idx in [self.LEFT_EAR, self.RIGHT_EAR, self.LEFT_SHOULDER, self.RIGHT_SHOULDER, 23, 24]:
            if idx < len(landmarks):
                pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                cv2.circle(frame, pt, 4, color, -1)

        return frame

    def reset_timer(self):
        """Reset the break timer."""
        self.session_start = time.time()
        self.break_notified = False
