"""
AG Vision - Tracked ViT Engine (V8: Multi-Person ID Isolation)
Resolves: age/gender cross-contamination when multiple people are in frame.

Architecture:
  - Internal queue of (track_id, crop) tuples processed FIFO
  - Per-ID result dictionary with built-in temporal smoothing
  - Stale ID purging after MAX_STALE_FRAMES
  - Hybrid inference: ViT ONNX (age) + Caffe CNN (gender)
"""

import cv2
import time
import os
import threading
import collections
import copy
import numpy as np
import onnxruntime as ort
from ag_vision.smoother import TemporalSmoother


class TrackedViTEngine:
    """
    Multi-person asynchronous inference engine.
    Each tracked face gets its own result history, preventing age/gender blending.
    """

    GENDER_LIST = ['Male', 'Female']
    # Larger queue than AsyncAestheticEngine (4) because ViT inference is
    # slower (~120ms). With 3 people in frame, 8 slots provide ~2.6 frames
    # of buffering before oldest crops are silently ejected by deque(maxlen).
    MAX_QUEUE_SIZE = 8        # Max pending crops in queue
    MAX_STALE_FRAMES = 30     # Purge IDs not seen for N frames
    SMOOTHING_WINDOW = 8      # Moving average window per person

    def __init__(self, vit_model_path):
        print(f"[*] TrackedViTEngine: Initializing Multi-Person Hybrid Engine...")

        # ViT for age regression
        print(f"    -> ViT ONNX (Age): {os.path.basename(vit_model_path)}")
        self.vit_sess = ort.InferenceSession(
            vit_model_path, providers=['CPUExecutionProvider']
        )
        self.vit_input = self.vit_sess.get_inputs()[0].name

        # Caffe for gender classification
        models_dir = os.path.dirname(vit_model_path)
        gender_proto = os.path.join(models_dir, "gender_deploy.prototxt")
        gender_model = os.path.join(models_dir, "gender_net.caffemodel")

        if os.path.exists(gender_proto) and os.path.exists(gender_model):
            print(f"    -> Caffe CNN (Gender): gender_net.caffemodel")
            self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)
            self.has_caffe_gender = True
        else:
            print(f"    [!] Caffe gender model not found. Using ViT fallback.")
            self.gender_net = None
            self.has_caffe_gender = False

        # --- Per-ID State ---
        self.lock = threading.Lock()
        self.queue = collections.deque(maxlen=self.MAX_QUEUE_SIZE)
        self.results = {}       # {track_id: {"age": ..., "gender": ..., ...}}
        self.smoother = TemporalSmoother(window_size=self.SMOOTHING_WINDOW)
        self.last_seen = {}     # {track_id: frame_counter}
        self.frame_counter = 0

        # Worker thread
        self.is_running = True
        self.new_data_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        print(f"    [+] TrackedViTEngine ready (queue={self.MAX_QUEUE_SIZE}, smooth={self.SMOOTHING_WINDOW})")

    # ─────────────────────── Preprocessing ───────────────────────

    def preprocess_vit(self, face_crop):
        """Preprocess for ViT (224x224, ImageNet normalization)."""
        try:
            img = cv2.resize(face_crop, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            return img
        except Exception:
            return None

    def classify_gender_caffe(self, face_crop):
        """Classify gender using the proven Caffe CNN model."""
        try:
            blob = cv2.dnn.blobFromImage(
                face_crop, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            self.gender_net.setInput(blob)
            preds = self.gender_net.forward()
            gender_idx = preds[0].argmax()
            confidence = float(preds[0][gender_idx])
            return self.GENDER_LIST[gender_idx], confidence
        except Exception:
            return "Unknown", 0.0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # ─────────────────────── Public API ───────────────────────

    def submit(self, track_id, crop):
        """Submit a face crop with its tracking ID for async processing."""
        with self.lock:
            self.queue.append((track_id, copy.deepcopy(crop)))
            self.last_seen[track_id] = self.frame_counter
        self.new_data_event.set()

    def tick(self):
        """Call once per frame to advance internal counter and purge stale IDs."""
        with self.lock:
            self.frame_counter += 1
            stale_ids = [
                tid for tid, last in self.last_seen.items()
                if (self.frame_counter - last) > self.MAX_STALE_FRAMES
            ]
            for tid in stale_ids:
                self.results.pop(tid, None)
                self.smoother.purge(tid)
                self.last_seen.pop(tid, None)

    def get_result(self, track_id):
        """Get the latest smoothed result for a specific track ID."""
        with self.lock:
            return copy.deepcopy(
                self.results.get(track_id, {
                    "age": "--", "gender": "Scanning",
                    "gender_prob": 1.0, "time_ms": 0.0
                })
            )

    # ─────────────────────── Worker Thread ───────────────────────

    def _worker_loop(self):
        """Background thread: processes the queue FIFO."""
        while self.is_running:
            self.new_data_event.wait()
            if not self.is_running:
                break

            # Drain the queue
            while True:
                with self.lock:
                    if len(self.queue) == 0:
                        break
                    track_id, crop = self.queue.popleft()

                if crop is None:
                    continue

                t0 = time.time()

                # --- AGE: ViT ONNX ---
                tensor = self.preprocess_vit(crop)
                raw_age = "--"
                if tensor is not None:
                    preds = self.vit_sess.run(
                        None, {self.vit_input: tensor}
                    )[0][0]
                    raw_age = float(preds[0])

                # --- GENDER: Caffe CNN or ViT fallback ---
                if self.has_caffe_gender:
                    gen_str, gen_conf = self.classify_gender_caffe(crop)
                else:
                    if tensor is not None:
                        gender_prob = self.sigmoid(preds[1])
                        if gender_prob > 0.5:
                            gen_str, gen_conf = "Female", float(gender_prob)
                        else:
                            gen_str, gen_conf = "Male", float(1.0 - gender_prob)
                    else:
                        gen_str, gen_conf = "Unknown", 0.0

                t_clf = (time.time() - t0) * 1000

                # --- Smooth & Store per ID (delegated to TemporalSmoother) ---
                with self.lock:
                    if isinstance(raw_age, (int, float)):
                        smoothed_age_str, _, smoothed_gen, smoothed_prob = \
                            self.smoother.update_and_get(
                                track_id, raw_age, 1.0, gen_str, gen_conf,
                                is_regression=True
                            )
                        smoothed_age = int(smoothed_age_str)
                    else:
                        smoothed_age = raw_age
                        smoothed_gen = gen_str
                        smoothed_prob = gen_conf

                    self.results[track_id] = {
                        "age": smoothed_age,
                        "gender": smoothed_gen,
                        "gender_prob": smoothed_prob,
                        "time_ms": t_clf
                    }

            self.new_data_event.clear()

    def stop(self):
        """Gracefully stop the worker thread."""
        self.is_running = False
        self.new_data_event.set()
        self.worker.join(timeout=2.0)
        print("[+] TrackedViTEngine stopped.")
