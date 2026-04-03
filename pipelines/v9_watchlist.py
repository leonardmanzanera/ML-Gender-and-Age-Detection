"""
AG Vision - V9 Watchlist Pipeline (Most Wanted Face Matching)
Fix: Resolves age/gender cross-contamination between multiple people.
Feature: Identifies high-priority targets from an offline watchlist folder.
Keyboard Toggles: [f] Face ID | [p] Privacy | [o] Posture | [r] Register | [q] Quit
"""

import cv2
import time
import os
import sys
import numpy as np
from ultralytics import YOLO

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from ag_vision.utils.config_utils import load_config
from ag_vision.utils.camera import init_camera
from ag_vision.engine_tracked import TrackedViTEngine
from ag_vision.utils.download_utils import get_model_path
from ag_vision.watchlist import Watchlist

# Optional modules (graceful fallback)
try:
    from ag_vision.face_registry import FaceRegistry
    HAS_FACE_ID = True
except ImportError:
    HAS_FACE_ID = False
    print("[!] face_recognition not available. Face ID disabled.")

try:
    from ag_vision.posture_coach import PostureCoach
    HAS_POSTURE = True
except ImportError:
    HAS_POSTURE = False
    print("[!] mediapipe not available. Posture Coach disabled.")


def draw_hud(frame, fps, modules, active_tracks, watchlist_size):
    """Draw the heads-up display with module status and track count."""
    h, w = frame.shape[:2]

    # Top bar
    thickness = 65
    cv2.rectangle(frame, (0, 0), (w, thickness), (0, 0, 0), -1)
    cv2.putText(frame, f"ANTIGRAVITY V9 WATCHLIST - FPS: {int(fps)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Module status indicators
    status_y = 50
    x = 10
    for key, (label, active) in modules.items():
        color = (0, 255, 0) if active else (80, 80, 80)
        text = f"[{key}] {label}"
        cv2.putText(frame, text, (x, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        x += 160

    # Track & Watchlist indicator
    track_text = f"Tracking: {active_tracks} | Watching: {watchlist_size} target(s)"
    cv2.putText(frame, track_text, (w - 350, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def main():
    print("==================================================")
    print("  🚨 ANTIGRAVITY V9 - MOST WANTED WATCHLIST 🚨")
    print("==================================================")
    print("  FEATURE: Face compares against data/watchlist folder.")
    print("  Keyboard Controls:")
    print("    [f] Toggle Face ID")
    print("    [p] Toggle Privacy Shield")
    print("    [o] Toggle Posture Coach")
    print("    [r] Register Your Face (Terminal Input)")
    print("    [c] Clear Face Database")
    print("    [q] Quit")
    print("==================================================")

    config = load_config()

    # Core Models
    print("[*] Loading Core Models...")
    model_obj = YOLO(get_model_path("yolov8n.pt"))
    model_face = YOLO(get_model_path("yolov8n-face.pt"))

    # V8: TrackedViTEngine instead of AsyncViTEngine
    tracked_engine = TrackedViTEngine(get_model_path("vit_age_gender.onnx"))

    # V9: Watchlist
    watchlist = Watchlist()

    # Optional Modules
    face_registry = FaceRegistry() if HAS_FACE_ID else None
    posture_coach = PostureCoach() if HAS_POSTURE else None

    # Module Toggles
    modules = {
        "f": ["Face ID", HAS_FACE_ID],
        "p": ["Privacy", False],
        "o": ["Posture", HAS_POSTURE],
    }

    cap = init_camera()
    if not cap:
        return

    prev_time = 0
    last_submit = {}  # Per-ID submit throttle
    SUBMIT_INTERVAL = 0.4  # seconds between ViT submissions per person

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display = frame.copy()

        # Tick tracked engine (advance frame counter, purge stale IDs)
        tracked_engine.tick()

        # ---- MODULE: Posture Coach ----
        if modules["o"][1] and posture_coach:
            posture_result = posture_coach.analyze(frame)
            display = posture_coach.draw_skeleton(display, posture_result)

            if posture_result.get("alert"):
                cv2.rectangle(display, (0, h - 50), (w, h), (0, 0, 200), -1)
                cv2.putText(display, posture_result["alert"], (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ---- PIPELINE A: General Objects ----
        res_obj = model_obj.predict(source=frame, conf=0.3, verbose=False)
        if len(res_obj) > 0 and len(res_obj[0].boxes) > 0:
            for box in res_obj[0].boxes:
                cls_id = int(box.cls[0])
                label = model_obj.names[cls_id]
                if label == "person":
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(display, (x1, y1), (x2, y2), (180, 180, 180), 1)
                cv2.putText(display, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # ---- PIPELINE B: Faces (TRACKED) ----
        # V8 KEY CHANGE: .track() instead of .predict()
        res_face = model_face.track(
            source=frame, conf=0.5, persist=True, verbose=False
        )

        active_tracks = 0

        if len(res_face) > 0 and res_face[0].boxes is not None and len(res_face[0].boxes) > 0:
            boxes = res_face[0].boxes

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if face_crop.size == 0:
                    continue

                # Get stable track ID from YOLOv8 tracker
                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = 1000 + i  # Fallback ID

                active_tracks += 1

                # Throttled submission per track ID
                now = time.time()
                if (now - last_submit.get(track_id, 0)) > SUBMIT_INTERVAL:
                    tracked_engine.submit(track_id, face_crop)
                    last_submit[track_id] = now

                # Get per-ID result (never cross-contaminated)
                vit_res = tracked_engine.get_result(track_id)
                age_str = str(vit_res["age"]) if vit_res["age"] != "--" else "..."
                gen_str = vit_res["gender"]
                
                # ---- MODULE: Watchlist ----
                watch_name, watch_score = watchlist.compare(face_crop)
                is_wanted = (watch_score > 50.0)

                # ---- MODULE: Face ID ----
                person_name = None
                is_known = False
                distance = 1.0
                if modules["f"][1] and face_registry:
                    person_name, distance = face_registry.identify(face_crop)
                    is_known = (person_name != "Unknown")

                # ---- MODULE: Privacy Shield ----
                if modules["p"][1] and not is_known and not is_wanted:
                    blurred = cv2.GaussianBlur(face_crop, (99, 99), 30)
                    display[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = blurred
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display, "MASKED", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # Normal display with track ID indicator
                    if is_wanted:
                        box_color = (0, 0, 255) # RED
                    elif is_known:
                        box_color = (0, 255, 255) # YELLOW
                    else:
                        box_color = (255, 165, 0) # ORANGE
                        
                    cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 3 if is_wanted else 2)

                    # Track ID badge
                    bg_badge = box_color
                    fg_badge = (0, 0, 0) if not is_wanted else (255, 255, 255)
                    cv2.circle(display, (x1 - 8, y1 - 8), 10, bg_badge, -1)
                    cv2.putText(display, str(track_id), (x1 - 14, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, fg_badge, 1)

                    # Label
                    label_parts = []
                    if is_wanted:
                        label_parts.append(f"🚨 ALERT: {watch_name} ({watch_score}%)")
                    elif person_name and is_known:
                        label_parts.append(f"{person_name} ({distance:.2f})")
                    else:
                        label_parts.append(f"ID:{track_id}")

                    label_parts.append(f"{gen_str}, {age_str}")
                    label = " | ".join(label_parts)

                    # Background for text
                    bg_text = (0, 0, 0)
                    if is_wanted:
                        # Flash background between red and black if highly matched
                        bg_text = (0, 0, 255) if int(time.time() * 4) % 2 == 0 else (0, 0, 150)
                        
                    cv2.rectangle(display, (x1, y1 - 45),
                                  (x1 + len(label) * 11, y1), bg_text, -1)
                    cv2.putText(display, label, (x1 + 5, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_color if not is_wanted else (255, 255, 255), 2)

        # FPS
        curr = time.time()
        fps = 1 / (curr - prev_time) if (curr - prev_time) > 0 else 0
        prev_time = curr

        # HUD
        draw_hud(display, fps, modules, active_tracks, len(watchlist.targets))

        cv2.imshow("Antigravity V9 WATCHLIST", display)

        # Keyboard handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f') and HAS_FACE_ID:
            modules["f"][1] = not modules["f"][1]
            print(f"[Toggle] Face ID: {'ON' if modules['f'][1] else 'OFF'}")
        elif key == ord('p'):
            modules["p"][1] = not modules["p"][1]
            print(f"[Toggle] Privacy: {'ON' if modules['p'][1] else 'OFF'}")
        elif key == ord('o') and HAS_POSTURE:
            modules["o"][1] = not modules["o"][1]
            print(f"[Toggle] Posture: {'ON' if modules['o'][1] else 'OFF'}")
        elif key == ord('c') and HAS_FACE_ID and face_registry:
            confirm = input(
                "Voulez-vous vraiment supprimer tous les visages (Face ID) ? (y/n) : "
            ).strip().lower()
            if confirm == 'y':
                face_registry.clear()
        elif key == ord('r') and HAS_FACE_ID and face_registry:
            ret2, reg_frame = cap.read()
            if ret2:
                reg_frame = cv2.flip(reg_frame, 1)
                print("\n[!] PASSEZ SUR LE TERMINAL pour taper le nom.")
                name = input("Entrez le prénom à enregistrer (Face ID régulier) : ").strip()
                if name:
                    face_registry.register(name, reg_frame)

    cap.release()
    cv2.destroyAllWindows()
    tracked_engine.stop()
    print("[+] Session V9 terminée.")


if __name__ == "__main__":
    main()
