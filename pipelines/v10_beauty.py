"""
AG Vision - V10 Aesthetic Pipeline (Golden Ratio & Symmetry)
Scores facial beauty using mathematical proportions (Phi = 1.618)
and bilateral symmetry analysis via MediaPipe Face Mesh.
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
from ag_vision.aesthetic import AestheticEngine

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


# ─── Color palette ───
GOLD       = (0, 215, 255)   # BGR gold
GOLD_DARK  = (0, 140, 180)
SILVER     = (192, 192, 192)
BRONZE     = (0, 100, 180)
WHITE      = (255, 255, 255)
BLACK      = (0, 0, 0)


def score_color(score):
    """Return a color based on the score tier."""
    if score >= 8.0:
        return GOLD
    elif score >= 6.0:
        return SILVER
    elif score >= 4.0:
        return BRONZE
    else:
        return (100, 100, 100)


def draw_radar_chart(display, x1, y1, x2, y2, aes_result):
    """Draw a 5-axis radar chart next to the face."""
    import math
    gauge_x = x2 + 10
    gauge_y = y1
    gauge_w = 160
    gauge_h = 200
    
    golden_score = aes_result["golden_score"]
    color = score_color(golden_score)

    # Background panel
    cv2.rectangle(display, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h),
                  (20, 20, 20), -1)
    cv2.rectangle(display, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h),
                  color, 1)

    # Title
    cv2.putText(display, "AESTHETIC RADAR", (gauge_x + 5, gauge_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Big score number
    score_text = f"{golden_score:.1f}"
    cv2.putText(display, score_text, (gauge_x + 10, gauge_y + 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
    cv2.putText(display, "/ 10", (gauge_x + 85, gauge_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Radar
    cx = gauge_x + 80
    cy = gauge_y + 130
    radius = 50

    # 5 axes: Sym, Phi, Regard, Harmonie, Teint
    labels = ["Sym", "Phi", "Regard", "Harmonie", "Teint"]
    keys = ["Symmetry", "Phi", "Regard", "Harmonie", "Teint"]
    radar_data = aes_result["radar"]
    
    # Draw axes
    angles = [math.pi/2 + (2 * math.pi * i / 5) for i in range(5)]
    points = []
    
    for i in range(5):
        # Background polygon
        bx = int(cx + radius * math.cos(angles[i]))
        by = int(cy - radius * math.sin(angles[i]))
        cv2.line(display, (cx, cy), (bx, by), (60, 60, 60), 1)
        
        # Label
        lx = int(cx + (radius+15) * math.cos(angles[i])) - 15
        ly = int(cy - (radius+10) * math.sin(angles[i])) + 5
        cv2.putText(display, labels[i], (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        # Value point
        val = radar_data.get(keys[i], 5.0) / 10.0
        px = int(cx + (radius * val) * math.cos(angles[i]))
        py = int(cy - (radius * val) * math.sin(angles[i]))
        points.append([px, py])

    # Draw data polygon with alpha blending
    overlay = display.copy()
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
    cv2.polylines(display, [pts], isClosed=True, color=color, thickness=2)


def draw_golden_mask(display, landmarks, crop_x, crop_y, crop_w, crop_h):
    """Draws a translucent golden point cloud representing the Face Mesh."""
    overlay = display.copy()
    for lm in landmarks:
        px = int(lm.x * crop_w) + crop_x
        py = int(lm.y * crop_h) + crop_y
        cv2.circle(overlay, (px, py), 1, GOLD, -1)
    
    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)


def draw_hud(frame, fps, modules, active_tracks):
    """Draw the heads-up display with module status and track count."""
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 65), (0, 0, 0), -1)
    cv2.putText(frame, f"ANTIGRAVITY V10 AESTHETICS - FPS: {int(fps)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GOLD, 2)

    # Module status indicators
    status_y = 50
    x = 10
    for key, (label, active) in modules.items():
        color = (0, 255, 0) if active else (80, 80, 80)
        text = f"[{key}] {label}"
        cv2.putText(frame, text, (x, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        x += 160

    # Track count
    track_text = f"Tracking: {active_tracks} person(s)"
    cv2.putText(frame, track_text, (w - 220, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GOLD, 1)


def main():
    print("==================================================")
    print("  ✨ ANTIGRAVITY V10 - AESTHETIC ANALYSIS ✨")
    print("==================================================")
    print("  FEATURE: Golden Ratio (Phi) + Symmetry scoring.")
    print("  Keyboard Controls:")
    print("    [f] Toggle Face ID")
    print("    [o] Toggle Posture Coach")
    print("    [m] Toggle 3D Golden Mask")
    print("    [p] Toggle Privacy Shield")
    print("    [r] Register Your Face (Terminal Input)")
    print("    [c] Clear Face Database")
    print("    [q] Quit")
    print("==================================================")

    config = load_config()

    # Core Models
    print("[*] Loading Core Models...")
    model_obj = YOLO(get_model_path("yolov8n.pt"))
    model_face = YOLO(get_model_path("yolov8n-face.pt"))

    # V8: TrackedViTEngine
    tracked_engine = TrackedViTEngine(get_model_path("vit_age_gender.onnx"))

    # V10: Aesthetic Engine
    aesthetic = AestheticEngine()

    # Optional Modules
    face_registry = FaceRegistry() if HAS_FACE_ID else None
    posture_coach = PostureCoach() if HAS_POSTURE else None

    # Module Toggles
    modules = {
        "m": ["Golden Mask", False],
        "o": ["Posture", HAS_POSTURE],
        "f": ["Face ID", HAS_FACE_ID],
        "p": ["Privacy", False],
    }

    # Auto-Capture Best Shot Setup
    best_scores = {}
    best_shot_alert = 0
    BEST_SHOT_DIR = os.path.join(PROJECT_ROOT, "data", "best_shots")
    os.makedirs(BEST_SHOT_DIR, exist_ok=True)

    cap = init_camera()
    if not cap:
        return

    prev_time = 0
    last_submit = {}
    SUBMIT_INTERVAL = 0.4

    # Cache aesthetic results per track ID to avoid computing every frame
    aesthetic_cache = {}
    aesthetic_last_update = {}
    AESTHETIC_INTERVAL = 0.8  # seconds between aesthetic recalculations

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display = frame.copy()

        tracked_engine.tick()

        # ---- MODULE: Posture Coach ----
        if modules["o"][1] and posture_coach:
            posture_result = posture_coach.analyze(frame)
            display = posture_coach.draw_skeleton(display, posture_result)
            if posture_result.get("alert"):
                cv2.rectangle(display, (0, h - 50), (w, h), (0, 0, 200), -1)
                cv2.putText(display, posture_result["alert"], (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        # ---- PIPELINE A: General Objects ----
        res_obj = model_obj.predict(source=frame, conf=0.3, verbose=False)
        if len(res_obj) > 0 and len(res_obj[0].boxes) > 0:
            for box in res_obj[0].boxes:
                cls_id = int(box.cls[0])
                label = model_obj.names[cls_id]
                if label == "person":
                    continue
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                cv2.rectangle(display, (bx1, by1), (bx2, by2), (180, 180, 180), 1)
                cv2.putText(display, label, (bx1, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # ---- PIPELINE B: Faces (TRACKED) ----
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

                # Track ID
                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = 1000 + i

                active_tracks += 1

                # Throttled ViT submission
                now = time.time()
                if (now - last_submit.get(track_id, 0)) > SUBMIT_INTERVAL:
                    tracked_engine.submit(track_id, face_crop)
                    last_submit[track_id] = now

                # Get per-ID result
                vit_res = tracked_engine.get_result(track_id)
                age_str = str(vit_res["age"]) if vit_res["age"] != "--" else "..."
                gen_str = vit_res["gender"]

                # ---- MODULE: Aesthetic Engine ----
                aes_result = aesthetic_cache.get(track_id)
                if (now - aesthetic_last_update.get(track_id, 0)) > AESTHETIC_INTERVAL:
                    new_result = aesthetic.analyze(face_crop)
                    if new_result is not None:
                        aes_result = new_result
                        aesthetic_cache[track_id] = aes_result
                        
                        # --- Best Shot Capture ---
                        score = aes_result["golden_score"]
                        if score >= 8.0:
                            if track_id not in best_scores or score > best_scores[track_id]:
                                best_scores[track_id] = score
                                # Save the frame!
                                filename = f"best_shot_ID{track_id}_{score:.1f}_{int(now)}.jpg"
                                cv2.imwrite(os.path.join(BEST_SHOT_DIR, filename), frame)
                                best_shot_alert = now
                                print(f" [📸] BEST PROFILE CAPTURED! Score: {score:.1f}")

                    aesthetic_last_update[track_id] = now

                # ---- MODULE: Face ID ----
                person_name = None
                is_known = False
                distance = 1.0
                if modules["f"][1] and face_registry:
                    person_name, distance = face_registry.identify(face_crop)
                    is_known = (person_name != "Unknown")

                # ---- MODULE: Privacy Shield ----
                if modules["p"][1] and not is_known:
                    blurred = cv2.GaussianBlur(face_crop, (99, 99), 30)
                    display[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = blurred
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display, "MASKED", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # Determine box color from aesthetic score
                    golden = aes_result["golden_score"] if aes_result else 0.0
                    box_color = score_color(golden) if aes_result else (255, 165, 0)

                    if is_known:
                        box_color = (0, 255, 255)

                    cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)

                    # Track ID badge
                    cv2.circle(display, (x1 - 8, y1 - 8), 10, box_color, -1)
                    cv2.putText(display, str(track_id), (x1 - 14, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, BLACK, 1)

                    # Label
                    label_parts = []
                    if person_name and is_known:
                        label_parts.append(f"{person_name} ({distance:.2f})")
                    else:
                        label_parts.append(f"ID:{track_id}")
                    label_parts.append(f"{gen_str}, {age_str}")

                    if aes_result:
                        label_parts.append(f"GOLDEN: {golden:.1f}/10")

                    label = " | ".join(label_parts)

                    # Background for text
                    cv2.rectangle(display, (x1, y1 - 45),
                                  (x1 + len(label) * 10 + 10, y1), BLACK, -1)
                    cv2.putText(display, label, (x1 + 5, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

                    # Draw Golden Mask Overley if active
                    if modules["m"][1] and aes_result:
                        draw_golden_mask(
                            display, aes_result["raw_landmarks"],
                            x1, y1, (x2-x1), (y2-y1)
                        )

                    # Draw Radar Chart UI
                    if aes_result:
                        draw_radar_chart(display, x1, y1, x2, y2, aes_result)

        # FPS
        curr = time.time()
        fps = 1 / (curr - prev_time) if (curr - prev_time) > 0 else 0
        prev_time = curr

        # HUD
        draw_hud(display, fps, modules, active_tracks)
        
        # Best Shot Notification Flash
        if (curr - best_shot_alert) < 2.0:
            cv2.rectangle(display, (0, 0), (w, h), (0, 215, 255), 4)
            cv2.putText(display, "[📸] BEST SHOT CAPTURED!", (w - 280, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GOLD, 2)

        cv2.imshow("Antigravity V10 AESTHETICS", display)

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
        elif key == ord('m'):
            modules["m"][1] = not modules["m"][1]
            print(f"[Toggle] Mask Overlay: {'ON' if modules['m'][1] else 'OFF'}")
        elif key == ord('o') and HAS_POSTURE:
            modules["o"][1] = not modules["o"][1]
            print(f"[Toggle] Posture: {'ON' if modules['o'][1] else 'OFF'}")
        elif key == ord('c') and HAS_FACE_ID and face_registry:
            confirm = input(
                "Voulez-vous vraiment supprimer tous les visages ? (y/n) : "
            ).strip().lower()
            if confirm == 'y':
                face_registry.clear()
        elif key == ord('r') and HAS_FACE_ID and face_registry:
            ret2, reg_frame = cap.read()
            if ret2:
                reg_frame = cv2.flip(reg_frame, 1)
                print("\n[!] PASSEZ SUR LE TERMINAL pour taper le nom.")
                name = input("Entrez le prénom à enregistrer : ").strip()
                if name:
                    face_registry.register(name, reg_frame)

    cap.release()
    cv2.destroyAllWindows()
    tracked_engine.stop()
    print("[+] Session V10 terminée.")


if __name__ == "__main__":
    main()
