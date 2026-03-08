"""
AG Vision - Ultimate Pipeline
Consolidates: Objects + Async ViT + Face Recognition + Posture Coach + Privacy Shield
Keyboard Toggles: [f] Face ID | [p] Privacy | [o] Posture | [r] Register Face | [q] Quit
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

from ag_vision.config_utils import load_config
from ag_vision.smoother import TemporalSmoother
from ag_vision.camera import init_camera
from ag_vision.engine_async import AsyncViTEngine
from ag_vision.utils.download_utils import get_model_path

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


def draw_hud(frame, fps, modules):
    """Draw the heads-up display with module status."""
    h, w = frame.shape[:2]
    
    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 65), (0, 0, 0), -1)
    cv2.putText(frame, f"ANTIGRAVITY ULTIMATE - FPS: {int(fps)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Module status indicators
    status_y = 50
    x = 10
    for key, (label, active) in modules.items():
        color = (0, 255, 0) if active else (80, 80, 80)
        text = f"[{key}] {label}"
        cv2.putText(frame, text, (x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        x += 160


def main():
    print("==================================================")
    print("  🧠 ANTIGRAVITY VISION - ULTIMATE EDITION 🧠")
    print("==================================================")
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
    model_obj = YOLO("yolov8n.pt")
    model_face = YOLO(get_model_path("yolov8n-face.pt"))
    async_vit = AsyncViTEngine(get_model_path("vit_age_gender.onnx"))
    smoother = TemporalSmoother()
    
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
    if not cap: return
    
    prev_time = 0
    last_vit_submit = 0
    registering = False

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display = frame.copy()
        
        # ---- MODULE: Posture Coach ----
        if modules["o"][1] and posture_coach:
            posture_result = posture_coach.analyze(frame)
            display = posture_coach.draw_skeleton(display, posture_result)
            
            if posture_result.get("alert"):
                # Draw alert banner
                cv2.rectangle(display, (0, h - 50), (w, h), (0, 0, 200), -1)
                cv2.putText(display, posture_result["alert"], (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ---- PIPELINE A: General Objects ----
        res_obj = model_obj.predict(source=frame, conf=0.3, verbose=False)
        if len(res_obj) > 0 and len(res_obj[0].boxes) > 0:
            for box in res_obj[0].boxes:
                cls_id = int(box.cls[0])
                label = model_obj.names[cls_id]
                if label == "person": continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(display, (x1, y1), (x2, y2), (180, 180, 180), 1)
                cv2.putText(display, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        
        # ---- PIPELINE B: Faces ----
        res_face = model_face.predict(source=frame, conf=0.5, verbose=False)
        if len(res_face) > 0 and len(res_face[0].boxes) > 0:
            for box in res_face[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if face_crop.size == 0: continue
                
                # Async ViT (Age/Gender)
                if (time.time() - last_vit_submit) > 0.4:
                    async_vit.submit(face_crop)
                    last_vit_submit = time.time()
                
                vit_res = async_vit.get_latest()
                age_str = str(vit_res["age"]) if vit_res["age"] != "--" else "..."
                gen_str = vit_res["gender"]
                
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
                    # Normal display
                    box_color = (0, 255, 255) if is_known else (255, 165, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Label
                    label_parts = []
                    if person_name and is_known:
                        label_parts.append(f"👋 {person_name} ({distance:.2f})")
                    else:
                        label_parts.append(f"({distance:.2f})")
                        
                    label_parts.append(f"{gen_str}, {age_str}")
                    label = " | ".join(label_parts)
                    
                    # Background for text
                    cv2.rectangle(display, (x1, y1 - 45), (x1 + len(label) * 12, y1), (0, 0, 0), -1)
                    cv2.putText(display, label, (x1 + 5, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
        
        # FPS
        curr = time.time()
        fps = 1 / (curr - prev_time) if (curr - prev_time) > 0 else 0
        prev_time = curr
        
        # HUD
        draw_hud(display, fps, modules)
        
        # Registration mode indicator
        if registering:
            cv2.rectangle(display, (0, 65), (w, 95), (0, 100, 255), -1)
            cv2.putText(display, "MODE ENREGISTREMENT: Tapez votre nom dans le terminal", 
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Antigravity ULTIMATE", display)
        
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
            confirm = input("Voulez-vous vraiment supprimer tous les visages ? (y/n) : ").strip().lower()
            if confirm == 'y':
                face_registry.clear()
        elif key == ord('r') and HAS_FACE_ID and face_registry:
            # Capture current frame for registration
            ret2, reg_frame = cap.read()
            if ret2:
                reg_frame = cv2.flip(reg_frame, 1)
                print("\n[!] PASSEZ SUR LE TERMINAL pour taper le nom.")
                name = input("Entrez le prénom à enregistrer : ").strip()
                if name:
                    face_registry.register(name, reg_frame)
    
    cap.release()
    cv2.destroyAllWindows()
    async_vit.stop()
    print("[+] Session terminée.")

if __name__ == "__main__":
    main()
