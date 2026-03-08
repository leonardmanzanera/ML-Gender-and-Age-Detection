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
from ag_vision.logger import PerformanceLogger
from ag_vision.smoother import TemporalSmoother
from ag_vision.camera import init_camera
from ag_vision.engine_async import AsyncViTEngine
from ag_vision.utils.download_utils import get_model_path

def main():
    print("==================================================")
    print("  🚀 ANTIGRAVITY VISION v5.0 (UNIFIED VISION) 🚀")
    print("==================================================")
    
    config = load_config()
    ui = config["ui"]
    
    # Models
    print("[*] Loading Models (Hybrid Mode)...")
    model_obj = YOLO("yolov8n.pt")
    model_face = YOLO(get_model_path("yolov8n-face.pt"))
    async_vit = AsyncViTEngine(get_model_path("vit_age_gender.onnx"))
    
    smoother = TemporalSmoother()
    cap = init_camera()
    
    prev_time = 0
    last_submit = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display_frame = frame.copy()
        
        # 1. Pipeline A: General Objects
        res_obj = model_obj.predict(source=frame, conf=0.3, verbose=False)
        
        # 2. Pipeline B: Faces
        res_face = model_face.predict(source=frame, conf=0.5, verbose=False)
        
        fps = 1 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0
        prev_time = time.time()
        
        # UI
        cv2.putText(display_frame, f"V5 Unified Vision - FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw Objects
        if len(res_obj) > 0 and len(res_obj[0].boxes) > 0:
            for box in res_obj[0].boxes:
                cls_id = int(box.cls[0])
                label = model_obj.names[cls_id]
                if label == "person": continue # Skip person box if we draw face box
                
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
                cv2.putText(display_frame, f"{label}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw Faces + Age/Gender
        if len(res_face) > 0 and len(res_face[0].boxes) > 0:
            for box in res_face[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                if face_crop.size > 0 and (time.time() - last_submit) > 0.4:
                    async_vit.submit(face_crop)
                    last_submit = time.time()
                
                res = async_vit.get_latest()
                s_age = res["age"] if res["age"] != "--" else "..."
                s_gen = res["gender"]
                
                # UI Styling for person
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.rectangle(display_frame, (x1, y1-40), (x1+150, y1), (0, 0, 0), -1)
                cv2.putText(display_frame, f"{s_gen}, {s_age}", (x1 + 5, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Antigravity UNIFIED V5", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()
    async_vit.stop()

if __name__ == "__main__":
    main()
