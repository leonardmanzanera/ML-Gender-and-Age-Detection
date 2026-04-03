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
from ag_vision.logger import PerformanceLogger
from ag_vision.smoother import TemporalSmoother
from ag_vision.utils.camera import init_camera
from ag_vision.engine_async import AsyncViTEngine
from ag_vision.utils.download_utils import get_model_path

def main():
    print("[*] Starting Pipeline V3.1: YOLOv8 + Async ViT...")
    config = load_config()
    yolo_model = YOLO(get_model_path("yolov8n-face.pt"))
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
        
        results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
        fps = 1 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0
        prev_time = time.time()
        
        cv2.putText(frame, f"V3.1 ViT Async - FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                if face_crop.size > 0 and (time.time() - last_submit) > 0.5:
                    async_vit.submit(face_crop)
                    last_submit = time.time()
                
                res = async_vit.get_latest()
                if res["age"] != "--":
                    s_age, _, s_gen, _ = smoother.update_and_get("F0", res["age"], 1.0, res["gender"], res["gender_prob"], is_regression=True)
                else:
                    s_age, s_gen = "--", "Loading"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"{s_gen}, {s_age}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("V3.1 Async", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()
    async_vit.stop()

if __name__ == "__main__":
    main()
