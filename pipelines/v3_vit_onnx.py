import cv2
import time
import os
import sys
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from ag_vision.utils.config_utils import load_config
from ag_vision.logger import PerformanceLogger
from ag_vision.smoother import TemporalSmoother
from ag_vision.utils.camera import init_camera
from ag_vision.utils.download_utils import get_model_path

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess_for_vit(face_crop):
    img = cv2.resize(face_crop, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def main():
    print("[*] Starting Pipeline V3: YOLOv8 + Synchronous ViT/ONNX...")
    config = load_config()
    ui = config["ui"]
    
    yolo_model = YOLO(get_model_path("yolov8n-face.pt"))
    sess = ort.InferenceSession(get_model_path("vit_age_gender.onnx"), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    logger = PerformanceLogger()
    smoother = TemporalSmoother()
    cap = init_camera()
    
    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
        fps = 1 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0
        prev_time = time.time()
        
        cv2.putText(frame, f"V3 ViT Sync - FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if face_crop.size == 0: continue
                
                input_tensor = preprocess_for_vit(face_crop)
                preds = sess.run(None, {input_name: input_tensor})[0][0]
                
                raw_age = preds[0]
                gender_prob = sigmoid(preds[1])
                # Model maps: sigmoid > 0.5 = Male, < 0.5 = Female
                gen_str = "Male" if gender_prob > 0.5 else "Female"
                gen_conf = gender_prob if gender_prob > 0.5 else 1.0 - gender_prob
                
                s_age, _, s_gen, _ = smoother.update_and_get("F0", raw_age, 1.0, gen_str, gen_conf, is_regression=True)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"{s_gen}, {s_age} yrs", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.imshow("V3 Sync", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
