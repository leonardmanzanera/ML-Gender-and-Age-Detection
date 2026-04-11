import cv2
import time
import os
import sys
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from ag_vision.utils.config_utils import load_config
from ag_vision.utils.camera import init_camera
from ag_vision.utils.download_utils import get_model_path

class SyncViTEngine:
    """True synchronous version for V3 baseline demonstration."""
    def __init__(self, vit_model_path):
        self.vit_sess = ort.InferenceSession(vit_model_path, providers=['CPUExecutionProvider'])
        self.vit_input = self.vit_sess.get_inputs()[0].name
        
        models_dir = os.path.dirname(vit_model_path)
        g_proto = os.path.join(models_dir, "gender_deploy.prototxt")
        g_model = os.path.join(models_dir, "gender_net.caffemodel")
        self.gender_net = cv2.dnn.readNet(g_model, g_proto) if os.path.exists(g_proto) else None
        self.GENDER_LIST = ['Male', 'Female']

    def process(self, face_crop):
        try:
            # Preprocess ViT
            img = cv2.resize(face_crop, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
            
            # Run ViT (Blocking)
            preds = self.vit_sess.run(None, {self.vit_input: img})[0][0]
            age = preds[0]
            
            # Run Caffe Gender (Blocking)
            if self.gender_net:
                blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                self.gender_net.setInput(blob)
                g_preds = self.gender_net.forward()
                gen_idx = g_preds[0].argmax()
                gender = self.GENDER_LIST[gen_idx]
            else:
                prob = 1 / (1 + np.exp(-preds[1]))
                gender = "Female" if prob > 0.5 else "Male"
                
            return {"age": age, "gender": gender}
        except Exception:
            return {"age": "--", "gender": "Unknown"}

def main():
    print("[*] Starting Pipeline V3: YOLOv8 + SYNCHRONOUS ViT/ONNX...")
    load_config()

    yolo_model = YOLO(get_model_path("yolov8n-face.pt"))
    sync_vit = SyncViTEngine(get_model_path("vit_age_gender.onnx"))

    cap = init_camera()
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # YOLO detection
        results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"V3 ViT SYNC (Blocking) - FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                if face_crop.size == 0:
                    continue

                # BLOCKING CALL - This is why it will be slow!
                res = sync_vit.process(face_crop)
                
                s_age = f"{res['age']:.1f}" if res["age"] != "--" else "..."
                s_gen = res["gender"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{s_gen}, {s_age}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("V3 Synchronous", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
