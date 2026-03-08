import cv2
import time
import os
import sys
from ultralytics import YOLO

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from ag_vision.config_utils import load_config
from ag_vision.logger import PerformanceLogger
from ag_vision.smoother import TemporalSmoother
from ag_vision.camera import init_camera
from ag_vision.utils.download_utils import get_model_path

def compile_caffe_network(prototxt, caffemodel):
    p_path, c_path = get_model_path(prototxt), get_model_path(caffemodel)
    net = cv2.dnn.readNetFromCaffe(p_path, c_path)
    return net

def main():
    print("[*] Starting Pipeline V2: YOLOv8 Face + Caffe CNN...")
    config = load_config()
    ui = config["ui"]
    
    yolo_model = YOLO(get_model_path("yolov8n-face.pt"))
    net_age = compile_caffe_network(config["model"]["age"]["prototxt"], config["model"]["age"]["caffemodel"])
    net_gender = compile_caffe_network(config["model"]["gender"]["prototxt"], config["model"]["gender"]["caffemodel"])
    
    logger = PerformanceLogger()
    smoother = TemporalSmoother()
    cap = init_camera()
    
    AGE_BINS = config["model"]["age"]["bins"]
    GENDER_CATS = config["model"]["gender"]["categories"]
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        t0 = time.time()
        results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
        t_det = (time.time() - t0) * 1000
        
        fps = 1 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0
        prev_time = time.time()
        
        cv2.putText(frame, f"V2 YOLO Face - FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if face_crop.size == 0: continue
                
                face_blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), (78.4, 87.7, 114.8), swapRB=False)
                
                net_gender.setInput(face_blob)
                t1 = time.time()
                g_preds = net_gender.forward()
                t_gen = (time.time() - t1) * 1000
                
                net_age.setInput(face_blob)
                t2 = time.time()
                a_preds = net_age.forward()
                t_age = (time.time() - t2) * 1000
                
                s_age, _, s_gen, _ = smoother.update_and_get("F0", AGE_BINS[a_preds.argmax()], 1.0, GENDER_CATS[g_preds.argmax()], 1.0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{s_gen}, {s_age}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("V2 Refactored", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
